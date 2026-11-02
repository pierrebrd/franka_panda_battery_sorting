"""Hardware-free tests of push_hot_objects.py: marker mapping, grasp planning and the camera/robot link.

Run with: .venv/bin/python -m unittest discover -s tests
"""

import tempfile
import threading
import unittest
from pathlib import Path

import cv2
import numpy as np

from push_hot_objects import (MARKER_CORNERS, MAX_REACH, PLACE_PAST_EDGE, ReportServer, RobotLink, TableMapping,
                              edge_y, fetch_report, plan_grasp)
from test_pipeline import RGB_SIZE, FakeRgbCamera, add_disk
from thermal_table.app import STABLE_AGE, App
from thermal_table.calibration import Calibration
from thermal_table.cameras import SimulatedThermalCamera

# Inner marker corners where the real webcam sees them (from data/background.png)
PIXEL_CORNERS = {0: (815.8, 598.0), 1: (381.0, 606.2), 2: (833.0, 174.4), 3: (319.4, 182.2)}
ROBOT_TO_PIXELS = cv2.getPerspectiveTransform(np.float32([MARKER_CORNERS[i] for i in range(4)]),
                                              np.float32([PIXEL_CORNERS[i] for i in range(4)]))
MARKER_SIZE = 0.047
BACKGROUND = Path(__file__).resolve().parent.parent / "data" / "background.png"


def robot_to_pixels(points):
    return cv2.perspectiveTransform(np.float64(points).reshape(-1, 1, 2), ROBOT_TO_PIXELS).reshape(-1, 2)


def table_with_markers():
    """Webcam view of a white table with the four markers, through the true robot -> pixel mapping."""
    res, x_max, y_max = 0.001, 0.95, 0.35   # top view texture, 1 mm per pixel, oriented like the webcam image
    texture = np.full((700, 700, 3), 230, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    size = round(MARKER_SIZE / res)
    for i, (x, y) in MARKER_CORNERS.items():
        # The marker lies outward of its inner corner: towards the robot for 0/1, away for 2/3
        x_far = x if i in (0, 1) else x + MARKER_SIZE
        y_left = y + MARKER_SIZE if y > 0 else y
        u, v = round((y_max - y_left) / res), round((x_max - x_far) / res)
        texture[v:v + size, u:u + size] = cv2.aruco.generateImageMarker(dictionary, i, size)[..., None]
    texture_to_robot = np.array([[0, -res, x_max], [-res, 0, y_max], [0, 0, 1]])
    return cv2.warpPerspective(texture, ROBOT_TO_PIXELS @ texture_to_robot, RGB_SIZE, flags=cv2.INTER_AREA,
                               borderValue=(230, 230, 230))


def disk(x, y, radius=0.015, id=1, temp=50.0):
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    outline = np.stack([x + radius * np.cos(angles), y + radius * np.sin(angles)], axis=1)
    return {"id": id, "temp": temp, "hot": True, "center": [x, y], "outline": outline.tolist()}


class MappingTests(unittest.TestCase):
    def test_markers_give_robot_coordinates(self):
        image = table_with_markers()
        with tempfile.TemporaryDirectory() as tmp:
            mapping = TableMapping(f"{tmp}/markers.json", RGB_SIZE)
            self.assertFalse(mapping.ready)
            self.assertEqual(sorted(mapping.update(image)), [0, 1, 2, 3])
            probe = [[0.45, -0.15], [0.6, 0.0], [0.75, 0.2]]
            np.testing.assert_allclose(mapping.to_robot(robot_to_pixels(probe)), probe, atol=0.003)
            np.testing.assert_allclose(mapping.to_pixels(probe), robot_to_pixels(probe), atol=1.5)

            # Saved for the next start, and kept while a marker is hidden
            reloaded = TableMapping(f"{tmp}/markers.json", RGB_SIZE)
            np.testing.assert_allclose(reloaded.H, mapping.H)
            hidden = image.copy()
            u, v = np.round(PIXEL_CORNERS[2]).astype(int)
            cv2.rectangle(hidden, (u - 5, v - 60), (u + 60, v + 5), (230, 230, 230), -1)
            self.assertNotIn(2, mapping.update(hidden))
            np.testing.assert_allclose(mapping.H, reloaded.H)

    def test_markers_in_the_wrong_arrangement_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping = TableMapping(f"{tmp}/markers.json", RGB_SIZE)
            self.assertEqual(len(mapping.update(cv2.rotate(table_with_markers(), cv2.ROTATE_180))), 4)
            self.assertFalse(mapping.ready)

    @unittest.skipUnless(BACKGROUND.exists(), "no saved background")
    def test_real_webcam_background(self):
        """The saved empty-table image (downscaled and blurred by the app) still shows all four markers."""
        image = cv2.resize(cv2.imread(str(BACKGROUND)), RGB_SIZE, interpolation=cv2.INTER_CUBIC)
        with tempfile.TemporaryDirectory() as tmp:
            mapping = TableMapping(f"{tmp}/markers.json", RGB_SIZE)
            self.assertEqual(sorted(mapping.update(image)), [0, 1, 2, 3])
            self.assertTrue(mapping.ready)


def box(x, y, length, width, id=1, temp=50.0):
    """Rectangle object, length along x and width along y."""
    outline = [[x - length / 2, y - width / 2], [x + length / 2, y - width / 2],
               [x + length / 2, y + width / 2], [x - length / 2, y + width / 2]]
    return {"id": id, "temp": temp, "hot": True, "center": [x, y], "outline": outline}


class PlanTests(unittest.TestCase):
    def test_sets_down_beyond_the_nearer_side_of_the_table_area(self):
        for y, side in ((0.08, 1), (-0.05, -1)):
            target = disk(0.55, y)
            plan = plan_grasp(target, [])
            np.testing.assert_allclose(plan["grasp"], [0.55, y], atol=1e-3)
            self.assertAlmostEqual(plan["place"][0], 0.55)
            # The whole object, carried as gripped, ends past the edge of the table area
            outline = np.asarray(target["outline"]) - plan["grasp"] + plan["place"]
            self.assertGreaterEqual(np.min(side * (outline[:, 1] - edge_y(0.55, side))), PLACE_PAST_EDGE - 1e-9)

    def test_grips_across_the_narrowest_width(self):
        plan = plan_grasp(box(0.55, 0.0, 0.06, 0.02), [])   # long along x: the fingers close along y
        np.testing.assert_allclose(np.abs(plan["closing"]), [0, 1], atol=1e-9)
        self.assertAlmostEqual(plan["width"], 0.02)
        self.assertGreater(plan["opening"], plan["width"])

    def test_too_wide_objects_are_left(self):
        self.assertIsNone(plan_grasp(box(0.55, 0.0, 0.09, 0.09), []))

    def test_fingers_do_not_come_down_on_another_object(self):
        # Closing along y would put a finger on the neighbour, so the fingers close along x
        plan = plan_grasp(box(0.55, 0.0, 0.05, 0.04), [disk(0.55, 0.05, id=2)])
        np.testing.assert_allclose(np.abs(plan["closing"]), [1, 0], atol=1e-9)
        ring = [disk(0.55 + 0.045 * np.cos(a), 0.045 * np.sin(a), id=2 + i)
                for i, a in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False))]
        self.assertIsNone(plan_grasp(disk(0.55, 0.0), ring))   # no room for the fingers

    def test_far_objects_are_set_down_closer_and_unreachable_ones_left(self):
        plan = plan_grasp(disk(0.75, 0.0), [])
        self.assertLess(plan["place"][0], 0.75)
        self.assertLessEqual(max(np.hypot(*plan["grasp"]), np.hypot(*plan["place"])), MAX_REACH + 1e-9)
        self.assertIsNone(plan_grasp(disk(0.80, 0.0), []))

    def test_edges_go_through_the_marker_corners(self):
        for i, side in ((0, -1), (1, 1), (2, -1), (3, 1)):
            self.assertAlmostEqual(edge_y(MARKER_CORNERS[i][0], side), MARKER_CORNERS[i][1])


class LinkTests(unittest.TestCase):
    def test_report_roundtrip(self):
        class Link:
            report = {"objects": [disk(0.5, 0.1)], "error": None}

        server = ReportServer(("127.0.0.1", 0), Link())
        try:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.assertEqual(fetch_report(server.server_address[1]), Link.report)
        finally:
            server.shutdown()
            server.server_close()

    def test_camera_side_names_the_marker_it_cannot_read(self):
        image = table_with_markers()
        u, v = np.round(PIXEL_CORNERS[0]).astype(int)
        cv2.line(image, (u - 10, v - 10), (u + 60, v + 60), (20, 20, 20), 12)   # cable across marker 0
        rgb = FakeRgbCamera(image)
        with tempfile.TemporaryDirectory() as tmp:
            app = App(rgb, SimulatedThermalCamera(rgb).start(), tmp)
            link = RobotLink(app, TableMapping(f"{tmp}/markers.json", RGB_SIZE), 40.0, False, 0)
            try:
                link.update(app.step())
                self.assertEqual(link.report["error"], "the webcam cannot read marker 0 (bottom right)")
            finally:
                link.server.shutdown()
                link.server.server_close()

    def test_camera_side_reports_hot_object_in_robot_coordinates(self):
        """App + simulated thermal camera (dark = hot): a black disk is ~47 C."""
        rgb = FakeRgbCamera(table_with_markers())
        thermal = SimulatedThermalCamera(rgb).start()
        with tempfile.TemporaryDirectory() as tmp:
            app = App(rgb, thermal, tmp)
            app.calibration = Calibration(thermal.homography, RGB_SIZE, thermal.size)
            link = RobotLink(app, TableMapping(f"{tmp}/markers.json", RGB_SIZE), 40.0, False, 0)
            try:
                app.on_key(ord("b"))
                for _ in range(25):
                    link.update(app.step())
                center = np.round(robot_to_pixels([[0.6, 0.05]])[0]).astype(int)
                rgb.frame = add_disk(table_with_markers(), tuple(center), 25, (0, 0, 0))
                for _ in range(STABLE_AGE + 2):
                    link.update(app.step())
                self.assertIsNone(link.report["error"])
                objects = link.report["objects"]
                self.assertEqual(len(objects), 1)
                np.testing.assert_allclose(objects[0]["center"], [0.6, 0.05], atol=0.005)
                self.assertTrue(objects[0]["hot"])
                self.assertIsNotNone(plan_grasp(objects[0], []))

                link.threshold = 60.0
                link.update(app.step())
                self.assertFalse(link.report["objects"][0]["hot"])

                app.on_key(ord("c"))   # e.g. a hot mug used for calibration must not be picked
                link.update(app.step())
                self.assertEqual(link.report["objects"], [])
                self.assertIn("calibrate", link.report["error"])
            finally:
                link.server.shutdown()
                link.server.server_close()


if __name__ == "__main__":
    unittest.main()
