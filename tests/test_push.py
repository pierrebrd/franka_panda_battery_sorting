"""Hardware-free tests of push_hot_objects.py: marker mapping, push planning and the camera/robot link.

Run with: .venv/bin/python -m unittest discover -s tests
"""

import tempfile
import unittest

import cv2
import numpy as np

from push_hot_objects import (FINGER_HALF_WIDTH, MARKER_CORNERS, MAX_REACH, ReportServer, RobotLink, TableMapping,
                              edge_y, fetch_report, plan_push)
from test_pipeline import RGB_SIZE, FakeRgbCamera, add_disk
from thermal_table.app import STABLE_AGE, App
from thermal_table.calibration import Calibration
from thermal_table.cameras import SimulatedThermalCamera

# Inner marker corners where the real webcam sees them (from data/background.png)
PIXEL_CORNERS = {0: (815.8, 598.0), 1: (381.0, 606.2), 2: (833.0, 174.4), 3: (319.4, 182.2)}
ROBOT_TO_PIXELS = cv2.getPerspectiveTransform(np.float32([MARKER_CORNERS[i] for i in range(4)]),
                                              np.float32([PIXEL_CORNERS[i] for i in range(4)]))
MARKER_SIZE = 0.047


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
            self.assertTrue(mapping.update(image))
            probe = [[0.45, -0.15], [0.6, 0.0], [0.75, 0.2]]
            np.testing.assert_allclose(mapping.to_robot(robot_to_pixels(probe)), probe, atol=0.003)
            np.testing.assert_allclose(mapping.to_pixels(probe), robot_to_pixels(probe), atol=1.5)

            # Saved for the next start, and kept while a marker is hidden
            reloaded = TableMapping(f"{tmp}/markers.json", RGB_SIZE)
            np.testing.assert_allclose(reloaded.H, mapping.H)
            hidden = image.copy()
            u, v = np.round(PIXEL_CORNERS[2]).astype(int)
            cv2.rectangle(hidden, (u - 5, v - 60), (u + 60, v + 5), (230, 230, 230), -1)
            self.assertFalse(mapping.update(hidden))
            np.testing.assert_allclose(mapping.H, reloaded.H)


class PlanTests(unittest.TestCase):
    def test_pushes_towards_the_nearer_side_out_of_the_table_area(self):
        for y, side in ((0.08, 1), (-0.05, -1)):
            start, end = plan_push(disk(0.55, y), [])
            self.assertAlmostEqual(start[0], 0.55)
            self.assertAlmostEqual(end[0], 0.55)
            self.assertLess(side * start[1], side * y - 0.015 - FINGER_HALF_WIDTH)   # comes down beside the object
            # The object, in front of the fingers, ends past the detection area (marker centres)
            self.assertGreater(side * (end[1] - edge_y(end[0], side)) + FINGER_HALF_WIDTH, MARKER_SIZE / 2)

    def test_avoids_the_side_where_another_object_is_in_the_way(self):
        start, end = plan_push(disk(0.55, 0.06), [disk(0.55, 0.16, id=2)])
        self.assertLess(end[1], 0)
        start, end = plan_push(disk(0.55, 0.06), [disk(0.40, 0.16, id=2)])  # not in the way
        self.assertGreater(end[1], 0)

    def test_fingers_do_not_come_down_on_another_object(self):
        # Pushing towards the nearer side (+y) would need the fingers right where the other object is
        start, end = plan_push(disk(0.55, 0.10), [disk(0.55, 0.05, id=2)])
        self.assertLess(end[1], 0)

    def test_far_objects_are_pushed_diagonally_or_left(self):
        start, end = plan_push(disk(0.74, 0.0), [])
        self.assertLess(end[0], 0.74)
        self.assertLessEqual(max(np.hypot(*start), np.hypot(*end)), MAX_REACH + 1e-9)
        self.assertIsNone(plan_push(disk(0.80, 0.0), []))

    def test_edges_go_through_the_marker_corners(self):
        for i, side in ((0, -1), (1, 1), (2, -1), (3, 1)):
            self.assertAlmostEqual(edge_y(MARKER_CORNERS[i][0], side), MARKER_CORNERS[i][1])


class LinkTests(unittest.TestCase):
    def test_report_roundtrip(self):
        class Link:
            report = {"objects": [disk(0.5, 0.1)], "error": None}

        server = ReportServer(("127.0.0.1", 0), Link())
        try:
            import threading
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.assertEqual(fetch_report(server.server_address[1]), Link.report)
        finally:
            server.shutdown()
            server.server_close()

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
                self.assertIsNotNone(plan_push(objects[0], []))

                link.threshold = 60.0
                link.update(app.step())
                self.assertFalse(link.report["objects"][0]["hot"])

                app.on_key(ord("c"))   # e.g. a hot mug used for calibration must not be pushed
                link.update(app.step())
                self.assertEqual(link.report["objects"], [])
                self.assertIn("calibrate", link.report["error"])
            finally:
                link.server.shutdown()
                link.server.server_close()


if __name__ == "__main__":
    unittest.main()
