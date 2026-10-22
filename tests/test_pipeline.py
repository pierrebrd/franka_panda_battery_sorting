"""Hardware-free tests: synthetic table images and the simulated thermal camera.

Run with: .venv/bin/python -m unittest discover tests
"""

import tempfile
import unittest

import cv2
import numpy as np

from thermal_table.app import App
from thermal_table.calibration import Calibration, find_thermal_spot, fit_homography
from thermal_table.cameras import SimulatedThermalCamera, simulated_homography
from thermal_table.detection import ObjectDetector
from thermal_table.measure import measure, object_mask

RGB_SIZE = (1280, 720)
THERMAL_SIZE = (382, 288)


def table_image(seed=0, gain=(1.0, 1.0, 1.0)):
    """White table with a soft lighting gradient and sensor noise."""
    rng = np.random.default_rng(seed)
    w, h = RGB_SIZE
    gradient = np.linspace(0.92, 1.0, w, dtype=np.float32)[None, :, None]
    image = np.full((h, w, 3), 200, np.float32) * gradient * np.float32(gain)
    image += rng.normal(0, 2.0, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def add_disk(image, center, radius, color, shadow=True):
    out = image.copy()
    if shadow:  # soft shadow offset from the object, like a lamp would cast
        shade = np.zeros(out.shape[:2], np.float32)
        cv2.circle(shade, (center[0] + 18, center[1] + 12), radius, 1.0, -1)
        shade = cv2.GaussianBlur(shade, (31, 31), 0)[..., None]
        out = (out * (1 - 0.18 * shade)).astype(np.uint8)
    cv2.circle(out, center, radius, color, -1, cv2.LINE_AA)
    return out


class FakeRgbCamera:
    def __init__(self, frame):
        self.size = RGB_SIZE
        self.frame = frame
        self.count = 0

    def read(self):
        self.count += 1
        return self.frame, self.count


class CalibrationTests(unittest.TestCase):
    def test_fit_recovers_homography_and_rejects_outlier(self):
        H_true = simulated_homography(RGB_SIZE, THERMAL_SIZE)
        rgb = np.array([[x, y] for y in (180, 380, 600) for x in (330, 640, 940)], float)  # 3x3 grid
        cal = Calibration(H_true, RGB_SIZE, THERMAL_SIZE)
        thermal = cal.rgb_to_thermal(rgb) + np.random.default_rng(1).normal(0, 0.3, (len(rgb), 2))
        thermal[4] += (12, -9)  # one bad pair, in the middle
        H, rms, errors, inliers = fit_homography(rgb, thermal)
        self.assertFalse(inliers[4])
        self.assertTrue(inliers[[0, 1, 2, 3, 5, 6, 7, 8]].all())
        self.assertLess(rms, 1.0)
        fitted = Calibration(H, RGB_SIZE, THERMAL_SIZE)
        np.testing.assert_allclose(fitted.rgb_to_thermal([[640, 360]]), cal.rgb_to_thermal([[640, 360]]), atol=1.0)

    def test_save_load_roundtrip(self):
        cal = Calibration(simulated_homography(RGB_SIZE, THERMAL_SIZE), RGB_SIZE, THERMAL_SIZE,
                          [[1, 2]], [[3, 4]], 0.5)
        with tempfile.TemporaryDirectory() as tmp:
            cal.save(f"{tmp}/c.json")
            loaded = Calibration.load(f"{tmp}/c.json")
        np.testing.assert_allclose(loaded.H, cal.H)
        self.assertEqual(loaded.thermal_size, THERMAL_SIZE)

    def test_thermal_spot_finds_warm_and_cold_objects(self):
        temps = np.full((288, 382), 22.0, np.float32) + np.random.default_rng(2).normal(0, 0.1, (288, 382))
        for value in (40.0, 5.0):
            image = temps.copy()
            cv2.circle(image, (250, 100), 10, value, -1)
            spot, _ = find_thermal_spot(image)
            np.testing.assert_allclose(spot, (250, 100), atol=1.0)
        self.assertIsNone(find_thermal_spot(temps)[0])

    def test_thermal_spot_ignores_static_warm_things_with_reference(self):
        reference = np.full((288, 382), 22.0, np.float32)
        cv2.rectangle(reference, (300, 0), (381, 287), 35.0, -1)  # person / laptop next to the table
        image = reference + 0.4  # slight global drift
        cv2.circle(image, (120, 150), 8, 30.0, -1)
        self.assertGreater(find_thermal_spot(image)[0][0], 300)  # without reference: the big warm area wins
        np.testing.assert_allclose(find_thermal_spot(image, reference)[0], (120, 150), atol=1.0)
        self.assertIsNone(find_thermal_spot(image, reference, near=(300, 50), radius=30)[0])

    def test_degenerate_points_are_rejected(self):
        rgb = [[300, 200], [500, 300], [700, 400], [900, 500]]  # on a line
        with self.assertRaises(ValueError):
            fit_homography(rgb, [[10, 10], [100, 20], [200, 200], [20, 250]])
        spread = [[300, 200], [900, 200], [900, 600], [300, 600]]
        with self.assertRaises(ValueError):  # all thermal points in the same place
            fit_homography(spread, [[200, 100], [201, 100], [200, 101], [201, 101]])


class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.detector = ObjectDetector()
        self.detector.set_background([table_image(seed) for seed in range(5)])

    def test_empty_table_with_exposure_and_white_balance_drift(self):
        drifted = table_image(seed=10, gain=(0.85, 0.9, 1.05))
        for _ in range(3):
            self.assertEqual(self.detector.detect(drifted), [])

    def test_colored_and_dark_objects_without_their_shadows(self):
        frame = add_disk(table_image(seed=11), (400, 300), 40, (40, 60, 200))     # red
        frame = add_disk(frame, (850, 450), 50, (50, 50, 50))                     # dark grey
        objects = sorted(self.detector.detect(frame), key=lambda o: o.centroid[0])
        self.assertEqual(len(objects), 2)
        for obj, (cx, cy, r) in zip(objects, [(400, 300, 40), (850, 450, 50)]):
            np.testing.assert_allclose(obj.centroid, (cx, cy), atol=4)
            self.assertAlmostEqual(obj.area / (np.pi * r * r), 1.0, delta=0.2)  # shadow not included
            self.assertFalse(obj.touches_border)

    def test_blob_reaching_the_table_edge_is_flagged(self):
        self.detector.set_roi([[100, 100], [1180, 100], [1180, 650], [100, 650]])
        frame = table_image(seed=12)
        cv2.rectangle(frame, (60, 300), (300, 360), (90, 120, 170), -1)  # arm coming from the left
        objects = self.detector.detect(frame)
        self.assertEqual(len(objects), 1)
        self.assertTrue(objects[0].touches_border)


class MeasureTests(unittest.TestCase):
    def test_object_temperature(self):
        cal = Calibration(simulated_homography(RGB_SIZE, THERMAL_SIZE), RGB_SIZE, THERMAL_SIZE)
        temps = np.full((288, 382), 21.0, np.float32)
        center = cal.rgb_to_thermal([[640, 360]])[0]
        cv2.circle(temps, tuple(np.round(center).astype(int)), 14, 55.0, -1)
        angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        polygon = np.stack([640 + 20 * np.cos(angles), 360 + 20 * np.sin(angles)], axis=1)  # ~10 thermal px
        mask, _, coverage = object_mask(polygon, cal, temps.shape)
        stats = measure(temps, mask, coverage)
        self.assertAlmostEqual(stats.median, 55.0, places=3)
        self.assertAlmostEqual(coverage, 1.0, delta=0.05)

    def test_object_outside_thermal_view(self):
        cal = Calibration(simulated_homography(RGB_SIZE, THERMAL_SIZE), RGB_SIZE, THERMAL_SIZE)
        polygon = np.array([[10, 10], [60, 10], [60, 60], [10, 60]], float)
        mask, _, coverage = object_mask(polygon, cal, (288, 382))
        self.assertIsNone(measure(np.zeros((288, 382), np.float32), mask, coverage))
        self.assertEqual(coverage, 0.0)


class RobustnessTests(unittest.TestCase):
    def test_degenerate_calibration_does_not_crash_rendering(self):
        """A homography sending points towards infinity must not break drawing or measuring."""
        rgb = FakeRgbCamera(table_image(seed=30))
        thermal = SimulatedThermalCamera(rgb).start()
        with tempfile.TemporaryDirectory() as tmp:
            app = App(rgb, thermal, tmp)
            H = np.array([[1, 0, 0], [0, 1, 0], [0.002, 0, -1.28]])  # x = 640 maps to infinity
            app.calibration = Calibration(H, RGB_SIZE, THERMAL_SIZE)
            app.roi = [[100, 100], [1180, 100], [1180, 650], [100, 650]]
            app.on_key(ord("b"))
            for _ in range(22):
                app.step()
            rgb.frame = add_disk(table_image(seed=31), (640, 360), 40, (20, 20, 200))
            app.on_key(ord("o"))
            for _ in range(3):
                self.assertIsNotNone(app.step())


class EndToEndTests(unittest.TestCase):
    """Background, automatic calibration and measurement through App, with the simulated thermal camera
    (dark = hot, true homography known)."""

    def test_full_workflow(self):
        rgb = FakeRgbCamera(table_image(seed=20))
        thermal = SimulatedThermalCamera(rgb).start()
        with tempfile.TemporaryDirectory() as tmp:
            app = App(rgb, thermal, tmp)
            app.step()
            app.on_key(ord("b"))
            for _ in range(25):
                app.step()
            self.assertTrue(app.detector.has_background)

            app.on_key(ord("c"))
            self.assertEqual(app.mode, "calibrate")
            positions = [(420, 200), (880, 220), (900, 560), (400, 560), (640, 380), (560, 300), (760, 470)]
            for center in positions:
                rgb.frame = add_disk(table_image(seed=21), center, 30, (20, 20, 20), shadow=False)
                for _ in range(3):
                    app.step()
                app.on_key(ord(" "))
                for _ in range(12):
                    app.step()
            self.assertEqual(len(app.cal_rgb), len(positions))

            # An object outside the thermal camera's view must not produce a pair
            rgb.frame = add_disk(table_image(seed=21), (120, 640), 30, (20, 20, 20), shadow=False)
            for _ in range(3):
                app.step()
            app.on_key(ord(" "))
            for _ in range(12):
                app.step()
            self.assertEqual(len(app.cal_rgb), len(positions))
            self.assertIn("no warm/cold spot", app.message)

            app.on_key(13)
            self.assertEqual(app.mode, "run")
            self.assertLess(app.calibration.rms, 1.5)

            truth = Calibration(thermal.homography, RGB_SIZE, THERMAL_SIZE)
            probe = [[500, 300], [800, 500]]
            np.testing.assert_allclose(app.calibration.rgb_to_thermal(probe), truth.rgb_to_thermal(probe), atol=2.0)

            # Simulated temperature is 22 + 25 * darkness^2: 47 C for a black object, ~23.5 C for the table
            rgb.frame = add_disk(table_image(seed=22), (650, 400), 45, (0, 0, 0))
            for _ in range(10):
                canvas = app.step()
            tracks = [t for t in app.tracks if not t.obj.touches_border]
            self.assertEqual(len(tracks), 1)
            self.assertAlmostEqual(tracks[0].temp, 47.0, delta=1.5)
            self.assertAlmostEqual(app.table_temp, 23.5, delta=0.7)
            self.assertEqual(canvas.shape[1], app.layout.width)

            app.on_key(ord("o"))
            app.step()
            app.on_key(ord("s"))
            self.assertEqual(len(list((app.data_dir / "snapshots").glob("*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
