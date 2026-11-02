"""RGB <-> thermal registration.

The objects lie on a flat table, so a single homography maps webcam pixels to thermal pixels for
everything on the table plane. It is estimated from pairs of matching points, collected either by
clicking or automatically from a warm object seen by both cameras.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Calibration:
    H: np.ndarray                      # 3x3, maps RGB pixels -> thermal pixels (table plane)
    rgb_size: tuple
    thermal_size: tuple
    rgb_points: list = field(default_factory=list)
    thermal_points: list = field(default_factory=list)
    rms: float = float("nan")          # reprojection error of the inlier pairs, in thermal pixels

    def __post_init__(self):
        self.H = np.asarray(self.H, dtype=np.float64)
        self.H_inv = np.linalg.inv(self.H)

    def rgb_to_thermal(self, points):
        return _transform(points, self.H)

    def thermal_to_rgb(self, points):
        return _transform(points, self.H_inv)

    def thermal_fov_in_rgb(self):
        """Outline of the thermal image in RGB pixel coordinates."""
        w, h = self.thermal_size
        return self.thermal_to_rgb([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

    def save(self, path):
        data = {
            "H_rgb_to_thermal": self.H.tolist(),
            "rgb_size": list(self.rgb_size),
            "thermal_size": list(self.thermal_size),
            "rgb_points": np.asarray(self.rgb_points).tolist(),
            "thermal_points": np.asarray(self.thermal_points).tolist(),
            "rms_thermal_px": self.rms,
            "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        return cls(H=data["H_rgb_to_thermal"], rgb_size=tuple(data["rgb_size"]),
                   thermal_size=tuple(data["thermal_size"]), rgb_points=data["rgb_points"],
                   thermal_points=data["thermal_points"], rms=data["rms_thermal_px"])


def _transform(points, H):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def fit_homography(rgb_points, thermal_points, ransac_threshold=3.0):
    """Estimates the RGB -> thermal homography.

    Returns (H, rms, per-pair errors, inlier flags). Uses RANSAC once there are enough pairs to
    reject a bad one, plain least squares otherwise.
    """
    src = np.asarray(rgb_points, dtype=np.float64)
    dst = np.asarray(thermal_points, dtype=np.float64)
    if len(src) < 4:
        raise ValueError("At least 4 point pairs are needed")
    # Points bunched together or on a line give a meaningless fit with a deceptively small error
    if _spread(src) < 40 or _spread(dst) < 8:
        raise ValueError("The points are too close together or aligned: spread them over the table")
    method = cv2.RANSAC if len(src) >= 6 else 0
    H, inliers = cv2.findHomography(src, dst, method, ransac_threshold)
    if H is None:
        raise ValueError("Could not fit a homography (are the points spread out, not on a line?)")
    errors = np.linalg.norm(_transform(src, H) - dst, axis=1)
    inliers = inliers.ravel().astype(bool) if inliers is not None else np.ones(len(src), bool)
    rms = float(np.sqrt(np.mean(errors[inliers] ** 2)))
    return H, rms, errors, inliers


def _spread(points):
    """Smallest standard deviation of the points along any direction (0 if they are on a line)."""
    centered = points - points.mean(axis=0)
    return float(np.sqrt(max(np.linalg.eigvalsh(centered.T @ centered / len(points))[0], 0.0)))


def find_thermal_spot(temps, reference=None, near=None, radius=40.0, min_delta=2.0, min_pixels=4):
    """Finds the region that stands out most in the thermal image (a warm or cold object).

    reference: thermal image of the empty table. When given, only changes are considered, so static
        warm things in view (electronics, people next to the table) are ignored.
    near: (x, y) where the object is expected; regions farther than ``radius`` are ignored.
    Returns ((x, y) centroid in thermal pixels, region mask) or (None, None).
    """
    deviation = temps - (reference if reference is not None else float(np.median(temps)))
    deviation = deviation - float(np.median(deviation))  # global drift since the reference
    if near is not None:
        ys, xs = np.mgrid[0:temps.shape[0], 0:temps.shape[1]]
        deviation[(xs - near[0]) ** 2 + (ys - near[1]) ** 2 > radius ** 2] = 0
    mad = float(np.median(np.abs(deviation))) * 1.4826
    mask = (np.abs(deviation) > max(min_delta, 5 * mad)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best, best_score = None, 0.0
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < min_pixels:
            continue
        score = float(np.abs(deviation[labels == label]).sum())
        if score > best_score:
            best, best_score = label, score
    if best is None:
        return None, None
    region = labels == best
    # Tighten the region to the part clearly belonging to the object (above half its peak contrast)
    peak = np.abs(deviation[region]).max()
    core = region & (np.abs(deviation) >= 0.5 * peak)
    ys, xs = np.nonzero(core)
    return (float(xs.mean()), float(ys.mean())), core
