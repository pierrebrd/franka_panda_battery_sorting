"""Temperature of the detected objects, read from the thermal image through the calibration."""

from dataclasses import dataclass

import cv2
import numpy as np

SUBPIXEL_SHIFT = 4  # fillPoly fixed-point bits, the thermal pixels are coarse


@dataclass
class TempStats:
    median: float
    mean: float
    min: float
    max: float
    pixels: int        # thermal pixels used
    coverage: float    # fraction of the object that lies inside the thermal image


def object_mask(polygon_rgb, calibration, thermal_shape, erode=True):
    """Rasterizes an RGB object polygon in the thermal image.

    Returns (mask, polygon in thermal pixels, coverage). The mask is eroded by one thermal pixel when
    the object is large enough, so mixed object/table pixels on the outline are not used.
    """
    # Clipping only guards against int32 overflow from a degenerate calibration, it does not change
    # anything inside the image for a sane one
    pts = np.clip(np.nan_to_num(calibration.rgb_to_thermal(polygon_rgb)), -1e4, 1e4)
    mask = np.zeros(thermal_shape, np.uint8)
    cv2.fillPoly(mask, [np.round(pts * (1 << SUBPIXEL_SHIFT)).astype(np.int32)], 1, cv2.LINE_8, SUBPIXEL_SHIFT)
    inside = int(mask.sum())
    total = cv2.contourArea(pts.astype(np.float32))
    coverage = min(1.0, inside / total) if total > 0 else 0.0
    if erode and inside > 12:
        eroded = cv2.erode(mask, np.ones((3, 3), np.uint8))
        if eroded.sum() >= 4:
            mask = eroded
    return mask.astype(bool), pts, coverage


def measure(temps, mask, coverage):
    values = temps[mask]
    if values.size == 0:
        return None
    return TempStats(median=float(np.median(values)), mean=float(values.mean()), min=float(values.min()),
                     max=float(values.max()), pixels=int(values.size), coverage=coverage)


def table_temperature(temps, object_masks, roi_polygon_rgb=None, calibration=None):
    """Median temperature of the table, ignoring the objects (and outside the ROI if one is set)."""
    usable = np.ones(temps.shape, bool)
    if roi_polygon_rgb is not None and calibration is not None:
        roi = np.zeros(temps.shape, np.uint8)
        pts = np.clip(np.nan_to_num(calibration.rgb_to_thermal(roi_polygon_rgb)), -1e4, 1e4)
        cv2.fillPoly(roi, [np.round(pts).astype(np.int32)], 1)
        usable &= roi.astype(bool)
    if object_masks:
        objects = np.any(object_masks, axis=0).astype(np.uint8)
        usable &= ~cv2.dilate(objects, np.ones((5, 5), np.uint8)).astype(bool)
    values = temps[usable]
    return float(np.median(values)) if values.size else None


@dataclass
class Track:
    id: int
    obj: object                 # latest DetectedObject
    stats: TempStats = None     # latest measurement
    temp: float = None          # smoothed median temperature
    temp_max: float = None      # smoothed max temperature
    thermal_polygon: np.ndarray = None
    age: int = 0
    missed: int = 0


class Tracker:
    """Keeps object identities across frames and smooths their temperature readings."""

    def __init__(self, max_distance=60.0, smoothing=0.3, max_missed=8):
        self.max_distance = max_distance
        self.smoothing = smoothing
        self.max_missed = max_missed
        self.tracks = []
        self._next_id = 1

    def update(self, objects, measurements, thermal_polygons):
        pairs = sorted(
            (np.hypot(o.centroid[0] - t.obj.centroid[0], o.centroid[1] - t.obj.centroid[1]), i, j)
            for i, o in enumerate(objects) for j, t in enumerate(self.tracks)
        )
        assigned_obj, assigned_track = {}, set()
        for dist, i, j in pairs:
            if dist > self.max_distance or i in assigned_obj or j in assigned_track:
                continue
            assigned_obj[i] = self.tracks[j]
            assigned_track.add(j)

        for j, track in enumerate(self.tracks):
            if j not in assigned_track:
                track.missed += 1
        for i, obj in enumerate(objects):
            track = assigned_obj.get(i)
            if track is None:
                track = Track(id=self._next_id, obj=obj)
                self._next_id += 1
                self.tracks.append(track)
            track.obj, track.stats, track.thermal_polygon = obj, measurements[i], thermal_polygons[i]
            track.age += 1
            track.missed = 0
            stats = measurements[i]
            if stats is None:
                track.temp = track.temp_max = None
            elif track.temp is None:
                track.temp, track.temp_max = stats.median, stats.max
            else:
                a = self.smoothing
                track.temp += a * (stats.median - track.temp)
                track.temp_max += a * (stats.max - track.temp_max)

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return [t for t in self.tracks if t.missed == 0]

    def reset(self):
        self.tracks = []
