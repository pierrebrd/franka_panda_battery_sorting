"""Object detection on the white table, from the webcam image.

A reference image of the empty table is compared with the live image in Lab color space. A pixel is
an object pixel if its color changed (a/b channels) or if it got much darker/brighter. Soft shadows
only darken the table a little and keep its color, so they stay below the darkness threshold.
Global brightness / white balance drifts from the webcam's automatic settings are compensated by
per-channel gains measured on the visible table.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DetectorConfig:
    process_width: int = 640      # detection runs on a downscaled image
    chroma_thresh: float = 14.0   # a/b change (OpenCV 8-bit Lab units) that marks an object
    dark_thresh: float = 50.0     # L drop that marks a neutral (grey/black) object; shadows stay below
    light_thresh: float = 40.0    # L increase that marks an object brighter than the table
    weak_ratio: float = 0.5       # pixels above this fraction of a threshold grow strong detections
    min_area: int = 150           # minimum object area in processing pixels
    border_margin: int = 4        # blobs this close to the table edge are hands/arms reaching in
    adapt_rate: float = 0.01      # background update rate for table pixels without objects


@dataclass
class DetectedObject:
    contour: np.ndarray           # (N, 2) float32 polygon in full-resolution RGB pixels
    area: float                   # full-resolution pixels
    centroid: tuple
    bbox: tuple                   # x, y, w, h in full-resolution pixels
    touches_border: bool          # probably a hand or arm entering the table area


class ObjectDetector:
    def __init__(self, config=None):
        self.config = config or DetectorConfig()
        self.frame_size = None     # full-resolution (w, h)
        self.proc_size = None      # processing (w, h)
        self.roi_polygon = None    # full-resolution polygon or None for the whole image
        self._roi = None           # processing-resolution masks
        self._roi_border = None
        self._background = None    # float32 BGR, processing resolution
        self._background_lab = None
        self._gains = np.ones(3, np.float32)
        self._exclude = None       # dilated object mask of the previous frame
        self.last_score = None     # for the debug view

    @property
    def has_background(self):
        return self._background is not None

    def _ensure_size(self, frame_shape):
        size = (frame_shape[1], frame_shape[0])
        if size == self.frame_size:
            return
        self.frame_size = size
        scale = min(1.0, self.config.process_width / size[0])
        self.proc_size = (round(size[0] * scale), round(size[1] * scale))
        self._background = self._background_lab = None
        self._update_roi()

    def set_roi(self, polygon):
        """Restricts detection to a polygon (full-resolution pixels), or the whole image if None."""
        self.roi_polygon = None if polygon is None or len(polygon) < 3 else np.asarray(polygon, np.float32)
        if self.frame_size is not None:
            self._update_roi()

    def _update_roi(self):
        w, h = self.proc_size
        roi = np.zeros((h, w), np.uint8)
        if self.roi_polygon is None:
            roi[:] = 1
        else:
            pts = self.roi_polygon * (w / self.frame_size[0])
            cv2.fillPoly(roi, [np.round(pts).astype(np.int32)], 1)
        m = self.config.border_margin
        inner = cv2.erode(roi, np.ones((2 * m + 1, 2 * m + 1), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        self._roi = roi.astype(bool)
        self._roi_border = self._roi & ~inner.astype(bool)
        self._exclude = np.zeros((h, w), bool)

    def _prepare(self, frame):
        small = cv2.resize(frame, self.proc_size, interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(small, (5, 5), 0).astype(np.float32)

    def set_background(self, frames):
        """Builds the empty-table reference from one or more full-resolution frames."""
        self._ensure_size(frames[0].shape)
        stack = np.stack([self._prepare(f) for f in frames])
        self._set_background(np.median(stack, axis=0).astype(np.float32))

    def _set_background(self, background):
        self._background = background
        self._background_lab = _lab(background)
        self._gains[:] = 1.0
        self._exclude[:] = False

    def save_background(self, path):
        cv2.imwrite(str(path), np.clip(self._background, 0, 255).astype(np.uint8))

    def load_background(self, path, frame_shape):
        """Loads a saved reference; ignored if it was taken at a different resolution."""
        self._ensure_size(frame_shape)
        image = cv2.imread(str(path))
        if image is None or (image.shape[1], image.shape[0]) != self.proc_size:
            return False
        self._set_background(image.astype(np.float32))
        return True

    def detect(self, frame):
        self._ensure_size(frame.shape)
        if self._background is None:
            return []
        cfg = self.config
        current = self._prepare(frame)

        # Compensate exposure / white balance drift with gains measured on visible table pixels
        # (a subsampled grid is plenty for a median and much faster)
        table = (self._roi & ~self._exclude)[::4, ::4]
        if table.sum() > 0.05 * table.size:
            ratios = current[::4, ::4][table] / np.maximum(self._background[::4, ::4][table], 1.0)
            self._gains = np.clip(np.median(ratios, axis=0), 0.5, 2.0).astype(np.float32)
        compensated = np.clip(current * (1.0 / self._gains), 0, 255)

        lab = _lab(compensated)
        d_light = lab[..., 0] - self._background_lab[..., 0]
        d_chroma = cv2.magnitude(lab[..., 1] - self._background_lab[..., 1], lab[..., 2] - self._background_lab[..., 2])
        color_score = np.maximum(d_chroma / cfg.chroma_thresh, np.maximum(d_light, 0) / cfg.light_thresh)
        dark_score = np.maximum(-d_light, 0) / cfg.dark_thresh
        score = np.maximum(color_score, dark_score)
        score[~self._roi] = 0
        self.last_score = score

        # Hysteresis: weak regions are kept only if they contain a strong detection. Being a little
        # darker is not enough to count as weak, otherwise objects would grow into their shadows.
        strong = score > 1.0
        weak = ((color_score > cfg.weak_ratio) | (dark_score > 0.85)) & self._roi
        weak = cv2.morphologyEx(weak.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, labels = cv2.connectedComponents(weak, connectivity=8)
        keep = np.zeros(count, bool)
        keep[np.unique(labels[strong & (weak > 0)])] = True
        keep[0] = False
        mask = keep[labels].astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scale = self.frame_size[0] / self.proc_size[0]
        objects, filled = [], np.zeros_like(mask)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < cfg.min_area:
                continue
            blob = np.zeros_like(mask)
            cv2.drawContours(blob, [contour], -1, 1, cv2.FILLED)
            filled |= blob
            m = cv2.moments(contour)
            x, y, w, h = cv2.boundingRect(contour)
            objects.append(DetectedObject(
                contour=contour.reshape(-1, 2).astype(np.float32) * scale,
                area=area * scale * scale,
                centroid=(m["m10"] / m["m00"] * scale, m["m01"] / m["m00"] * scale),
                bbox=tuple(int(round(v * scale)) for v in (x, y, w, h)),
                touches_border=bool((blob.astype(bool) & self._roi_border).any()),
            ))

        # Slowly follow lighting changes on table pixels that are far from any object
        self._exclude = cv2.dilate(filled | (score > cfg.weak_ratio), np.ones((15, 15), np.uint8)).astype(bool)
        update = (self._roi & ~self._exclude).astype(np.uint8)
        if cfg.adapt_rate > 0 and update.any():
            cv2.accumulateWeighted(compensated.astype(np.float32), self._background, cfg.adapt_rate, update)
            self._background_lab = _lab(self._background)
        return objects


def _lab(bgr_float):
    return cv2.cvtColor(np.clip(bgr_float, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
