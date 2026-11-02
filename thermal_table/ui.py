"""Drawing helpers. Nothing here opens a window, so rendering can be tested headlessly."""

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, BLACK, GREY = (255, 255, 255), (0, 0, 0), (150, 150, 150)
CYAN, YELLOW, RED, GREEN = (255, 255, 0), (0, 255, 255), (0, 0, 255), (0, 220, 0)
PALETTE = [(80, 220, 80), (255, 160, 40), (60, 200, 255), (220, 80, 220), (255, 255, 90), (90, 120, 255),
           (180, 255, 180), (40, 120, 255)]


def object_color(index):
    return PALETTE[index % len(PALETTE)]


class Layout:
    """Side by side canvas: webcam view on the left, thermal view on the right, status bar below."""

    def __init__(self, rgb_size, thermal_size, height=540, status_height=58):
        self.height = height
        self.status_height = status_height
        self.rgb_scale = height / rgb_size[1]
        self.rgb_width = round(rgb_size[0] * self.rgb_scale)
        self.thermal_scale = height / thermal_size[1]
        self.thermal_width = round(thermal_size[0] * self.thermal_scale)
        self.width = self.rgb_width + self.thermal_width

    def locate(self, x, y):
        """Maps a canvas click to ('rgb' | 'thermal', (x, y) in that camera's pixels) or (None, None)."""
        if y >= self.height:
            return None, None
        if x < self.rgb_width:
            return "rgb", (x / self.rgb_scale, y / self.rgb_scale)
        return "thermal", ((x - self.rgb_width) / self.thermal_scale, y / self.thermal_scale)

    def rgb_points(self, points):
        return np.asarray(points, np.float64) * self.rgb_scale

    def thermal_points(self, points):
        return np.asarray(points, np.float64) * self.thermal_scale


class ColorRange:
    """Auto-scaling temperature range for the false color image, smoothed to avoid flicker."""

    def __init__(self, min_span=5.0, smoothing=0.15):
        self.min_span = min_span
        self.smoothing = smoothing
        self.lo = self.hi = None

    def update(self, temps):
        lo, hi = np.percentile(temps, [1, 99.5])
        if hi - lo < self.min_span:
            mid = (lo + hi) / 2
            lo, hi = mid - self.min_span / 2, mid + self.min_span / 2
        if self.lo is None:
            self.lo, self.hi = lo, hi
        else:
            self.lo += self.smoothing * (lo - self.lo)
            self.hi += self.smoothing * (hi - self.hi)
        return self.lo, self.hi


def colorize(temps, lo, hi):
    normalized = np.clip((temps - lo) / max(hi - lo, 1e-3) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)


def draw_colorbar(image, lo, hi, width=14, margin=10):
    h = image.shape[0]
    top, bottom = margin + 14, h - margin - 14
    x = image.shape[1] - width - margin
    gradient = np.linspace(255, 0, bottom - top).astype(np.uint8).reshape(-1, 1)
    bar = cv2.applyColorMap(np.repeat(gradient, width, axis=1), cv2.COLORMAP_INFERNO)
    image[top:bottom, x:x + width] = bar
    cv2.rectangle(image, (x - 1, top - 1), (x + width, bottom), WHITE, 1)
    put_text(image, f"{hi:.1f}", (x - 34, top - 4), 0.42)
    put_text(image, f"{lo:.1f}", (x - 34, bottom + 13), 0.42)


def _drawable(points):
    """A bad calibration can send points towards infinity, which OpenCV's int32 drawing rejects."""
    points = np.asarray(points, np.float64)
    return points.size > 0 and np.isfinite(points).all() and np.abs(points).max() < 1e5


def put_text(image, text, origin, scale=0.5, color=WHITE, thickness=1):
    """Text with a dark outline so it stays readable on any background."""
    if not _drawable(origin):
        return
    origin = (int(origin[0]), int(origin[1]))
    cv2.putText(image, text, origin, FONT, scale, BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_label(image, lines, anchor, color, scale=0.55):
    """Draws a small box of text lines whose bottom-left corner is at anchor (kept inside the image)."""
    sizes = [cv2.getTextSize(line, FONT, scale, 1)[0] for line in lines]
    line_h = max(s[1] for s in sizes) + 6
    box_w = max(s[0] for s in sizes) + 8
    box_h = line_h * len(lines) + 4
    x = int(np.clip(anchor[0], 0, max(0, image.shape[1] - box_w)))
    y = int(np.clip(anchor[1] - box_h, 0, max(0, image.shape[0] - box_h)))
    roi = image[y:y + box_h, x:x + box_w]
    roi[:] = (roi * 0.35).astype(np.uint8)
    cv2.rectangle(image, (x, y), (x + box_w, y + box_h), color, 1)
    for i, line in enumerate(lines):
        cv2.putText(image, line, (x + 4, y + (i + 1) * line_h - 2), FONT, scale, color if i == 0 else WHITE, 1,
                    cv2.LINE_AA)


def draw_polygon(image, points, color, thickness=2, closed=True):
    if points is None or len(points) < 2 or not _drawable(points):
        return
    cv2.polylines(image, [np.round(points).astype(np.int32)], closed, color, thickness, cv2.LINE_AA)


def draw_cross(image, point, color, size=10, thickness=2):
    if not _drawable(point):
        return
    cv2.drawMarker(image, (int(round(point[0])), int(round(point[1]))), color, cv2.MARKER_CROSS, size, thickness,
                   cv2.LINE_AA)


def draw_numbered_point(image, point, number, color, filled=True):
    if not _drawable(point):
        return
    p = (int(round(point[0])), int(round(point[1])))
    cv2.circle(image, p, 6, BLACK, -1 if filled else 3, cv2.LINE_AA)
    cv2.circle(image, p, 5, color, -1 if filled else 2, cv2.LINE_AA)
    if number is not None:
        put_text(image, str(number), (p[0] + 7, p[1] - 7), 0.5, color)


def blend_thermal(rgb_view, thermal_color, H_thermal_to_view, alpha=0.5):
    """Warps the false color thermal image onto the webcam view (where the thermal camera sees)."""
    h, w = rgb_view.shape[:2]
    warped = cv2.warpPerspective(thermal_color, H_thermal_to_view, (w, h), flags=cv2.INTER_LINEAR)
    coverage = cv2.warpPerspective(np.full(thermal_color.shape[:2], 255, np.uint8), H_thermal_to_view, (w, h))
    mixed = cv2.addWeighted(rgb_view, 1 - alpha, warped, alpha, 0)
    out = rgb_view.copy()
    inside = coverage > 127
    out[inside] = mixed[inside]
    return out


def placeholder(size, lines):
    w, h = size
    image = np.full((h, w, 3), 40, np.uint8)
    for i, line in enumerate(lines):
        put_text(image, line, (16, 40 + 26 * i), 0.55, (200, 200, 200))
    return image


def compose(layout, rgb_view, thermal_view, status_lines):
    canvas = np.zeros((layout.height + layout.status_height, layout.width, 3), np.uint8)
    canvas[:layout.height, :layout.rgb_width] = rgb_view
    canvas[:layout.height, layout.rgb_width:] = thermal_view
    cv2.line(canvas, (layout.rgb_width, 0), (layout.rgb_width, layout.height), GREY, 1)
    for i, (text, color) in enumerate(status_lines):
        cv2.putText(canvas, text, (10, layout.height + 22 + 24 * i), FONT, 0.55, color, 1, cv2.LINE_AA)
    return canvas


def instructions_box(image, lines, color=YELLOW):
    """Semi-transparent help panel in the top-left corner."""
    sizes = [cv2.getTextSize(line, FONT, 0.5, 1)[0] for line in lines]
    w = max(s[0] for s in sizes) + 20
    h = 24 * len(lines) + 12
    roi = image[0:h, 0:w]
    roi[:] = (roi * 0.3).astype(np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(image, line, (10, 24 + 24 * i), FONT, 0.5, color if i == 0 else WHITE, 1, cv2.LINE_AA)
