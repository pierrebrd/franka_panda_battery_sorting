"""Main application: modes, keyboard/mouse handling and the processing loop.

``App.step()`` does all the work for one frame and returns the image to display, so the app can be
driven without a window (tests); ``run()`` adds the OpenCV window, mouse and keyboard.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import ui
from .calibration import Calibration, find_thermal_spot, fit_homography
from .detection import ObjectDetector
from .measure import Tracker, measure, object_mask, table_temperature

WINDOW = "Thermal table"
DEBUG_WINDOW = "Detection debug"
BACKGROUND_FRAMES = 20   # frames averaged for the empty-table reference
CAPTURE_FRAMES = 10      # frames averaged for one automatic calibration pair
STABLE_AGE = 15          # frames an object must be seen before its temperature is printed
DEFAULT_THERMAL_SIZE = (382, 288)

KEY_ENTER = (10, 13)
KEY_ESC = 27
KEY_SPACE = 32
KEY_BACKSPACE = (8, 127)

RUN_KEYS = "b: empty-table background   c: calibrate   r: table area   o: overlay   d: debug   f: flag   s: snapshot   q: quit"


class App:
    def __init__(self, rgb, thermal, data_dir, thermal_error=None, display_height=540):
        self.rgb = rgb
        self.thermal = thermal                 # None when unavailable
        self.thermal_error = thermal_error
        self.data_dir = Path(data_dir)
        self.thermal_size = tuple(thermal.size) if thermal is not None else DEFAULT_THERMAL_SIZE
        self.layout = ui.Layout(rgb.size, self.thermal_size, display_height, status_height=80)
        self.detector = ObjectDetector()
        self.tracker = Tracker()
        self.color_range = ui.ColorRange()
        self.mode = "run"
        self.overlay = False
        self.debug = False
        self.show_help = True
        self.running = True
        self.message, self.message_until, self.message_is_error = "", 0.0, False
        self.calibration = None
        self.roi = None

        self.frame = None
        self.temps = None
        self.temps_valid = False
        self.temp_range = (20.0, 30.0)
        self.objects = []
        self.tracks = []
        self.table_temp = None
        self.canvas = None
        self._rgb_count = -1
        self._background_frames = None         # frames being collected, or None
        self._thermal_frames = []
        self.thermal_reference = None          # thermal image of the empty table, used by calibration

        self.cal_rgb, self.cal_th = [], []     # point pairs
        self.cal_pending = {}                  # "rgb" / "thermal" -> clicked point waiting for its partner
        self.cal_preview = None                # Calibration fitted on the current pairs
        self.cal_errors = self.cal_inliers = None
        self.cal_samples = None                # automatic capture in progress
        self.cal_capture_frames = 0
        self.cal_live_rgb = self.cal_live_th = None

        self.roi_points = []
        self._load_setup()

    # ----- persistence -------------------------------------------------------------------------

    @property
    def calibration_path(self):
        return self.data_dir / "calibration.json"

    @property
    def roi_path(self):
        return self.data_dir / "table_area.json"

    @property
    def background_path(self):
        return self.data_dir / "background.png"

    def _load_setup(self):
        if self.calibration_path.exists():
            try:
                calibration = Calibration.load(self.calibration_path)
            except (ValueError, KeyError, np.linalg.LinAlgError) as ex:
                print(f"Ignoring unreadable calibration {self.calibration_path}: {ex}")
            else:
                if tuple(calibration.rgb_size) == tuple(self.rgb.size) and \
                        tuple(calibration.thermal_size) == self.thermal_size:
                    self.calibration = calibration
                else:
                    print("Saved calibration was made with other image sizes, ignoring it.")
        if self.roi_path.exists():
            self.roi = json.loads(self.roi_path.read_text()).get("polygon")
            self.detector.set_roi(self.roi)
        frame, _ = self.rgb.read()
        if self.background_path.exists() and frame is not None:
            if not self.detector.load_background(self.background_path, frame.shape):
                print("Saved background has another size, ignoring it.")

    # ----- processing --------------------------------------------------------------------------

    def step(self):
        frame, count = self.rgb.read()
        if frame is None:
            return None
        if count != self._rgb_count:
            self._rgb_count = count
            self.frame = frame
            if self.thermal is not None:
                temps, _, valid = self.thermal.read()
                self.temps, self.temps_valid = temps, valid and temps is not None
                if self.temps is not None:
                    self.temp_range = self.color_range.update(self.temps)
            if self._background_frames is not None:
                self._collect_background(frame)
                self.objects = []
            else:
                self.objects = self.detector.detect(frame)
            self._measure()
            if self.mode == "calibrate":
                self._calibration_step()
        self.canvas = self._render()
        return self.canvas

    def _measure(self):
        can_measure = self.calibration is not None and self.temps_valid
        masks, stats, polygons = [], [], []
        for obj in self.objects:
            if can_measure:
                mask, polygon, coverage = object_mask(obj.contour, self.calibration, self.temps.shape)
                masks.append(mask)
                stats.append(measure(self.temps, mask, coverage))
                polygons.append(polygon)
            else:
                stats.append(None)
                polygons.append(None)
        self.table_temp = table_temperature(self.temps, masks, self.roi, self.calibration) if can_measure else None
        self.tracks = self.tracker.update(self.objects, stats, polygons)
        for track in self.tracks:
            if track.age == STABLE_AGE and track.temp is not None and not track.obj.touches_border:
                x, y = track.obj.centroid
                print(f"Object #{track.id} at ({x:.0f}, {y:.0f}): {track.temp:.1f} C "
                      f"(min {track.stats.min:.1f}, max {track.temp_max:.1f})")

    def _collect_background(self, frame):
        self._background_frames.append(frame)
        if self.temps_valid:
            self._thermal_frames.append(self.temps)
        if len(self._background_frames) < BACKGROUND_FRAMES:
            return
        self.detector.set_background(self._background_frames)
        if self._thermal_frames:
            self.thermal_reference = np.median(self._thermal_frames, axis=0).astype(np.float32)
        self._background_frames, self._thermal_frames = None, []
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.detector.save_background(self.background_path)
        self.tracker.reset()
        self.notify("Empty-table background saved. Objects placed on the table will now be detected.")

    def _calibration_step(self):
        candidates = [o for o in self.objects if not o.touches_border]
        self.cal_live_rgb = candidates[0].centroid if len(candidates) == 1 else None
        self.cal_live_th = None
        if self.temps_valid:
            # Once a first fit exists, look first where the webcam says the object should be
            if self.cal_preview is not None and self.cal_live_rgb is not None:
                near = self.cal_preview.rgb_to_thermal([self.cal_live_rgb])[0]
                self.cal_live_th = find_thermal_spot(self.temps, self.thermal_reference, near)[0]
            if self.cal_live_th is None:
                self.cal_live_th = find_thermal_spot(self.temps, self.thermal_reference)[0]
        if self.cal_samples is None:
            return
        if self.cal_live_rgb is not None and self.cal_live_th is not None:
            self.cal_samples.append((self.cal_live_rgb, self.cal_live_th))
        self.cal_capture_frames += 1
        if self.cal_capture_frames < CAPTURE_FRAMES:
            return
        samples, self.cal_samples = self.cal_samples, None
        if len(samples) >= CAPTURE_FRAMES // 2:
            rgb_point = np.median([s[0] for s in samples], axis=0)
            thermal_point = np.median([s[1] for s in samples], axis=0)
            if self._add_pair(rgb_point, thermal_point):
                self.notify(f"Pair {len(self.cal_rgb)} added. Move the object elsewhere and press SPACE again.")
        elif len(candidates) != 1:
            self.notify(f"Capture failed: the webcam sees {len(candidates)} objects, it needs exactly one "
                        "(hands must be off the table).", error=True)
        else:
            self.notify("Capture failed: no warm/cold spot in the thermal image. Use an object at least "
                        "a few degrees warmer or colder than the table.", error=True)

    # ----- calibration mode --------------------------------------------------------------------

    def enter_calibration(self):
        if self.thermal is None:
            self.notify(f"Calibration needs the thermal camera: {self.thermal_error}", error=True)
            return
        self.mode = "calibrate"
        self.cal_pending = {}
        self.cal_samples = None
        # Resume unsaved pairs from a previous visit, otherwise start from the saved calibration
        if not self.cal_rgb and self.calibration is not None:
            self.cal_rgb = [list(p) for p in self.calibration.rgb_points]
            self.cal_th = [list(p) for p in self.calibration.thermal_points]
        self._update_cal_preview()

    def _add_pair(self, rgb_point, thermal_point):
        for i, (p_rgb, p_th) in enumerate(zip(self.cal_rgb, self.cal_th)):
            if np.hypot(*np.subtract(p_rgb, rgb_point)) < 15 or np.hypot(*np.subtract(p_th, thermal_point)) < 4:
                self.notify(f"Same place as pair {i + 1}: move the object to another spot.", error=True)
                return False
        self.cal_rgb.append([float(v) for v in rgb_point])
        self.cal_th.append([float(v) for v in thermal_point])
        self._update_cal_preview()
        return True

    def _undo_pair(self):
        if self.cal_pending:
            self.cal_pending = {}
        elif self.cal_rgb:
            self.cal_rgb.pop()
            self.cal_th.pop()
            self._update_cal_preview()

    def _update_cal_preview(self):
        self.cal_preview = self.cal_errors = self.cal_inliers = None
        if len(self.cal_rgb) < 4:
            return
        try:
            H, rms, errors, inliers = fit_homography(self.cal_rgb, self.cal_th)
        except ValueError as ex:
            self.notify(str(ex), error=True)
            return
        self.cal_preview = Calibration(H, self.rgb.size, self.thermal_size, list(self.cal_rgb), list(self.cal_th), rms)
        self.cal_errors, self.cal_inliers = errors, inliers

    def _calibration_key(self, key, char):
        if key == KEY_SPACE:
            if not self.detector.has_background:
                self.notify("Automatic capture needs the empty-table background first: clear the table, press b.",
                            error=True)
            elif not self.temps_valid:
                self.notify("The thermal camera is not ready yet.", error=True)
            else:
                self.cal_samples, self.cal_capture_frames = [], 0
                self.notify("Capturing... keep still.")
        elif char == "u" or key in KEY_BACKSPACE:
            self._undo_pair()
        elif char == "x":
            self.cal_rgb, self.cal_th, self.cal_pending = [], [], {}
            self._update_cal_preview()
        elif key in KEY_ENTER:
            if self.cal_preview is None:
                self.notify("At least 4 point pairs are needed.", error=True)
                return
            self.calibration = self.cal_preview
            self.calibration.save(self.calibration_path)
            self.tracker.reset()
            self.mode = "run"
            self.notify(f"Calibration saved ({len(self.cal_rgb)} pairs, error {self.calibration.rms:.2f} thermal px).")
        elif key == KEY_ESC:
            self.mode = "run"
            self.notify("Left calibration without saving (the pairs are kept until you quit).")
        else:
            self._common_key(char)

    # ----- table area mode ---------------------------------------------------------------------

    def enter_roi(self):
        self.mode = "roi"
        self.roi_points = [list(p) for p in self.roi] if self.roi else []

    def _roi_key(self, key, char):
        if key in KEY_ENTER:
            self.roi = self.roi_points if len(self.roi_points) >= 3 else None
            self.detector.set_roi(self.roi)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.roi_path.write_text(json.dumps({"polygon": self.roi}, indent=2))
            self.tracker.reset()
            self.mode = "run"
            self.notify("Table area saved." if self.roi else "Table area reset to the whole image.")
        elif key == KEY_ESC:
            self.mode = "run"
        elif char == "u" or key in KEY_BACKSPACE:
            if self.roi_points:
                self.roi_points.pop()
        elif char == "x":
            self.roi_points = []
        else:
            self._common_key(char)

    # ----- input -------------------------------------------------------------------------------

    def on_key(self, key):
        if key < 0:
            return
        key &= 0xFF
        char = chr(key) if 32 < key < 127 else ""
        if self.mode == "calibrate":
            self._calibration_key(key, char)
        elif self.mode == "roi":
            self._roi_key(key, char)
        elif char == "q" or key == KEY_ESC:
            self.running = False
        elif char == "c":
            self.enter_calibration()
        elif char == "r":
            self.enter_roi()
        elif char == "s":
            self.snapshot()
        else:
            self._common_key(char)

    def _common_key(self, char):
        """Keys available in every mode."""
        if char == "b":
            self._background_frames = []
            self.notify("Capturing the empty table, keep it clear...")
        elif char == "o":
            self.overlay = not self.overlay
        elif char == "d":
            self.debug = not self.debug
        elif char == "h":
            self.show_help = not self.show_help
        elif char == "f" and self.thermal is not None:
            self.thermal.force_flag()
            self.notify("Flag (shutter) cycle requested.")

    def on_mouse(self, event, x, y, flags=0, param=None):
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        view, point = self.layout.locate(x, y)
        if view is None:
            return
        if self.mode == "calibrate":
            if event == cv2.EVENT_RBUTTONDOWN:
                self._undo_pair()
                return
            self.cal_pending[view] = point
            if "rgb" in self.cal_pending and "thermal" in self.cal_pending:
                self._add_pair(self.cal_pending.pop("rgb"), self.cal_pending.pop("thermal"))
        elif self.mode == "roi" and view == "rgb":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.roi_points.append([float(point[0]), float(point[1])])
            elif self.roi_points:
                self.roi_points.pop()

    def notify(self, message, error=False, seconds=6.0):
        print(message)
        self.message, self.message_until = message, time.monotonic() + seconds
        self.message_is_error = error

    # ----- output ------------------------------------------------------------------------------

    def snapshot(self):
        folder = self.data_dir / "snapshots"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(str(folder / f"{stamp}.png"), self.canvas)
        cv2.imwrite(str(folder / f"{stamp}_rgb.png"), self.frame)
        if self.temps is not None:
            np.save(folder / f"{stamp}_thermal.npy", self.temps)
        objects = []
        for track in self.tracks:
            record = {"id": track.id, "centroid_rgb": [round(v, 1) for v in track.obj.centroid],
                      "bbox_rgb": list(track.obj.bbox), "area_rgb_px": round(track.obj.area),
                      "touches_table_edge": track.obj.touches_border}
            if track.stats is not None:
                s = track.stats
                record.update(temperature_median=round(s.median, 2), temperature_mean=round(s.mean, 2),
                              temperature_min=round(s.min, 2), temperature_max=round(s.max, 2),
                              thermal_pixels=s.pixels, thermal_coverage=round(s.coverage, 2))
            objects.append(record)
        report = {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "table_temperature": None if self.table_temp is None else round(self.table_temp, 2),
                  "objects": objects}
        (folder / f"{stamp}.json").write_text(json.dumps(report, indent=2))
        self.notify(f"Snapshot saved: {folder / stamp}.png / .json")

    def _render(self):
        L = self.layout
        rgb_view = cv2.resize(self.frame, (L.rgb_width, L.height), interpolation=cv2.INTER_AREA)
        calibration = self.cal_preview if self.mode == "calibrate" and self.cal_preview else self.calibration

        if self.temps is not None:
            lo, hi = self.temp_range
            thermal_color = ui.colorize(self.temps, lo, hi)
            if self.overlay and calibration is not None:
                to_view = np.diag([L.rgb_scale, L.rgb_scale, 1.0]) @ calibration.H_inv
                rgb_view = ui.blend_thermal(rgb_view, thermal_color, to_view)
            thermal_view = cv2.resize(thermal_color, (L.thermal_width, L.height), interpolation=cv2.INTER_LINEAR)
            ui.draw_colorbar(thermal_view, lo, hi)
            if not self.temps_valid:
                ui.put_text(thermal_view, "Camera initializing (flag cycle)...", (16, L.height - 20), 0.6, ui.YELLOW)
        else:
            lines = ["No thermal image."]
            if self.thermal_error:
                lines += _wrap(self.thermal_error, 52)
            thermal_view = ui.placeholder((L.thermal_width, L.height), lines)

        if self.roi and self.mode != "roi":
            ui.draw_polygon(rgb_view, L.rgb_points(self.roi), ui.GREY, 1)
        if calibration is not None:
            fov = L.rgb_points(calibration.thermal_fov_in_rgb())
            ui.draw_polygon(rgb_view, fov, ui.CYAN, 1)
            ui.put_text(rgb_view, "thermal view", fov[0] + (6, 18), 0.45, ui.CYAN)

        if self.mode == "calibrate":
            self._draw_calibration(rgb_view, thermal_view)
        elif self.mode == "roi":
            self._draw_roi(rgb_view)
        else:
            self._draw_objects(rgb_view, thermal_view)
        return ui.compose(L, rgb_view, thermal_view, self._status_lines())

    def _draw_objects(self, rgb_view, thermal_view):
        L = self.layout
        for track in self.tracks:
            obj = track.obj
            color = ui.GREY if obj.touches_border else ui.object_color(track.id)
            thickness = 1 if obj.touches_border else 2
            ui.draw_polygon(rgb_view, L.rgb_points(obj.contour), color, thickness)
            title = f"#{track.id}" + (" (edge)" if obj.touches_border else "")
            lines = [title]
            if track.temp is not None:
                lines[0] = f"{title}  {track.temp:.1f} C"
                lines.append(f"max {track.temp_max:.1f}  min {track.stats.min:.1f}")
                if track.stats.coverage < 0.6:
                    lines.append("partly outside thermal view")
            elif self.calibration is None:
                lines.append("not calibrated")
            elif self.temps_valid:
                lines.append("outside thermal view")
            x, y, _, _ = obj.bbox
            ui.draw_label(rgb_view, lines, (x * L.rgb_scale, y * L.rgb_scale - 4), color)
            if track.thermal_polygon is not None:
                polygon = L.thermal_points(track.thermal_polygon)
                ui.draw_polygon(thermal_view, polygon, color, thickness)
                if track.temp is not None:
                    x0, y0 = polygon.min(axis=0)
                    ui.put_text(thermal_view, f"#{track.id} {track.temp:.1f}", (max(x0, 2), max(y0 - 5, 14)), 0.5,
                                color)

    def _draw_calibration(self, rgb_view, thermal_view):
        L = self.layout
        for i, (p_rgb, p_th) in enumerate(zip(self.cal_rgb, self.cal_th)):
            outlier = self.cal_inliers is not None and not self.cal_inliers[i]
            color = ui.RED if outlier else ui.object_color(i)
            ui.draw_numbered_point(rgb_view, L.rgb_points([p_rgb])[0], i + 1, color)
            th = L.thermal_points([p_th])[0]
            ui.draw_numbered_point(thermal_view, th, i + 1, color)
            if self.cal_preview is not None:
                # Where the fitted homography sends the webcam point: should land on the thermal point
                projected = L.thermal_points(self.cal_preview.rgb_to_thermal([p_rgb]))[0]
                cv2.line(thermal_view, tuple(np.round(th).astype(int)), tuple(np.round(projected).astype(int)),
                         color, 1, cv2.LINE_AA)
                cv2.circle(thermal_view, tuple(np.round(projected).astype(int)), 3, ui.WHITE, 1, cv2.LINE_AA)
        if "rgb" in self.cal_pending:
            ui.draw_numbered_point(rgb_view, L.rgb_points([self.cal_pending["rgb"]])[0], None, ui.YELLOW, False)
        if "thermal" in self.cal_pending:
            ui.draw_numbered_point(thermal_view, L.thermal_points([self.cal_pending["thermal"]])[0], None, ui.YELLOW,
                                   False)
        if self.cal_live_rgb is not None:
            ui.draw_cross(rgb_view, L.rgb_points([self.cal_live_rgb])[0], ui.YELLOW, 18)
        if self.cal_live_th is not None:
            ui.draw_cross(thermal_view, L.thermal_points([self.cal_live_th])[0], ui.CYAN, 18)
        for obj in self.objects:
            ui.draw_polygon(rgb_view, L.rgb_points(obj.contour), ui.GREY if obj.touches_border else ui.YELLOW, 1)

        if self.show_help:
            rms = f", error {self.cal_preview.rms:.2f} px" if self.cal_preview else ""
            lines = [f"CALIBRATION - {len(self.cal_rgb)} pairs{rms}",
                     "Auto: put ONE warm or cold object on the table,",
                     "  take your hand away, press SPACE. Repeat at 6+ places.",
                     "Manual: click a point here, then the same point",
                     "  in the thermal view (right click: undo).",
                     "Spread the pairs over the whole table.",
                     "u: undo  x: clear  o: overlay  ENTER: save  ESC: cancel"]
            if not self.detector.has_background or self.thermal_reference is None:
                lines.insert(1, "(auto mode: clear the table and press b first)")
            ui.instructions_box(rgb_view, lines)

    def _draw_roi(self, rgb_view):
        L = self.layout
        points = L.rgb_points(self.roi_points) if self.roi_points else []
        if len(points) >= 2:
            ui.draw_polygon(rgb_view, points, ui.YELLOW, 2, closed=len(points) >= 3)
        for i, p in enumerate(points):
            ui.draw_numbered_point(rgb_view, p, i + 1, ui.YELLOW)
        if self.show_help:
            ui.instructions_box(rgb_view, ["TABLE AREA - objects are only detected inside it",
                                           "Click the corners of the usable table surface.",
                                           "Right click / u: undo   x: clear",
                                           "ENTER: save (fewer than 3 points = whole image)",
                                           "ESC: cancel"])

    def _status_lines(self):
        orange = (0, 165, 255)
        if self.thermal is None:
            thermal = "thermal: unavailable"
        else:
            thermal = f"thermal: {self.thermal.status}, flag {self.thermal.flag_state}"
        table = f"table {self.table_temp:.1f} C" if self.table_temp is not None else "table -"
        n_objects = sum(1 for t in self.tracks if not t.obj.touches_border)
        calibration = f"calibration error {self.calibration.rms:.2f} px" if self.calibration else "not calibrated"
        line1 = (f"{self.mode.upper()}   objects: {n_objects}   {table}   {thermal}   {calibration}", ui.WHITE)

        if self.message and time.monotonic() < self.message_until:
            line2 = (self.message, orange if self.message_is_error else ui.GREEN)
        elif self._background_frames is not None:
            line2 = (f"Capturing empty table ({len(self._background_frames)}/{BACKGROUND_FRAMES})...", ui.YELLOW)
        elif self.roi is None and self.mode == "run":
            line2 = ("Step 1: press r and click the corners of the table (otherwise the whole image is watched).",
                     orange)
        elif not self.detector.has_background:
            line2 = ("Step 2: clear the table and press b to capture the empty-table background.", orange)
        elif self.calibration is None and self.mode == "run":
            line2 = ("Step 3: press c to calibrate the webcam against the thermal camera.", orange)
        else:
            line2 = ("Place objects on the table.", ui.GREY)
        keys = RUN_KEYS if self.mode == "run" else "h: hide/show help   b: background   o: overlay   f: flag"
        return [line1, line2, (keys, ui.GREY)]


def _wrap(text, width):
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + ([line] if line else [])


def run(app):
    """Shows the app in an OpenCV window until the user quits."""
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(WINDOW, app.on_mouse)
    sized = debug_open = False
    while app.running:
        canvas = app.step()
        if canvas is not None:
            cv2.imshow(WINDOW, canvas)
            if not sized:
                cv2.resizeWindow(WINDOW, canvas.shape[1], canvas.shape[0])
                sized = True
        if app.debug and not debug_open:
            _open_debug_window(app.detector)
            debug_open = True
        elif not app.debug and debug_open:
            cv2.destroyWindow(DEBUG_WINDOW)
            debug_open = False
        if debug_open and app.detector.last_score is not None:
            score = np.clip(app.detector.last_score / 2.0, 0, 1)
            view = cv2.applyColorMap((score * 255).astype(np.uint8), cv2.COLORMAP_JET)
            ui.put_text(view, "change score (1.0 = threshold, at the middle of the scale)", (8, 20), 0.45)
            cv2.imshow(DEBUG_WINDOW, view)
        app.on_key(cv2.waitKey(5))
        if sized and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def _open_debug_window(detector):
    """Score view with sliders to tune the detection thresholds live."""
    cfg = detector.config
    cv2.namedWindow(DEBUG_WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)

    def slider(name, attribute, maximum, minimum=1):
        def changed(value):
            setattr(cfg, attribute, max(minimum, value))
        cv2.createTrackbar(name, DEBUG_WINDOW, int(getattr(cfg, attribute)), maximum, changed)

    slider("color change", "chroma_thresh", 60)
    slider("darker by", "dark_thresh", 150)
    slider("brighter by", "light_thresh", 150)
    slider("min area", "min_area", 2000, 10)
