#!/usr/bin/env python3
"""Pushes the objects hotter than a threshold off the thermal table with the Franka arm.

The camera app and frankapy need different Python environments (the .venv with Python 3.13 and the
Optris SDK, and the ROS Python of frankapy), so this file runs as two processes on the same machine:

    .venv/bin/python push_hot_objects.py camera --threshold 40    # thermal table window + object server
    python3 push_hot_objects.py robot                              # in the frankapy environment

The camera side is the thermal table app. It also finds the ArUco markers at the table corners in the
webcam image, maps the detected objects into robot coordinates and serves them on a local TCP port.
The robot side waits at home, asks for the objects, pushes the hottest one above the threshold
sideways out of the table area, goes home again so the cameras see the table, and repeats.

To check the setup without risk, start the robot side with --dry-run (no motion, prints the pushes it
would do) and then --hover-only (follows each push path 5 cm above the table).
"""

import argparse
import json
import math
import socket
import socketserver
import threading
import time
from pathlib import Path

import numpy as np

try:  # camera side (.venv)
    import cv2
    from thermal_table import ui
    from thermal_table.app import STABLE_AGE, App, run
    from thermal_table.cameras import RgbCamera, SimulatedThermalCamera, ThermalCamera, ThermalUnavailable
except ImportError:
    cv2 = None

try:  # robot side (frankapy environment)
    from autolab_core import RigidTransform
    from frankapy import FrankaArm
except ImportError:
    FrankaArm = None

HERE = Path(__file__).resolve().parent
HOST = "127.0.0.1"         # both sides run on the same machine
DEFAULT_PORT = 5055

# ----- table geometry ---------------------------------------------------------------------------
# Robot frame = frankapy "world": origin on the axis of joint 1, x away from the robot across the
# table, y to the robot's left, z up. The webcam is upside down, so in its image x points up and y left.
#
# The four ArUco markers (DICT_4X4_50) at the table corners give the webcam -> robot mapping. Each
# one is used through its inner corner, the one facing the table centre. Positions from ref/:
#   - the robot-side edge of markers 0 and 1 is 25.5 cm in front of the robot base, whose axis is
#     8 cm behind its front surface, and the robot axis passes midway between them;
#   - the markers are 4.7 cm wide (measured on the photos), so their inner corners are 4.7 cm farther;
#   - inner corners 0-1 are 47.5 cm apart, 0-2 47 cm, 1-3 45.5 cm and 2-3 41.5 cm. These lengths leave
#     the sideways position of the far pair open: shifting it 5.5 cm towards the robot's right is what
#     makes the markers come out square in both the measurement photo and the webcam image.
# Replace them with measured robot positions (test_old/get_coords.py) if pushes miss their target.
MARKER_CORNERS = {
    0: (0.382, -0.2375),   # near, robot's right (bottom right in the webcam image)
    1: (0.382, 0.2375),    # near, robot's left (bottom left)
    2: (0.849, -0.1825),   # far, robot's right (top right)
    3: (0.837, 0.2325),    # far, robot's left (top left)
}
MARKER_CHECK_PERIOD = 2.0  # s between marker detections; the last mapping is kept while markers are hidden

# ----- pushing ----------------------------------------------------------------------------------
TABLE_Z = 0.0              # table surface height in the robot frame
PUSH_HEIGHT = 0.01         # fingertip height above the table while pushing
SAFE_HEIGHT = 0.12         # travel height above the table, clear of the objects
HOVER_HEIGHT = 0.05        # --hover-only: the push path is followed at this height
MAX_REACH = 0.77           # m from the robot axis. Pointing down at table height, the arm is fully
                           # stretched at 0.795 m (Panda kinematics), so the far strip is out of reach.
FINGER_HALF_WIDTH = 0.01   # half the width of the closed fingertips
APPROACH_GAP = 0.025       # gap left between fingertips and object when the gripper comes down
PUSH_PAST_EDGE = 0.035     # the fingertips stop this far beyond the line of inner marker corners, so the
                           # object in front of them ends past the detection area (it ends at marker centres)
PUSH_SPEED = 0.08          # m/s
SETTLE_TIME = 1.5          # s waited at home, so the camera app tracks the objects again without the arm
POLL_PERIOD = 0.5          # s between requests when there is nothing to push
GRIPPER_DOWN = np.array([  # same orientation as the sweep in test_old/test.py
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
])


# ----- push planning (pure geometry, shared by both sides) --------------------------------------

def edge_y(x, side):
    """y of a side edge of the table area (line through the inner marker corners) at distance x.

    side: +1 for the robot's left, -1 for its right.
    """
    (x0, y0), (x1, y1) = (MARKER_CORNERS[1], MARKER_CORNERS[3]) if side > 0 else (MARKER_CORNERS[0], MARKER_CORNERS[2])
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def plan_push(target, others):
    """Fingertip path (start, end) that pushes target sideways out of the table area, or None.

    target and others are objects as served by the camera side ("center" and "outline" in robot
    coordinates). The object goes to the nearer side, unless another object is in the way only on
    that side. Near the far end of the table, the end point is pulled back within reach, so the push
    goes diagonally towards the robot.
    """
    center = np.asarray(target["center"], float)
    outline = np.asarray(target["outline"], float)
    plans = []
    for side in (1, -1):
        y_end = edge_y(center[0], side) + side * PUSH_PAST_EDGE
        if side * (y_end - center[1]) <= 0 or abs(y_end) >= MAX_REACH:
            continue
        end = np.array([min(center[0], math.sqrt(MAX_REACH ** 2 - y_end ** 2)), y_end])
        length = float(np.linalg.norm(end - center))
        direction = (end - center) / length
        along, across = _project(outline, center, direction)
        start = center - direction * (max(0.0, -along.min()) + FINGER_HALF_WIDTH + APPROACH_GAP)
        if np.hypot(*start) > MAX_REACH:
            continue
        if any(_distance(start, np.asarray(o["outline"], float)) < APPROACH_GAP for o in others):
            continue  # the fingers would come down on (or within 1.5 cm of) another object
        # Objects the target would run into on its way
        half_width = max(np.abs(across).max(), FINGER_HALF_WIDTH)
        blocked = False
        for other in others:
            o_along, o_across = _project(np.asarray(other["outline"], float), center, direction)
            blocked |= bool(np.any((o_along > 0) & (o_along < length + along.max()) & (np.abs(o_across) < half_width)))
        plans.append((blocked, length, start, end))
    if not plans:
        return None
    _, _, start, end = min(plans, key=lambda plan: plan[:2])
    return start, end


def _project(points, origin, direction):
    """Coordinates of points along direction and across it, from origin."""
    offsets = points - origin
    return offsets @ direction, offsets @ np.array([-direction[1], direction[0]])


def _distance(point, polygon):
    """Distance from a point to a convex polygon (0 inside it)."""
    a = polygon
    ab = np.roll(polygon, -1, axis=0) - a
    t = np.clip(np.sum((point - a) * ab, axis=1) / np.maximum(np.sum(ab * ab, axis=1), 1e-12), 0, 1)
    distance = float(np.min(np.linalg.norm(a + t[:, None] * ab - point, axis=1)))
    cross = ab[:, 0] * (point - a)[:, 1] - ab[:, 1] * (point - a)[:, 0]
    inside = len(polygon) >= 3 and (np.all(cross >= 0) or np.all(cross <= 0))
    return 0.0 if inside else distance


# ----- camera side ------------------------------------------------------------------------------

def find_marker_corners(image):
    """Inner corner (webcam pixels) of each table marker, or None unless all of them are visible."""
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), parameters)
    corners, ids, _ = detector.detectMarkers(image)
    found = {} if ids is None else {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}
    if not all(i in found for i in MARKER_CORNERS):
        return None
    centre = np.mean([found[i].mean(axis=0) for i in MARKER_CORNERS], axis=0)
    return {i: found[i][np.argmin(np.linalg.norm(found[i] - centre, axis=1))].tolist() for i in MARKER_CORNERS}


class TableMapping:
    """Webcam pixels <-> robot (x, y) on the table plane, from the inner corners of the table markers.

    The last marker positions are saved, so the mapping is available at startup even if the arm or
    an object hides a marker.
    """

    def __init__(self, path, image_size):
        self.path = Path(path)
        self.image_size = list(image_size)
        self.corners = None           # {marker id: [u, v]} inner corners in webcam pixels
        self.H = self.H_inv = None    # pixels -> robot, robot -> pixels
        if self.path.exists():
            data = json.loads(self.path.read_text())
            if data["image_size"] == self.image_size:
                self._set({int(i): p for i, p in data["corners"].items()})

    @property
    def ready(self):
        return self.H is not None

    def update(self, frame):
        """Looks for the markers in a webcam frame. Returns whether all of them were found."""
        corners = find_marker_corners(frame)
        if corners is None:
            return False
        if self.corners is None or max(np.hypot(*np.subtract(corners[i], self.corners[i])) for i in corners) > 2.0:
            moved = self.corners is not None
            self._set(corners)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"image_size": self.image_size, "corners": corners,
                                             "saved": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
            print("Table markers found: webcam -> robot mapping " + ("updated (did the webcam move?)" if moved else "set"))
        return True

    def _set(self, corners):
        ids = sorted(MARKER_CORNERS)
        self.corners = corners
        self.H = cv2.getPerspectiveTransform(np.float32([corners[i] for i in ids]),
                                             np.float32([MARKER_CORNERS[i] for i in ids]))
        self.H_inv = np.linalg.inv(self.H)

    def to_robot(self, points):
        return cv2.perspectiveTransform(np.asarray(points, np.float64).reshape(-1, 1, 2), self.H).reshape(-1, 2)

    def to_pixels(self, points):
        return cv2.perspectiveTransform(np.asarray(points, np.float64).reshape(-1, 1, 2), self.H_inv).reshape(-1, 2)


class ReportServer(socketserver.ThreadingTCPServer):
    """Answers "objects" with the latest report of the camera side, as one line of JSON."""
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, link):
        super().__init__(address, _ReportRequest)
        self.link = link


class _ReportRequest(socketserver.StreamRequestHandler):
    def handle(self):
        if self.rfile.readline().strip() == b"objects":
            self.wfile.write(json.dumps(self.server.link.report).encode() + b"\n")


class RobotLink:
    """Camera side: puts the tracked objects in robot coordinates, shows them and serves them."""

    def __init__(self, app, mapping, threshold, use_max, port):
        self.app = app
        self.mapping = mapping
        self.threshold = threshold
        self.use_max = use_max
        self.report = {"objects": [], "error": "camera side starting"}
        self.markers_visible = False
        self._next_marker_check = 0.0
        self.server = ReportServer((HOST, port), self)
        threading.Thread(target=self.server.serve_forever, name="robot-link", daemon=True).start()
        print(f"Serving the objects to the robot side on port {port}")

    def update(self, canvas):
        """Called after each app step: refreshes the mapping and the report, draws on the canvas."""
        app = self.app
        if time.monotonic() >= self._next_marker_check:
            self._next_marker_check = time.monotonic() + MARKER_CHECK_PERIOD
            self.markers_visible = self.mapping.update(app.frame)

        tracks, objects = [], []
        if self.mapping.ready:
            for track in app.tracks:
                if not track.obj.touches_border and track.age >= STABLE_AGE:
                    tracks.append(track)
                    objects.append(self._describe(track))
        if not self.mapping.ready:
            error = "the webcam has not seen the four table markers yet"
        elif app.mode != "run":
            error = f"the camera app is in {app.mode} mode"   # e.g. a hot mug used for calibration
        else:
            error = None
        self.report = {"objects": [] if error else objects, "error": error}

        if app.mode == "run" and self.mapping.ready:
            self._draw_mapping(canvas)
            for track, obj in zip(tracks, objects):
                self._draw_object(canvas, track, obj, [o for o in objects if o is not obj])
        status = f"robot link: {error}" if error else f"robot link: pushing objects above {self.threshold:g} C"
        if self.mapping.ready and not self.markers_visible:
            status += " (markers hidden, last mapping)"
        width = cv2.getTextSize(status, ui.FONT, 0.5, 1)[0][0]
        ui.put_text(canvas, status, (app.layout.rgb_width - width - 10, 20), 0.5, (0, 165, 255) if error else ui.GREEN)

    def _describe(self, track):
        temp = track.temp_max if self.use_max else track.temp
        hull = cv2.convexHull(track.obj.contour).reshape(-1, 2)
        return {
            "id": track.id,
            "temp": None if temp is None else round(temp, 2),
            "hot": temp is not None and temp >= self.threshold,
            "center": np.round(self.mapping.to_robot([track.obj.centroid])[0], 4).tolist(),
            "outline": np.round(self.mapping.to_robot(hull), 4).tolist(),
        }

    def _draw_mapping(self, canvas):
        L = self.app.layout
        for corner in self.mapping.corners.values():
            ui.draw_cross(canvas, L.rgb_points([corner])[0], ui.CYAN if self.markers_visible else ui.GREY, 14, 1)
        # Reach limit of the arm at table height, only the part over the table
        arc = [[MAX_REACH * math.cos(a), MAX_REACH * math.sin(a)] for a in np.linspace(-1.2, 1.2, 49)]
        ui.draw_polygon(canvas, L.rgb_points(self.mapping.to_pixels(arc)), ui.YELLOW, 1, closed=False)
        label = L.rgb_points(self.mapping.to_pixels([[MAX_REACH, 0.0]]))[0]
        ui.put_text(canvas, "robot reach", (label[0] - 40, label[1] - 6), 0.45, ui.YELLOW)

    def _draw_object(self, canvas, track, obj, others):
        L = self.app.layout
        x, y, _, h = track.obj.bbox
        text = f"x {obj['center'][0]:.2f}  y {obj['center'][1]:+.2f} m"
        if obj["hot"]:
            text += "  HOT: " + ("push" if plan_push(obj, others) is not None else "out of reach")
        ui.put_text(canvas, text, (x * L.rgb_scale, (y + h) * L.rgb_scale + 16), 0.45, ui.RED if obj["hot"] else ui.WHITE)


def camera_main(args):
    if cv2 is None:
        raise SystemExit("The camera side needs the thermal table environment: .venv/bin/python push_hot_objects.py camera")
    from thermal_table.__main__ import DEFAULT_RGB_DEVICE, DEFAULT_THERMAL_SERIAL, configure_qt

    configure_qt()
    rgb = RgbCamera(args.rgb_device or DEFAULT_RGB_DEVICE)
    try:
        rgb.start()
    except RuntimeError as ex:
        raise SystemExit(ex)
    thermal, thermal_error, data_dir = None, None, args.data_dir
    if args.simulate_thermal:
        thermal = SimulatedThermalCamera(rgb).start()
        data_dir = data_dir / "simulated"   # never mix a simulated calibration with the real one
    else:
        serial = DEFAULT_THERMAL_SERIAL if args.thermal_serial is None else args.thermal_serial
        try:
            thermal = ThermalCamera(serial, args.emissivity).start()
            print(f"Thermal camera: {thermal.description}")
        except ThermalUnavailable as ex:
            thermal_error = str(ex)
            print(f"WARNING: {thermal_error}\nNo temperatures, so nothing will be pushed.")

    app = App(rgb, thermal, data_dir, thermal_error)
    # The markers belong to the webcam setup, so the mapping is shared with --simulate-thermal
    link = RobotLink(app, TableMapping(args.data_dir / "robot_markers.json", rgb.size), args.threshold,
                     args.use_max, args.port)
    step = app.step

    def step_and_publish():
        canvas = step()
        if canvas is not None:
            link.update(canvas)
        return canvas

    app.step = step_and_publish   # run() calls app.step() for every frame
    try:
        run(app)
    except KeyboardInterrupt:
        pass
    finally:
        link.server.shutdown()
        link.server.server_close()
        if thermal is not None:
            thermal.stop()
        rgb.stop()


# ----- robot side -------------------------------------------------------------------------------

def fetch_report(port):
    with socket.create_connection((HOST, port), timeout=2.0) as sock:
        sock.sendall(b"objects\n")
        with sock.makefile("rb") as reply:
            return json.loads(reply.readline().decode())


def gripper_pose(xy, height):
    return RigidTransform(rotation=GRIPPER_DOWN, translation=np.array([xy[0], xy[1], TABLE_Z + height]),
                          from_frame="franka_tool", to_frame="world")


def push(fa, start, end, height):
    """Comes down next to the object, pushes it from start to end, and goes back home."""
    fa.goto_pose(gripper_pose(start, SAFE_HEIGHT), duration=3.0)
    fa.goto_pose(gripper_pose(start, height), duration=1.5)
    fa.goto_pose(gripper_pose(end, height), duration=max(1.0, float(np.linalg.norm(end - start)) / PUSH_SPEED))
    fa.goto_pose(gripper_pose(end, SAFE_HEIGHT), duration=1.5)
    fa.reset_joints()


def robot_main(args):
    fa = None
    if not args.dry_run:
        if FrankaArm is None:
            raise SystemExit("frankapy not found: run the robot side in the frankapy environment (or use --dry-run).")
        fa = FrankaArm()
        fa.reset_joints()
        fa.close_gripper()
    done = set()   # camera track ids not to handle again
    last_problem = None
    print("Waiting for hot objects (Ctrl+C to stop)...")
    try:
        while True:
            try:
                report = fetch_report(args.port)
                problem = report["error"]
            except (OSError, ValueError) as ex:
                problem = f"camera side not reachable on port {args.port} ({ex})"
            if problem != last_problem:
                print(f"Waiting: {problem}" if problem else "Camera side ready.")
                last_problem = problem
                done.clear()   # track ids start again from 1 when the camera side restarts
            if problem:
                time.sleep(1.0)
                continue

            hot = [o for o in report["objects"] if o["hot"] and o["id"] not in done]
            if not hot:
                time.sleep(POLL_PERIOD)
                continue
            target = max(hot, key=lambda o: o["temp"])
            path = plan_push(target, [o for o in report["objects"] if o is not target])
            description = "object #{} ({:.1f} C) at x={:.3f} y={:.3f} m".format(target["id"], target["temp"],
                                                                               *target["center"])
            if path is None:
                print(f"Leaving {description}: out of reach, or other objects are in the way.")
                done.add(target["id"])
                continue
            start, end = path
            print(f"Pushing {description}: fingertips from ({start[0]:.3f}, {start[1]:.3f}) "
                  f"to ({end[0]:.3f}, {end[1]:.3f})")
            if fa is None or args.hover_only:
                done.add(target["id"])   # the object does not move, do not do it again
            if fa is not None:
                push(fa, start, end, HOVER_HEIGHT if args.hover_only else PUSH_HEIGHT)
                time.sleep(SETTLE_TIME)
    except KeyboardInterrupt:
        print("Stopped.")
        if fa is not None:
            fa.stop_skill()


def main():
    parser = argparse.ArgumentParser(description="Push the objects hotter than a threshold off the table.")
    sides = parser.add_subparsers(dest="side")
    sides.required = True

    camera = sides.add_parser("camera", help="thermal table app that serves the objects (run with .venv/bin/python)")
    camera.add_argument("--threshold", type=float, default=40.0,
                        help="objects at least this hot (degrees C) are pushed (default: 40)")
    camera.add_argument("--use-max", action="store_true",
                        help="compare the hottest part of each object with the threshold instead of its median")
    camera.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"TCP port (default: {DEFAULT_PORT})")
    camera.add_argument("--rgb-device", help="webcam device path or index (default: the Logitech C930e)")
    camera.add_argument("--thermal-serial", type=int, help="serial number of the Optris camera, 0 = first one found")
    camera.add_argument("--emissivity", type=float, help="emissivity used by the thermal camera")
    camera.add_argument("--simulate-thermal", action="store_true",
                        help="fake the thermal camera from the webcam image (dark = hot)")
    camera.add_argument("--data-dir", type=Path, default=HERE / "data",
                        help="where the camera app's setup and the marker positions are stored")

    robot = sides.add_parser("robot", help="pushes the hot objects (run in the frankapy environment)")
    robot.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"TCP port (default: {DEFAULT_PORT})")
    check = robot.add_mutually_exclusive_group()
    check.add_argument("--dry-run", action="store_true", help="do not move, only print the pushes")
    check.add_argument("--hover-only", action="store_true",
                       help=f"follow each push path {HOVER_HEIGHT * 100:.0f} cm above the table instead of pushing")

    args = parser.parse_args()
    if args.side == "camera":
        camera_main(args)
    else:
        robot_main(args)


if __name__ == "__main__":
    main()
