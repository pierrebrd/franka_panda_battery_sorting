#!/usr/bin/env python3
"""Picks the objects hotter than a threshold off the thermal table with the Franka arm.

The camera app and frankapy need different Python environments (the .venv with Python 3.13 and the
Optris SDK, and the ROS Python of frankapy), so this file runs as two processes on the same machine:

    .venv/bin/python push_hot_objects.py camera --threshold 40    # thermal table window + object server
    python3 push_hot_objects.py robot                              # in the frankapy environment

The camera side is the thermal table app. It also finds the ArUco markers at the table corners in the
webcam image, maps the detected objects into robot coordinates and serves them on a local TCP port.
The robot side waits at home, asks for the objects, grips the hottest one above the threshold, sets
it down just outside the nearer side of the table area, goes home again so the cameras see the table,
and repeats.

To check the setup without risk, start the robot side with --dry-run (no motion, prints the grasps it
would do) and then --hover-only (follows each grasp path 5 cm above the table, without gripping).
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
    import rospy
    from autolab_core import RigidTransform
    from frankapy import FrankaArm
    from sensor_msgs.msg import JointState
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
# one is used through its inner corner, the one facing the table centre, whose position was measured
# with the robot (gripper tip on the corner, pose read with test_old/get_coords.py). The table is not
# square to the robot: it is shifted ~2 cm to the robot's right and turned by a few degrees.
MARKER_CORNERS = {
    0: (0.36027, -0.2586),   # near, robot's right
    1: (0.38831, 0.221),     # near, robot's left
    2: (0.8248, -0.22999),   # far, robot's right
    3: (0.83612, 0.18129),   # far, robot's left
}
MARKER_PLACES = {0: "bottom right", 1: "bottom left", 2: "top right", 3: "top left"}  # in the webcam image
MARKER_CHECK_PERIOD = 2.0  # s between marker detections; the last mapping is kept while markers are hidden

# ----- gripping ---------------------------------------------------------------------------------
TABLE_Z = 0.02             # table surface height in the robot frame (gripper tip on the markers: 0.019 to 0.023)
GRASP_HEIGHT = 0.01        # fingertip height above the table when gripping and setting down
SAFE_HEIGHT = 0.12         # travel height above the table, clear of the objects
HOVER_HEIGHT = 0.05        # --hover-only: the grasp path is followed at this height
MAX_REACH = 0.77           # m from the robot axis. Pointing down at table height, the arm is fully
                           # stretched at 0.795 m (Panda kinematics), so the far strip is out of reach.
MAX_OPENING = 0.08         # Franka Hand stroke (distance between the fingers fully open)
FINGER_HALF_WIDTH = 0.01   # half the size of a fingertip, seen from above
GRIP_MARGIN = 0.012        # gap left on each side between fingers and object when the gripper comes down
PLACE_PAST_EDGE = -0.02     # the object is set down this far beyond the line of inner marker corners,
                           # so it is out of the detection area (it ends at marker centres)
GRIP_FORCE = 20.0          # N
MIN_GRIP_WIDTH = 0.002     # fingers closer than this after gripping: they missed the object
GRIPPER_READ_TIMEOUT = 1.0 # s waited for the finger width; without it, the grasp is assumed to have worked
MAX_ATTEMPTS = 2           # grasps tried on the same object before leaving it
SETTLE_TIME = 1.5          # s waited at home, so the camera app tracks the objects again without the arm
POLL_PERIOD = 0.5          # s between requests when there is nothing to pick
GRIPPER_DOWN = np.array([  # same orientation as the sweep in test_old/test.py
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
])


# ----- grasp planning (pure geometry, shared by both sides) -------------------------------------

def edge_y(x, side):
    """y of a side edge of the table area (line through the inner marker corners) at distance x.

    side: +1 for the robot's left, -1 for its right.
    """
    (x0, y0), (x1, y1) = (MARKER_CORNERS[1], MARKER_CORNERS[3]) if side > 0 else (MARKER_CORNERS[0], MARKER_CORNERS[2])
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def plan_grasp(target, others):
    """How to pick target and where to set it down, or None if it cannot be done.

    target and others are objects as served by the camera side ("center" and "outline" in robot
    coordinates). Returns a dict with "grasp" and "place" (fingertip centre, x y), "closing" (unit
    vector along which the fingers close), "width" (object width between the fingers) and "opening"
    (finger opening when coming down).

    The fingers close across the narrowest width of the object that fits in the gripper with both
    fingers coming down clear of the other objects. The object is set down just outside the nearer
    side of the table area, pulled back towards the robot if that is out of reach.
    """
    outline = np.asarray(target["outline"], float)
    others = [np.asarray(o["outline"], float) for o in others]
    grasps = []
    for direction in _edge_normals(outline):
        along, _ = _project(outline, np.zeros(2), direction)
        width = float(along.max() - along.min())
        opening = width + 2 * GRIP_MARGIN
        if opening > MAX_OPENING:
            continue
        # Fingertip centre: midway between the two sides of the object, through its centre otherwise
        center = np.asarray(target["center"], float)
        grasp = center + direction * ((along.max() + along.min()) / 2 - center @ direction)
        if np.hypot(*grasp) > MAX_REACH:
            continue
        fingers = [grasp + sign * direction * (opening / 2 + FINGER_HALF_WIDTH) for sign in (1, -1)]
        if any(_distance(f, o) < FINGER_HALF_WIDTH for f in fingers for o in others):
            continue  # a finger would come down on (or right next to) another object
        grasps.append((width, opening, grasp, direction))
    if not grasps:
        return None
    width, opening, grasp, direction = min(grasps, key=lambda g: g[0])

    places = []
    for side in (1, -1):
        # The whole object, carried as it is gripped, must end PLACE_PAST_EDGE beyond the edge
        inward = float(np.max(side * (grasp[1] - outline[:, 1])))
        x = grasp[0]
        for _ in range(5):   # pulling x back moves the slanted edge a little, so y is updated and so on
            y = edge_y(x, side) + side * (PLACE_PAST_EDGE + inward)
            if abs(y) >= MAX_REACH:
                break
            x = min(grasp[0], math.sqrt(MAX_REACH ** 2 - y ** 2))
        place = np.array([x, edge_y(x, side) + side * (PLACE_PAST_EDGE + inward)])
        if np.hypot(*place) > MAX_REACH + 1e-4:
            continue
        places.append((float(np.linalg.norm(place - grasp)), place))
    if not places:
        return None
    return {"grasp": grasp, "place": min(places, key=lambda p: p[0])[1], "closing": direction,
            "width": width, "opening": opening}


def _edge_normals(polygon):
    """Unit normals of the edges of a polygon (the narrowest width of a convex one is along one of them)."""
    edges = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(edges, axis=1)
    edges = edges[lengths > 1e-9] / lengths[lengths > 1e-9, None]
    return np.stack([-edges[:, 1], edges[:, 0]], axis=1)


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

def detect_table_markers(image):
    """Outline (4x2 webcam pixels) of each table marker visible in the image, by marker id."""
    parameters = cv2.aruco.DetectorParameters()
    # More tolerant than the defaults, which miss the small, slightly blurred markers of the webcam
    # image. A wrong detection cannot slip through: all four must be found, in the right arrangement.
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.polygonalApproxAccuracyRate = 0.08
    parameters.minMarkerPerimeterRate = 0.01
    parameters.minCornerDistanceRate = 0.01
    parameters.perspectiveRemovePixelPerCell = 8
    parameters.errorCorrectionRate = 1.0
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), parameters)
    found = {}
    for scale in (1.0, 0.5):   # a blurred marker is sometimes only decoded at one of the scales
        scaled = image if scale == 1.0 else cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        corners, ids, _ = detector.detectMarkers(scaled)
        for i, outline in zip([] if ids is None else ids.ravel().tolist(), corners):
            if i in MARKER_CORNERS and i not in found:
                found[i] = outline.reshape(4, 2) / scale
    return found


def inner_corners(markers):
    """Inner corner of each table marker, or None unless all four are found in the expected arrangement."""
    if len(markers) < len(MARKER_CORNERS):
        return None
    centre = np.mean([outline.mean(axis=0) for outline in markers.values()], axis=0)
    inner = {i: outline[np.argmin(np.linalg.norm(outline - centre, axis=1))].tolist() for i, outline in markers.items()}
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = (inner[i] for i in range(4))
    if not (x3 < x2 and x1 < x0 and y3 < y1 and y2 < y0):
        return None
    return inner


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
        """Looks for the markers in a webcam frame and updates the mapping when all of them are found.

        Returns the markers seen, as detect_table_markers().
        """
        markers = detect_table_markers(frame)
        corners = inner_corners(markers)
        if corners is None:
            return markers
        if self.corners is None or max(np.hypot(*np.subtract(corners[i], self.corners[i])) for i in corners) > 2.0:
            moved = self.corners is not None
            self._set(corners)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"image_size": self.image_size, "corners": corners,
                                             "saved": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
            print("Table markers found: webcam -> robot mapping " + ("updated (did the webcam move?)" if moved else "set"))
        return markers

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
        self.markers_seen = {}         # last detect_table_markers() result
        self._next_marker_check = 0.0
        self.server = ReportServer((HOST, port), self)
        threading.Thread(target=self.server.serve_forever, name="robot-link", daemon=True).start()
        print(f"Serving the objects to the robot side on port {port}")

    def update(self, canvas):
        """Called after each app step: refreshes the mapping and the report, draws on the canvas."""
        app = self.app
        if time.monotonic() >= self._next_marker_check:
            self._next_marker_check = time.monotonic() + MARKER_CHECK_PERIOD
            self.markers_seen = self.mapping.update(app.frame)

        tracks, objects = [], []
        if self.mapping.ready:
            for track in app.tracks:
                if not track.obj.touches_border and track.age >= STABLE_AGE:
                    tracks.append(track)
                    objects.append(self._describe(track))
        missing = [i for i in sorted(MARKER_CORNERS) if i not in self.markers_seen]
        if not self.mapping.ready and missing:
            error = "the webcam cannot read marker " + ", ".join(f"{i} ({MARKER_PLACES[i]})" for i in missing)
        elif not self.mapping.ready:
            error = "the markers are not where expected (is the webcam upside down?)"
        elif app.mode != "run":
            error = f"the camera app is in {app.mode} mode"   # e.g. a hot mug used for calibration
        else:
            error = None
        self.report = {"objects": [] if error else objects, "error": error}

        if app.mode == "run":
            self._draw_markers(canvas)
            if self.mapping.ready:
                self._draw_reach(canvas)
                for track, obj in zip(tracks, objects):
                    self._draw_object(canvas, track, obj, [o for o in objects if o is not obj])
        status = f"robot link: {error}" if error else f"robot link: picking objects above {self.threshold:g} C"
        if self.mapping.ready and missing:
            status += " (saved mapping, marker {} hidden)".format(", ".join(map(str, missing)))
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

    def _draw_markers(self, canvas):
        L = self.app.layout
        for i, outline in self.markers_seen.items():
            outline = L.rgb_points(outline)
            ui.draw_polygon(canvas, outline, ui.CYAN, 2)
            ui.put_text(canvas, str(i), outline.max(axis=0) + (4, 0), 0.6, ui.CYAN)
        if self.mapping.ready:
            for i, corner in self.mapping.corners.items():
                if i not in self.markers_seen:   # hidden marker: where the saved mapping has its inner corner
                    ui.draw_cross(canvas, L.rgb_points([corner])[0], ui.GREY, 14, 1)

    def _draw_reach(self, canvas):
        """Reach limit of the arm at table height, only the part over the table."""
        L = self.app.layout
        arc = [[MAX_REACH * math.cos(a), MAX_REACH * math.sin(a)] for a in np.linspace(-1.2, 1.2, 49)]
        ui.draw_polygon(canvas, L.rgb_points(self.mapping.to_pixels(arc)), ui.YELLOW, 1, closed=False)
        label = L.rgb_points(self.mapping.to_pixels([[MAX_REACH, 0.0]]))[0]
        ui.put_text(canvas, "robot reach", (label[0] - 40, label[1] - 6), 0.45, ui.YELLOW)

    def _draw_object(self, canvas, track, obj, others):
        L = self.app.layout
        x, y, _, h = track.obj.bbox
        text = f"x {obj['center'][0]:.2f}  y {obj['center'][1]:+.2f} m"
        if obj["hot"]:
            text += "  HOT: " + ("grip" if plan_grasp(obj, others) is not None else "cannot grip")
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
            print(f"WARNING: {thermal_error}\nNo temperatures, so nothing will be picked.")

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


def gripper_pose(xy, height, closing=(0.0, -1.0)):
    """Gripper pointing down at (x, y, TABLE_Z + height), its fingers closing along closing (x y)."""
    # GRIPPER_DOWN closes the fingers along world -y. Turn it about z to close along closing, by the
    # smaller angle (closing and -closing are the same grasp), so the wrist turns at most 90 degrees.
    yaw = math.atan2(closing[0], -closing[1])
    yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ GRIPPER_DOWN
    return RigidTransform(rotation=rotation, translation=np.array([xy[0], xy[1], TABLE_Z + height]),
                          from_frame="franka_tool", to_frame="world")


def gripper_width(fa):
    """Distance between the fingers, or None if the gripper does not publish it.

    fa.get_gripper_width() waits for the gripper joint states without a timeout, so it hangs for good
    when nothing publishes them.
    """
    if getattr(fa, "_old_gripper", False):
        return fa.get_gripper_width()
    try:
        state = rospy.wait_for_message(fa._gripper_joint_states_name, JointState, timeout=GRIPPER_READ_TIMEOUT)
    except rospy.ROSException:
        print(f"No gripper width on {fa._gripper_joint_states_name}: assuming the object is gripped.")
        return None
    return state.position[0] + state.position[1]


def pick_and_place(fa, plan, height, grip=True):
    """Grips the object, sets it down outside the table area and goes back home.

    Returns False if the fingers closed on nothing; the arm then goes straight back home.
    With grip=False, only follows the path at height, with the gripper open.
    """
    closing, opening = plan["closing"], plan["opening"]
    fa.goto_pose(gripper_pose(plan["grasp"], SAFE_HEIGHT, closing), duration=3.0)
    fa.goto_gripper(opening)
    fa.goto_pose(gripper_pose(plan["grasp"], height, closing), duration=1.5)
    if grip:
        fa.goto_gripper(0.0, grasp=True, force=GRIP_FORCE)
        width = gripper_width(fa)
        if width is not None and width < MIN_GRIP_WIDTH:
            fa.goto_gripper(opening)
            fa.goto_pose(gripper_pose(plan["grasp"], SAFE_HEIGHT, closing), duration=1.5)
            fa.reset_joints()
            return False
    fa.goto_pose(gripper_pose(plan["grasp"], SAFE_HEIGHT, closing), duration=1.5)
    fa.goto_pose(gripper_pose(plan["place"], SAFE_HEIGHT, closing), duration=3.0)
    fa.goto_pose(gripper_pose(plan["place"], height, closing), duration=1.5)
    fa.goto_gripper(opening)
    fa.goto_pose(gripper_pose(plan["place"], SAFE_HEIGHT, closing), duration=1.5)
    fa.reset_joints()
    return True


def robot_main(args):
    fa = None
    if not args.dry_run:
        if FrankaArm is None:
            raise SystemExit("frankapy not found: run the robot side in the frankapy environment (or use --dry-run).")
        fa = FrankaArm()
        fa.reset_joints()
        fa.open_gripper()
    done = set()   # camera track ids not to handle again
    attempts = {}  # camera track id -> grasps tried
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
                attempts.clear()
            if problem:
                time.sleep(1.0)
                continue

            hot = [o for o in report["objects"] if o["hot"] and o["id"] not in done]
            if not hot:
                time.sleep(POLL_PERIOD)
                continue
            target = max(hot, key=lambda o: o["temp"])
            plan = plan_grasp(target, [o for o in report["objects"] if o is not target])
            description = "object #{} ({:.1f} C) at x={:.3f} y={:.3f} m".format(target["id"], target["temp"],
                                                                               *target["center"])
            if plan is None:
                print(f"Leaving {description}: out of reach, too wide, or other objects are in the way.")
                done.add(target["id"])
                continue
            grasp, place = plan["grasp"], plan["place"]
            print(f"Picking {description}: {plan['width'] * 100:.1f} cm wide, gripped at "
                  f"({grasp[0]:.3f}, {grasp[1]:.3f}), set down at ({place[0]:.3f}, {place[1]:.3f})")
            if fa is None or args.hover_only:
                done.add(target["id"])   # the object does not move, do not do it again
            if fa is not None:
                if args.hover_only:
                    pick_and_place(fa, plan, HOVER_HEIGHT, grip=False)
                elif not pick_and_place(fa, plan, GRASP_HEIGHT):
                    attempts[target["id"]] = attempts.get(target["id"], 0) + 1
                    if attempts[target["id"]] >= MAX_ATTEMPTS:
                        print(f"Missed it {MAX_ATTEMPTS} times, leaving it.")
                        done.add(target["id"])
                    else:
                        print("Missed it, trying again.")
                time.sleep(SETTLE_TIME)
    except KeyboardInterrupt:
        print("Stopped.")
        if fa is not None:
            fa.stop_skill()


def main():
    parser = argparse.ArgumentParser(description="Pick the objects hotter than a threshold off the table.")
    sides = parser.add_subparsers(dest="side")
    sides.required = True

    camera = sides.add_parser("camera", help="thermal table app that serves the objects (run with .venv/bin/python)")
    camera.add_argument("--threshold", type=float, default=40.0,
                        help="objects at least this hot (degrees C) are picked (default: 40)")
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

    robot = sides.add_parser("robot", help="picks the hot objects (run in the frankapy environment)")
    robot.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"TCP port (default: {DEFAULT_PORT})")
    check = robot.add_mutually_exclusive_group()
    check.add_argument("--dry-run", action="store_true", help="do not move, only print the grasps")
    check.add_argument("--hover-only", action="store_true",
                       help=f"follow each grasp path {HOVER_HEIGHT * 100:.0f} cm above the table without gripping")

    args = parser.parse_args()
    if args.side == "camera":
        camera_main(args)
    else:
        robot_main(args)


if __name__ == "__main__":
    main()
