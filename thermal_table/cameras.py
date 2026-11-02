"""Camera sources.

Every source runs its own acquisition thread and exposes ``read()``, which returns the most recent
frame together with a counter that increases with each new frame (so callers can tell whether the
frame is new).
"""

import sys
import threading
import time

import cv2
import numpy as np

# The Python bindings of the Optris OTC SDK are compiled for one specific Python version and crash
# (segfault) when loaded by another one. The Debian 13 package used on this machine targets 3.13.
SDK_PYTHON_VERSION = (3, 13)


class ThermalUnavailable(RuntimeError):
    """Raised when the thermal camera cannot be used (SDK missing, camera not found, ...)."""


class RgbCamera:
    """Webcam read through V4L2 in a background thread."""

    def __init__(self, device, width=1280, height=720, fps=30):
        self.device = device
        self.requested = (width, height, fps)
        self.size = None  # (width, height) actually delivered
        self._cap = None
        self._lock = threading.Lock()
        self._frame = None
        self._count = 0
        self._running = False
        self._thread = None

    def start(self):
        device = int(self.device) if str(self.device).isdigit() else self.device
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam {self.device}")
        width, height, fps = self.requested
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Webcam {self.device} opened but returned no frame")
        self._cap = cap
        self.size = (frame.shape[1], frame.shape[0])
        self._frame, self._count = frame, 1
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="rgb-camera", daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._count += 1

    def read(self):
        with self._lock:
            return self._frame, self._count

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()


def import_otcsdk():
    """Imports the Optris OTC SDK Python module, with a helpful error if it is not usable."""
    if sys.version_info[:2] != SDK_PYTHON_VERSION:
        wanted = ".".join(map(str, SDK_PYTHON_VERSION))
        raise ThermalUnavailable(
            f"The Optris SDK bindings need Python {wanted}, this is Python "
            f"{sys.version_info.major}.{sys.version_info.minor}. Run the app with .venv/bin/python (see README)."
        )
    try:
        import optris.otcsdk as otc
    except ImportError as ex:
        raise ThermalUnavailable(
            f"Optris OTC SDK not found ({ex}). Install the otcsdk package and run ./setup.sh (see README)."
        ) from ex
    return otc


class ThermalCamera:
    """Optris PI/Xi camera through the OTC SDK. Frames are temperatures in degrees Celsius (float32, HxW)."""

    def __init__(self, serial=0, emissivity=None):
        self.serial = serial
        self.emissivity = emissivity
        self.size = None
        self.description = ""
        self.status = "not started"
        self._otc = None
        self._imager = None
        self._client = None
        self._lock = threading.Lock()
        self._temps = None
        self._count = 0
        self._flag_state = "unknown"
        self._valid = False
        self._dtype = np.float32

    def start(self):
        otc = self._otc = import_otcsdk()
        otc.Sdk.init(otc.Verbosity_Warning, otc.Verbosity_Off, "thermal_table")

        camera = self

        class Client(otc.IRImagerClient):
            # Called from the SDK thread. The event object is reused by the SDK, so copy the data out.
            def onFrame(self, evt):
                camera._on_frame(evt)

            def onConnection(self, evt):
                camera._on_connection(evt)

        self._client = Client()
        self._imager = otc.IRImagerFactory.getInstance().create("native")
        self._imager.addClient(self._client)
        try:
            self._imager.connect(self.serial)
        except otc.SDKException as ex:
            self._imager.removeClient(self._client)
            raise ThermalUnavailable(f"Cannot connect to thermal camera {self.serial or ''}: {ex}") from ex

        if self.emissivity is not None:
            radiation = self._imager.getRadiationParameters()
            radiation.emissivity = float(self.emissivity)
            self._imager.setRadiationParameters(radiation)

        mode = self._imager.getActiveOperationMode()
        self.size = (mode.getFrameWidth(), mode.getFrameHeight())
        self.description = (
            f"{otc.deviceTypeToString(self._imager.getDeviceType())} S/N {self._imager.getSerialNumber()}, "
            f"FOV {mode.getFieldOfView()}deg, {mode.getTemperatureLowerLimit():g}..{mode.getTemperatureUpperLimit():g} C, "
            f"{self.size[0]}x{self.size[1]} @ {mode.getFramerate()} Hz"
        )
        self.status = "connected"
        self._imager.runAsync()
        return self

    def _on_frame(self, evt):
        frame = evt.thermalFrame
        width, height = frame.getWidth(), frame.getHeight()
        if width == 0 or height == 0:
            return
        buffer = np.empty(width * height, dtype=self._dtype)
        try:
            frame.copyTemperaturesTo(buffer)
        except TypeError:
            # Bindings built with a different floating point type than expected.
            self._dtype = np.float64 if self._dtype == np.float32 else np.float32
            return
        flag = evt.meta.getFlagState()
        with self._lock:
            self._temps = buffer.reshape(height, width).astype(np.float32, copy=False)
            self._count += 1
            self._flag_state = self._otc.flagStateToString(flag)
            # Temperatures are not valid until the camera finished its startup flag (shutter) cycle.
            self._valid = flag != self._otc.FlagState_Initializing

    def _on_connection(self, evt):
        self.status = self._otc.connectionStateToString(evt.state)

    def read(self):
        """Returns (temperatures or None, frame counter, temperatures are valid)."""
        with self._lock:
            return self._temps, self._count, self._valid

    @property
    def flag_state(self):
        return self._flag_state

    def force_flag(self):
        """Triggers a shutter (non-uniformity correction) cycle."""
        if self._imager is not None:
            try:
                self._imager.forceFlagEvent()
            except self._otc.SDKException as ex:
                print(f"Flag refresh failed: {ex}")

    def stop(self):
        if self._imager is None:
            return
        self._imager.stopRunning()
        self._imager.disconnect()
        self._imager.removeClient(self._client)
        self._imager = None


class SimulatedThermalCamera:
    """Fake thermal camera for trying the app without the SDK.

    It looks at the webcam image through a fixed homography and turns darkness into heat: the white
    table is ~22 C and dark objects look warm. Use it to test calibration and measurement end to end.
    """

    def __init__(self, rgb_camera, size=(382, 288)):
        self.rgb = rgb_camera
        self.size = size
        self.description = f"SIMULATED {size[0]}x{size[1]}"
        self.status = "simulated"
        self.flag_state = "simulated"
        self.homography = None  # true RGB -> thermal mapping, set in start()
        self._rng = np.random.default_rng(0)
        self._last_count = -1
        self._temps = None
        self._count = 0

    def start(self):
        rgb_w, rgb_h = self.rgb.size
        self.homography = simulated_homography((rgb_w, rgb_h), self.size)
        return self

    def read(self):
        frame, count = self.rgb.read()
        if frame is not None and count != self._last_count:
            self._last_count = count
            self._temps = self.render(frame)
            self._count += 1
        return self._temps, self._count, self._temps is not None

    def render(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        warped = cv2.warpPerspective(gray, self.homography, self.size, flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
        darkness = 1.0 - warped.astype(np.float32) / 255.0
        temps = 22.0 + 25.0 * darkness ** 2
        temps = cv2.GaussianBlur(temps, (5, 5), 1.2)
        return temps + self._rng.normal(0, 0.08, temps.shape).astype(np.float32)

    def force_flag(self):
        pass

    def stop(self):
        pass


def simulated_homography(rgb_size, thermal_size):
    """A plausible RGB -> thermal mapping: the thermal camera sees the central part of the RGB view,
    slightly rotated and with a bit of perspective."""
    rgb_w, rgb_h = rgb_size
    th_w, th_h = thermal_size
    src = (np.float32([[0.22, 0.12], [0.80, 0.16], [0.78, 0.90], [0.20, 0.86]]) * [rgb_w, rgb_h]).astype(np.float32)
    dst = np.float32([[0, 0], [th_w - 1, 0], [th_w - 1, th_h - 1], [0, th_h - 1]])
    return cv2.getPerspectiveTransform(src, dst)
