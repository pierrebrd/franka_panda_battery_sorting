"""Entry point: python -m thermal_table"""

import argparse
import glob
import os
from pathlib import Path

from .app import App, run
from .cameras import RgbCamera, SimulatedThermalCamera, ThermalCamera, ThermalUnavailable

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RGB_DEVICE = "/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_B68355BE-video-index0"
DEFAULT_THERMAL_SERIAL = 16070070


def configure_qt():
    """opencv-python bundles a Qt with only the X11 plugin and no fonts (slider labels would be blank)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    for fonts in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation", "/usr/share/fonts"):
        if os.path.isdir(fonts):
            os.environ.setdefault("QT_QPA_FONTDIR", fonts)
            break


def main():
    configure_qt()
    parser = argparse.ArgumentParser(
        prog="python -m thermal_table",
        description="Detect objects on a table with a webcam and show their temperature from an Optris thermal camera.")
    parser.add_argument("--rgb-device", default=DEFAULT_RGB_DEVICE,
                        help="webcam device path or index (default: the Logitech C930e)")
    parser.add_argument("--rgb-size", default="1280x720", help="webcam resolution, WIDTHxHEIGHT")
    parser.add_argument("--thermal-serial", type=int, default=DEFAULT_THERMAL_SERIAL,
                        help="serial number of the Optris camera, 0 = first one found")
    parser.add_argument("--emissivity", type=float, default=None,
                        help="emissivity used by the thermal camera (default: the camera's setting)")
    parser.add_argument("--simulate-thermal", action="store_true",
                        help="fake the thermal camera from the webcam image (to try the app without the SDK)")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data",
                        help="where calibration, table area, background and snapshots are stored")
    parser.add_argument("--display-height", type=int, default=540, help="height of the camera views on screen")
    args = parser.parse_args()

    width, height = (int(v) for v in args.rgb_size.lower().split("x"))
    rgb = RgbCamera(args.rgb_device, width, height)
    try:
        rgb.start()
    except RuntimeError as ex:
        devices = "\n  ".join(sorted(glob.glob("/dev/v4l/by-id/*-video-index0"))) or "(none)"
        raise SystemExit(f"{ex}\nAvailable cameras:\n  {devices}\nChoose one with --rgb-device.")
    print(f"Webcam: {args.rgb_device} {rgb.size[0]}x{rgb.size[1]}")

    thermal, thermal_error, data_dir = None, None, args.data_dir
    if args.simulate_thermal:
        thermal = SimulatedThermalCamera(rgb).start()
        data_dir = data_dir / "simulated"   # never mix a simulated calibration with the real one
    else:
        try:
            thermal = ThermalCamera(args.thermal_serial, args.emissivity).start()
            print(f"Thermal camera: {thermal.description}")
        except ThermalUnavailable as ex:
            thermal_error = str(ex)
            print(f"WARNING: {thermal_error}\nContinuing without temperatures.")

    app = App(rgb, thermal, data_dir, thermal_error, args.display_height)
    try:
        run(app)
    except KeyboardInterrupt:
        pass
    finally:
        if thermal is not None:
            thermal.stop()
        rgb.stop()


if __name__ == "__main__":
    main()
