# Thermal table

A webcam and a thermal camera look down at a white table. When someone puts an object on the table,
the webcam detects it and the app shows its temperature, measured by the thermal camera.

The window shows the webcam view on the left, with each detected object outlined and labelled with
its temperature (median over the object, plus max and min). The thermal view on the right shows the
same objects. A cyan outline on the webcam view marks the part of the table the thermal camera sees.

## Hardware

| Camera | Model | How it is accessed |
|---|---|---|
| Regular | Logitech C930e | V4L2 through OpenCV (`/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_...`) |
| Thermal | Optris PI 450 (382x288), S/N 16070070 | Optris OTC SDK |

The thermal camera's raw USB video stream holds no usable image: temperatures need the vendor SDK,
which configures the sensor and applies the factory calibration. The XML files in `ref/` are configs for
Optris' old *libirimager* SDK, which is no longer available (evocortex.com is gone). The current SDK
(OTC SDK) downloads the camera calibration files by itself the first time it connects, so it needs
internet access once.

## Installation

1. **Optris OTC SDK** (needs sudo once). It installs the library, the Python bindings and a udev rule
   that gives your user access to the camera:

   ```bash
   sudo apt install ~/Downloads/otcsdk-11.4.10-debian-13-amd64-ubuntu26.04.deb
   ```

   Unplug and replug the thermal camera after installing, so the new permissions apply.

   Optris publishes no Ubuntu 26.04 package. The Debian 13 build from
   <https://github.com/Optris/otcsdk_downloads/releases> (v11.4.10) is the right base, because its
   Python bindings target Python 3.13, which Ubuntu 26.04 also provides. But it does not install as is:
   - it requires `libglew2.2 (>= 2.2.0-4+b2)`, and Ubuntu's `2.2.0-4build2` sorts lower, although it is
     the same GLEW 2.2.0. GLEW is only used by the vendor's C++ demo program, not by the library or the
     Python bindings;
   - its post-install script (udev reload, `ldconfig`) is not executable, so `dpkg-deb` rejects it.

   The `-ubuntu26.04` package is the Debian 13 one with only these two things fixed. The installed files
   are unchanged. To rebuild it from the original:

   ```bash
   dpkg-deb -R otcsdk-11.4.10-debian-13-amd64.deb pkg
   sed -i -e 's/libglew2\.2 (>= 2\.2\.0-4+b2)/libglew2.2 (>= 2.2.0)/' \
          -e 's/^Version: .*/&+ubuntu26.04/' pkg/DEBIAN/control
   chmod 0755 pkg/DEBIAN/postinst
   dpkg-deb --root-owner-group -b pkg otcsdk-11.4.10-debian-13-amd64-ubuntu26.04.deb
   ```

2. **Python environment**:

   ```bash
   ./setup.sh
   ```

   This creates `.venv` with **Python 3.13** (the SDK bindings crash with the system's Python 3.14),
   installs numpy and OpenCV, and links the SDK bindings into the venv.

## Running

```bash
.venv/bin/python -m thermal_table
```

Useful options (see `--help`):

- `--simulate-thermal`: fakes the thermal camera from the webcam image (dark = hot). Lets you try the
  whole workflow without the SDK. It stores its settings in `data/simulated/`, apart from the real ones.
- `--emissivity 0.95`: emissivity used for the temperature computation (see Limitations).
- `--rgb-device /dev/videoN`, `--thermal-serial N`: use other cameras.

## First-time setup

The status bar at the bottom says what to do next. Each step is saved in `data/` and reloaded at the
next start.

1. **Table area (`r`)**: click the corners of the usable table surface, then press ENTER. Detection is
   restricted to this polygon, so people moving next to the table are ignored.
2. **Empty-table background (`b`)**: clear the table and press `b`. The app averages a few frames of
   the empty table (webcam and thermal) and detects objects as changes from it. Do this again if the
   lighting changes a lot.
3. **Calibration (`c`)**: teaches the app which thermal pixel corresponds to which webcam pixel.
   - *Automatic*: put **one** object that is clearly warmer or colder than the table (a mug of hot
     water, a cold drink can, a hand warmer) on the table. Take your hand away and press SPACE. The
     object's position is measured in both cameras and recorded as a pair. Repeat at 6 to 10 places
     spread over the whole table, including near the corners of the thermal view. Flat objects work
     best (see parallax below).
   - *Manual*: click a point in the webcam view, then the same point in the thermal view. Right click
     undoes the last pair.
   - After 4 pairs, the error (in thermal pixels) is displayed, and each thermal point gets a line to
     where the fit predicts it. Pairs in red were rejected as outliers. Press `o` to blend the thermal
     image over the webcam view and check that edges line up, then ENTER to save.

   Calibrate again whenever one of the cameras moves.

## Keys

| Key | Action |
|---|---|
| `r` | define the table area |
| `b` | capture the empty-table background (webcam and thermal) |
| `c` | calibrate webcam against thermal camera |
| `o` | toggle the thermal overlay on the webcam view |
| `d` | detection debug window, with sliders for the detection thresholds |
| `f` | force a flag (shutter) cycle of the thermal camera |
| `s` | snapshot: screen image, webcam image, thermal data (`.npy`, degrees C) and a JSON report, saved in `data/snapshots/` |
| `h` | hide or show the help panel (calibration and table-area modes) |
| `q` / ESC | quit (ESC leaves calibration or table-area mode) |

Each object's temperature is also printed in the terminal once it has been stable for half a second.

## How it works

- **Detection** (`thermal_table/detection.py`): the live image is compared with the empty-table image
  in Lab color space. A pixel belongs to an object if its color changed, or if it got much darker or
  brighter. Soft shadows only darken the table a little without changing its color, so they are not
  counted. Brightness and white balance drifts from the webcam's automatic settings are compensated
  with per-channel gains measured on the free table area. Blobs touching the edge of the table area are
  marked as *edge* (a hand or arm reaching in) and drawn in grey.
- **Calibration** (`thermal_table/calibration.py`): since objects lie on a flat table, one homography
  maps webcam pixels to thermal pixels for everything on the table plane. It is fitted with RANSAC on
  the point pairs. In automatic mode, the thermal position is the region that changed most compared
  with the empty-table thermal image, so static warm things in view are ignored.
- **Measurement** (`thermal_table/measure.py`): each object's outline is mapped into the thermal image
  and filled. The edge pixels, which mix object and table, are dropped. The median, max and min of the
  remaining pixels are reported and smoothed over time. The table temperature is the median of the
  table pixels without objects.

## Limitations

- **Parallax**: the homography is exact only for points *on* the table. The top of a tall object is
  shifted by roughly `baseline x height / (camera height - height)`, where the baseline is the distance
  between the two cameras. For example, cameras 5 cm apart, 1 m above the table, and a 10 cm tall object
  give about 6 mm. Keep the cameras close together. The median and the edge erosion absorb small shifts.
- **Emissivity**: a thermal camera computes temperature assuming an emissivity. Most matte objects,
  skin, paper, plastic, painted surfaces and water are around 0.95. Shiny metal is much lower and reads
  far too cold (it reflects the room). Set `--emissivity` for your objects, or put matte tape on metal.
- **White objects on the white table** are hard to see for the webcam. Colored, dark or textured
  objects work best. Use the `d` debug window to tune the thresholds if needed.
- The thermal camera sees less of the table than the webcam (cyan outline). Objects outside it are
  detected, but show no temperature.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

The tests need no hardware. They cover detection on synthetic tables, the homography fit and outlier
rejection, the measurement, and the full workflow (background, automatic calibration, measurement)
with the simulated thermal camera, whose true homography is known.

## Files

```
thermal_table/
  __main__.py      command line, camera startup
  app.py           modes, keyboard/mouse, processing loop
  cameras.py       webcam thread, Optris SDK wrapper, simulated thermal camera
  detection.py     empty-table background model and object detection
  calibration.py   homography fit, warm/cold spot finder
  measure.py       per-object temperature, table temperature, tracking
  ui.py            drawing helpers
data/              saved table area, background, calibration and snapshots (created at runtime)
ref/               configs for the old libirimager SDK (reference only, not used)
```
