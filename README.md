# Final Project: battery sorting assisted by a Robot Arm

*Group 2: Pierre Bordeau, Nathan Chandanson, Gabriel Colin de Verdière, Hugo Lequy, Rémi Moshfeghi*

**See Project_Description.md for a presentation of the project idea**

- [Final Project: battery sorting assisted by a Robot Arm](#final-project-battery-sorting-assisted-by-a-robot-arm)
  - [Hardware used](#hardware-used)
  - [Installation](#installation)
    - [Robot side](#robot-side)
    - [Cameras side](#cameras-side)
  - [Launching the demo](#launching-the-demo)
    - [Cameras side](#cameras-side-1)
      - [Calibration](#calibration)
      - [Keys](#keys)
    - [Robot side](#robot-side-1)
  - [Interacting with the robot](#interacting-with-the-robot)
  - [Other notes](#other-notes)


## Hardware used

- Franka Emika Panda robot arm, connected via Ethernet
- RGB camera (Logitech C930e) connected via USB
- Thermal camera (Optris PI 450) connected via USB

All the code was tested on Ubuntu 26.04, with the system Docker (not Docker Desktop)

## Installation

The project runs on 2 parts: the "robot" part, which controls the robot arm and manages the interaction with the user, and the "table" part, which detects objects on the table and their temperature.

The "table" part runs directly on the computer, using a python venv, and the "robot" part runs on a Docker container

### Robot side

You need to use the system Docker socket, not the Docker Desktop one, to avoid limitations of the VM Docker Desktop uses.

- Download the Docker image: `docker pull jgarcia10/u2is-franka-ros1:latest`
- Create a container running that image. The container needs different devices and ports to be able to play sound, create windows, access the network adapter, and it also needs settings to run in real time, which is needed for the robot to work properly. It also needs to mount the workspace. The command to run the container is (run it in a terminal in the root of the repo):

```bash
docker run -d --name franka_container \
  --net=host --ipc=host \
  --ulimit rtprio=99 --ulimit memlock=-1 --cap-add=sys_nice \
  --security-opt seccomp=unconfined \
  --device /dev/snd \
  --device /dev/dri:/dev/dri \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$(pwd):/workspace" \
  jgarcia10/u2is-franka-ros1:latest sleep infinity
```

*Note : if you want to be able to run GUI apps such as rviz or rqt_graph, run `xhost +local:root` on the host machine*

- After that, create a terminal in the container with `docker exec -it franka_container bash`

Now, we have to modify the franka_interface code to modify the virtual walls of the robot, to allow the robot to reach the whole table, and the side of the table to drop the objects. 

- Open the file `/root/franka-interface/franka-interface/include/franka-interface/termination_handler/termination_handler.h`
- Go to the end of the file and replace the virtual walls with these ones:

```cpp
// Create hyperplanes
  const std::vector<Eigen::Hyperplane<double,3>> planes_ {
    // Front
    Eigen::Hyperplane<double, 3>(Eigen::Vector3d(1., 0., 0.), Eigen::Vector3d(0.85, 0., 0.)),
    // Sides 
    Eigen::Hyperplane<double, 3>(Eigen::Vector3d(0., 1., 0.), Eigen::Vector3d(0., 0.67, 0.)),
    Eigen::Hyperplane<double, 3>(Eigen::Vector3d(0., 1., 0.), Eigen::Vector3d(0., -0.47, 0.)),
    // Bottom
    Eigen::Hyperplane<double, 3>(Eigen::Vector3d(0., 0., 1.), Eigen::Vector3d(0., 0., -0.015)),
    // Top
    Eigen::Hyperplane<double, 3>(Eigen::Vector3d(0., 0., 1.), Eigen::Vector3d(0., 0., 1.25)),
    // Back
    Eigen::Hyperplane<double, 3>(Eigen::Vector3d(1., 0., 0.), Eigen::Vector3d(-0.46, 0., 0.))
  };
```

- Now go to `~/franka-interface/build` and run `make -j4` to rebuild the franka_interface
- Finally, run `sudo apt install alsa-utils` to be able to play audio in the container

### Cameras side

- Install the Optris OTC SDK with `sudo apt install ./otcsdk-11.4.10-debian-13-amd64-ubuntu26.04.deb`. It is needed to access the thermal camera and get the temperature of the objects. You can find the deb file in the root of the repo, but it would be better to download the latest version for your platform online. To adapt the package to ubuntu 26, we had to run this to create the package in this repo:
```bash
dpkg-deb -R otcsdk-11.4.10-debian-13-amd64.deb pkg
sed -i -e 's/libglew2\.2 (>= 2\.2\.0-4+b2)/libglew2.2 (>= 2.2.0)/' \
      -e 's/^Version: .*/&+ubuntu26.04/' pkg/DEBIAN/control
chmod 0755 pkg/DEBIAN/postinst
dpkg-deb --root-owner-group -b pkg otcsdk-11.4.10-debian-13-amd64-ubuntu26.04.deb
```

After installing the SDK, unplug and plug back the thermal camera

Then run `./setup.sh` to create the python venv and link the SDK bindings to it. You can also do it manually with the following steps:
- Create a virtual environment with python 3.13 with uv for example and activate it
- Install the dependencies with `pip install -r requirements.txt`
- Bind the SDK python bindings to the venv with
```bash
SDK=/usr/lib/python3/dist-packages/optris
SITE=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
if [ -d "$SDK" ]; then
    ln -sfn "$SDK" "$SITE/optris"
    .venv/bin/python -c 'import optris.otcsdk' && echo "Optris SDK: OK"
else
    echo "Optris SDK not installed yet. Install it (see README), then run ./setup.sh again:"
    echo "  sudo apt install ~/Downloads/otcsdk-11.4.10-debian-13-amd64-ubuntu26.04.deb"
fi
```

## Launching the demo

First launch the cameras side, and then the robot side, following these steps:

### Cameras side

Activate the venv, and then run `python3 whole_cycle.py camera --threshold 25` to launch the camera side. Replace 25 with the temperature threshold you want to use to detect hot objects.

A window will open, showing the webcam view on the left and the thermal view on the right. The detected objects will be outlined and labelled with their temperature.

The thermal camera's raw USB video stream holds no usable image: temperatures need the vendor SDK, which configures the sensor and applies the factory calibration. The current SDK downloads the camera calibration files by itself the first time it connects, so it needs internet access once.

#### Calibration

The RGB camera should be able to see the 4 calibration patterns on the corners of the table, and the thermal camera should have a POV similar to the RGB camera. 

You can then make the calibration:

1. **Table area (`r`)**: click the corners of the usable table surface, then press ENTER. Detection is restricted to this polygon, so people moving next to the table are ignored. This should already be done in this repo so no need to do it again.
2. **Empty-table background (`b`)**: clear the table and press `b`. The app averages a few frames of the empty table (webcam and thermal) and detects objects as changes from it. Do this again if the lighting changes a lot.
3. **Calibration (`c`)**: teaches the app which thermal pixel corresponds to which webcam pixel.
   - *Automatic*: put **one** object that is clearly warmer or colder than the table (a mug of hot water, a cold drink can, a hand warmer) on the table. Take your hand away and press SPACE. The object's position is measured in both cameras and recorded as a pair. Repeat at 6 to 10 places spread over the whole table, including near the corners of the thermal view. Flat objects work best (see parallax below).
   - *Manual*: click a point in the webcam view, then the same point in the thermal view. Right click undoes the last pair.
   - After 4 pairs, the error (in thermal pixels) is displayed, and each thermal point gets a line to where the fit predicts it. Pairs in red were rejected as outliers. Press `o` to blend the thermal image over the webcam view and check that edges line up, then ENTER to save.

Calibrate again whenever one of the cameras moves or when the background changes and you have false detections. 

#### Keys

Here is a summary of the keys you can use in the camera side:

| Key       | Action                                                                                                               |
| --------- | -------------------------------------------------------------------------------------------------------------------- |
| `r`       | define the table area                                                                                                |
| `b`       | capture the empty-table background (webcam and thermal)                                                              |
| `c`       | calibrate webcam against thermal camera                                                                              |
| `o`       | toggle the thermal overlay on the webcam view                                                                        |
| `d`       | detection debug window, with sliders for the detection thresholds                                                    |
| `f`       | force a flag (shutter) cycle of the thermal camera                                                                   |
| `s`       | snapshot: screen image, webcam image, thermal data (`.npy`, degrees C) and a JSON report, saved in `data/snapshots/` |
| `h`       | hide or show the help panel (calibration and table-area modes)                                                       |
| `q` / ESC | quit (ESC leaves calibration or table-area mode)                                                                     |


### Robot side

- Connect the robot via Ethernet and power it on. 
- Configure the Ethernet connection as explained in the PDF provided by Caio C. 
- Connect to the Franka Desk in a browser using the IP address of the robot
- Unlock the joints and activate FCI
- Launch 5 terminals within the Docker container (use `docker exec -it franka_container bash` in 5 terminals)
- In the terminals, run the following commands (each in 1 terminal):
  - `roscore`
  - `~/franka-interface/build/franka_interface --robot_ip 147.250.35.11`
  - `roslaunch franka_ros_interface franka_ros_interface.launch robot_ip:=147.250.35.11`
  - `roslaunch franka_gripper franka_gripper.launch robot_ip:=147.250.35.11 __ns:=/franka_gripper_1`
- In the last terminal, go to the `/workspace` folder and run `python3 whole_cycle.py robot --drop` to launch the robot side. The `--drop` argument makes the robot drop the objects after picking them, instead of replacing them on the table.

When the robot encounters an error (or when you take back manual control when pushing the push button), to make it work again you need to Ctrl+C the 4 terminal commands (franka-interface, franka_ros_interface, franka gripper, and the whole_cycle.py script), and restart them in the right order. You can usually keep `roscore` running.

Sometimes, it is also needed to completely reboot the robot.

Also, be careful with the Ethernet cable, the connection to the robot can be lost if the cable is moved too much.

## Interacting with the robot

When everything is launched, the robot will first greet the user and explain the functioning of the self-testing desk. 

Then, you can place objects on the table, both cold and hot objects, and the robot will detect them and announce when a hot object is detected.

If you have your hand in the working area of the robot, the robot will detect it and wait for you to remove your hand before continuing.

When you're done, you can press `q` in the terminal running the robot side; the robot will thank the user and exit the program.

## Other notes

- If the position of the 4 calibration patterns on the robot table has changed, you have to remap the coordinates in the code. To do this, you can use RViz to visualize the robot and its movements, and access the coordinates of the middle of the gripper fingers. Getting the value of the position for the inner corner of each calibration pattern will allow you to remap the coordinates in the code. 
  - To do so, kill the terminal commands (franka-interface, franka_ros_interface, franka gripper) and launch `roslaunch franka_visualisation franka_visualisation.launch`. In the RViz window, unroll the RobotModel tree and look at the coordinates of panda_hand_tcp (the middle of the gripper fingers).
