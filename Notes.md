## Tips

When the robot encounters an error or you push the emergency stops button (the led of the robot goes white), you need to ctrl+c the 3 terminal commands (franka-interface, franka_ros_interface, franka gripper), and restart them in the right order.

If you want to see the actual postion of the robot head/gripper, kill the terminal commands and launch `roslaunch franka_visualisation franka_visualisation.launch`. In the RVIZ window, unroll the RobotModel tree and look at the coordinates of panda_hand_tcp (the middle of the gripper fingers)

You should also be able to launh rviz while running this repo's code by opening rviz and opening the rviz config without using the franka_visualisation.launch file.

## Modifications made:

If you have virtual walls errors (either the program stops on an error or a movement is canceled), you can try to change the virtual walls positions by editing the hyperplanes in `~/franka-interface/include/franka-interface/termination_handler/termination_handler.h`

then build the interface wih 
```
cd ~/franka-interface/build
make -j4
```

## Logs from last runs 

Some logs obtained during the execution of the code in this branch.

In push_hot_objects.py robot:
```
Leaving object #85 (27.6 C) at x=0.564 y=-0.018 m: out of reach, too wide, or other objects are in the way.
Leaving object #89 (27.5 C) at x=0.563 y=-0.011 m: out of reach, too wide, or other objects are in the way.
Picking object #99 (25.2 C) at x=0.724 y=0.002 m: 4.5 cm wide, gripped at (0.724, 0.001), set down at (0.724, 0.194)
No gripper width on /franka_gripper_1/joint_states: assuming the object is gripped.
Leaving object #118 (24.7 C) at x=0.800 y=-0.029 m: out of reach, too wide, or other objects are in the way.
Picking object #127 (24.7 C) at x=0.631 y=-0.015 m: 4.7 cm wide, gripped at (0.631, -0.015), set down at (0.631, 0.203)
[ERROR] [1792674969.505010]: Got a transition callback on a goal handle that we're not tracking
No gripper width on /franka_gripper_1/joint_states: assuming the object is gripped.
[ERROR] [1792674974.621801]: Got a transition callback on a goal handle that we're not tracking
```

The error should be addressed


