#!/usr/bin/env python3

import subprocess

import time
import numpy as np

from frankapy import FrankaArm
from autolab_core import RigidTransform

def play_sound(sound_filename):
    subprocess.run(["aplay", "-D", "hw:1,0", sound_filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)



def wave_hand(fa):
    # Run the hello sound
    subprocess.run(["aplay", "-D", "hw:1,0", "audios/welcome_audio.wav"])

    # Make the joint more upright
    joints = fa.get_joints()
    joints[1] -= np.deg2rad(30)
    joints[3] += np.deg2rad(10)
    joints[5] += np.deg2rad(40)
    fa.goto_joints(joints)

    # Start with a small 2 cm displacement.
    duration = 1.0

    try:
        # Get the current joint positions
        start_joints = fa.get_joints()

        start_joint2 = start_joints[2]

        amplitude = np.deg2rad(15)  # Wave amplitude: 15 degrees

        target_joints = start_joints.copy()
        target_joints[2] += amplitude
        fa.goto_joints(target_joints, duration=duration/2)
        target_joints[2] -= 2*amplitude
        fa.goto_joints(target_joints, duration=duration)
        target_joints[2] += 2*amplitude
        fa.goto_joints(target_joints, duration=duration)
        target_joints[2] -= 2*amplitude
        fa.goto_joints(target_joints, duration=duration)
        #target_joints[2] += 2*amplitude
        #fa.goto_joints(target_joints, duration=duration)
        #target_joints[2] -= 2*amplitude
        #fa.goto_joints(target_joints, duration=duration)

        print("Wave complete.")

    except KeyboardInterrupt:
        print("Interrupted. Stop motion safely using the robot's "
              "approved stop procedure.")

def point_place(fa):
    subprocess.run(["aplay", "-D", "hw:1,0", "audios/instruction_audio.wav"])
    joints = fa.get_joints()
    joints[1] += np.deg2rad(80)
    joints[2] += np.deg2rad(15/2)
    joints[5] += np.deg2rad(20)
    fa.goto_joints(joints)

def sweep(fa, to_right):
    # Desired position in meters (x, y, z)
    x = 0.64
    y = -0.2 if to_right else 0.2
    z = 0.05
    initial_position = np.array([x, y, z])
    final_position = np.array([x, -y, z])

    # Rotation: end effector pointing straight down
    # Assumes the tool's local Z-axis should point downward
    target_rotation = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])

    initial_pose = RigidTransform(
        rotation=target_rotation,
        translation=initial_position,
        from_frame="franka_tool",
        to_frame="world"
    )

    final_pose = RigidTransform(
        rotation=target_rotation,
        translation=final_position,
        from_frame="franka_tool",
        to_frame="world"
    )

    fa.goto_pose(initial_pose, duration=4.0)
    fa.goto_pose(final_pose, duration=4.0)

def high_five(fa):
    # Desired position in meters (x, y, z)
    x = 0.8
    y = 0
    z = 0.4
    target_position = np.array([x, y, z])

    # Rotation: end effector pointing straight down
    # Assumes the tool's local Z-axis should point downward
    target_rotation = np.array([
        [ 0,  0,  1],
        [ 0,  1,  0],
        [-1,  0,  0]
    ])

    target_pose = RigidTransform(
        rotation=target_rotation,
        translation=target_position,
        from_frame="franka_tool",
        to_frame="world"
    )

    subprocess.run(["aplay", "-D", "hw:1,0", "audios/thanking_audio.wav"])
    fa.goto_pose(target_pose, duration=4.0)
    time.sleep(1)                               # Here, see whether the user gave high five 
    fa.reset_joints()


if __name__ == "__main__":
    # Connect to the Franka robot
    fa = FrankaArm()

    # reset franka to its home joints
    fa.reset_joints()

    wave_hand(fa)

    point_place(fa)

    sweep(fa, True)

    high_five(fa)

