from frankapy import FrankaArm

fa = FrankaArm()

pose = fa.get_pose()

print("X:", pose.translation[0], "m")
print("Y:", pose.translation[1], "m")
print("Z:", pose.translation[2], "m")

print("Orientation (quaternion):", pose.quaternion)
