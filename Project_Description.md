# Final Project: battery sorting assited by a Robot Arm

*Team: Pierre Bordeau, Nathan Chandanson, Gabriel Colin de Verdière, Hugo Lequy, Rémi Moshfeghi*

## Context and Idea

Our project takes place in the context of a battery recycling and sorting site. A user brings one or multiple batteries, and tries to recharge them on a self-test desk, to see if they are still usable and get metrics on their health.

Unfortunately, old or damaged batteries can be dangerous: they can leak, heat up, catch fire or explode. To avoid this, it's important to detect when a battery starts to dysfunction in the sorting site, to then isolate the battery and protect the users.

To do so, we make use of a thermal camera to detect the temperature of the batteries. In a real situation, we could also use other sensors (for instance sensors that detect the voltage of the batteries, the resistance, or the current they can provide). When a hot battery is detected, the users present around the desk are warned that a battery is overheating, and a robot arm is used to pick up the battery and drop it in a safe place (a fireproof box for example).

Apart from the safety aspect, the goal of the robot is to interact with the user, making them feel safe and comfortable, while encouraging the use of the self-testing desk and the recycling of batteries, for a greener future.

## Implementation

For our project, we worked with the Franka Emika Panda robot arm, a RGB camera, and a thermal camera.

A computer is connected to the cameras, and using machine vision, it detects the objets on the desk and their temperature, and sends the information of the detected objects to another computer (or the same computer but running a different instance of the program), connected to the Franka robot arm and using `frankapy` (a Python library to control the Franka robot without directly using ROS) to control the robot to pick up and drop the batteries.

The computer controlling the robot also manages the interaction with the user: it firstly greets the user, then explains the functioning of the self-testing desk, and finally warns the user when a battery is overheating and is being picked up by the robot. 