# tamp-perception
The repo builds on BWI project of UTAustin and contains a mobile manipulator platform in gazebo.

## Requirments
Make sure you have installed ROS Melodic and ROS_DISTRO environment variable is set correctly.

## Installation

Create workspace and download repo:
```
mkdir -p ~/test_ws/src
cd ~/test_ws
wstool init src https://raw.githubusercontent.com/keke-220/segbot-ur5/master/rosinstall/melodic.rosinstall
```

Install dependencies:
```
rosdep update
rosdep install --from-paths src --ignore-src --rosdistro $ROS_DISTRO -y
```

Build everything:
```
catkin build -j2
```

## Usage
Launch banquet environment and bring up mobile manipulator:
```
roslaunch tamp_perception segbot_ur5.launch
```

Test with a simple pick&place task:
```
rosrun tamp_perception pick_n_place.py
```
