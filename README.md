# tamp-perception
The repo builds on BWI project of UTAustin and contains a mobile manipulator platform in gazebo.

## Requirments
Make sure you have installed ROS Melodic and ROS_DISTRO environment variable is set correctly.

## Installation

Create a workspace and download the repo:
```
mkdir -p ~/test_ws/src
cd ~/test_ws
wstool init src https://raw.githubusercontent.com/keke-220/tamp-perception/master/rosinstall/melodic.rosinstall
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
