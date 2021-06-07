# tamp-perception
The repo contains a mobile manipulator platform in gazebo.

```
mkdir -p ~/test_ws/src
cd ~/test_ws
wstool init src https://raw.githubusercontent.com/keke-220/tamp-perception/master/rosinstall/melodic.rosinstall
rosdep update
rosdep install --from-paths src --ignore-src --rosdistro $ROS_DISTRO -y
catkin build -j2
```
