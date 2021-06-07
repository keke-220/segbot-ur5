#!/usr/bin/env python
import time
import roslib; roslib.load_manifest('ur_driver')
import rospy
import actionlib
from control_msgs.msg import *
from trajectory_msgs.msg import *

JOINT_NAMES = ['finger_joint']

init_pos = [0, -2.2, 2.2, -1.57, -1.57, 0]

client = None

def move1():
    g = FollowJointTrajectoryGoal()
    g.trajectory = JointTrajectory()
    g.trajectory.joint_names = JOINT_NAMES
    g.trajectory.points = [JointTrajectoryPoint(positions=init_pos, velocities=[0]*6, time_from_start=rospy.Duration(4.0))]
    client.send_goal(g)
    try:
        client.wait_for_result()
    except KeyboardInterrupt:
        client.cancel_goal()
        raise

def open():
    print ('hello')

def close():
    g = GripperCommandGoal()
    g.command.position = 0.33
    g.command.max_effort = -1
    rospy.loginfo("Sending close goal...")
    client.send_goal(g)
    client.wait_for_result()

def open():
    g = GripperCommandGoal()
    g.command.position = 0
    g.command.max_effort = -1
    rospy.loginfo("Sending open goal...")
    client.send_goal(g)
    client.wait_for_result()


def main():
    global client
    try:
        rospy.init_node("test_gripper", anonymous=True, disable_signals=True)
        client = actionlib.SimpleActionClient('gripper_controller/gripper_cmd', GripperCommandAction)
        print "Waiting for server..."
        client.wait_for_server()
        print "Connected to server"
        open()
        #move_repeated()
        #move_disordered()
        #move_interrupt()
    except KeyboardInterrupt:
        rospy.signal_shutdown("KeyboardInterrupt")
        raise

if __name__ == '__main__': main()
