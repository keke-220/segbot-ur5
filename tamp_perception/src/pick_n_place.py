#!/usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import time
import actionlib
from math import pi, acos, tan, cos, sin, atan2
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from control_msgs.msg import *
from trajectory_msgs.msg import *
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Point
## END_SUB_TUTORIAL

robot_pose = None
init_joints = [0, -2.2, 2.2, -1.57, -1.57, 0]

pick_loc = [6.8, 0]
pick_orien = [0.0, 0.0, 0.0, 1.0]


object_loc = [[7.3, 0.2], [7.3, 0], [7.3, -0.2]]
54
place_loc = [[-2.4000260221, 5.27721117742], [-1.7000260221, 5.27721117742], [-1.069132618, 6.52850726385]]
place_orien = [[0, 0, -0.705757194679, -0.708453782101], [0, 0, -0.705757194679, -0.708453782101], [0, 0, 0.761778820339, -0.647837184558]]
place_ee_loc = [[-2.4, 5.8], [-1.7, 6.1], [-1.0, 6.1]]

def all_close(goal, actual, tolerance):
  """
  Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
  @param: goal       A list of floats, a Pose or a PoseStamped
  @param: actual     A list of floats, a Pose or a PoseStamped
  @param: tolerance  A float
  @returns: bool
  """
  all_equal = True
  if type(goal) is list:
    for index in range(len(goal)):
      if abs(actual[index] - goal[index]) > tolerance:
        return False

  elif type(goal) is geometry_msgs.msg.PoseStamped:
    return all_close(goal.pose, actual.pose, tolerance)

  elif type(goal) is geometry_msgs.msg.Pose:
    return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

  return True


def get_pose(msg):
    #print (msg.pose.pose)
    robot_pose = msg.pose.pose


def go_to_init_joints(move_group):


    joint_goal = move_group.get_current_joint_values()
    joint_goal[0] = init_joints[0]
    joint_goal[1] = init_joints[1]
    joint_goal[2] = init_joints[2]
    joint_goal[3] = init_joints[3]
    joint_goal[4] = init_joints[4]
    joint_goal[5] = init_joints[5]

    # The go command can be called with joint values, poses, or without any
    # parameters if you have already set the pose or joint target for the group
    move_group.go(joint_goal, wait=True)

    # Calling ``stop()`` ensures that there is no residual movement
    move_group.stop()

    ## END_SUB_TUTORIAL

    # For testing:
    current_joints = move_group.get_current_joint_values()
    return all_close(joint_goal, current_joints, 0.01)

def go_to_ee_pose(move_group, object_pose):
    robot_pose = rospy.wait_for_message('/odom', Odometry).pose.pose
    
    #robot pose
    rx = robot_pose.position.x
    ry = robot_pose.position.y

    #convert quaternion to yaw value
    w = rospy.wait_for_message('/odom', Odometry).pose.pose.orientation.w
    x = rospy.wait_for_message('/odom', Odometry).pose.pose.orientation.x
    y = rospy.wait_for_message('/odom', Odometry).pose.pose.orientation.y
    z = rospy.wait_for_message('/odom', Odometry).pose.pose.orientation.z

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    theta = atan2(siny_cosp, cosy_cosp)


    #object pose
    dx = object_pose[0]
    dy = object_pose[1]
    pose_goal = geometry_msgs.msg.Pose()
    current_pose = move_group.get_current_pose().pose
    pose_goal.orientation = current_pose.orientation
    pose_goal.position.x = ((dy-ry)*tan(theta) - (rx-dx))/(cos(theta) + sin(theta)*tan(theta))
    pose_goal.position.y = (dy - ry - (dx - rx)*tan(theta))/(sin(theta)*tan(theta) + cos(theta))

    pose_goal.position.z = 1.3

    move_group.set_pose_target(pose_goal)

    ## Now, we call the planner to compute the plan and execute it.

    plan = move_group.go(wait=True)
    # Calling `stop()` ensures that there is no residual movement
    move_group.stop()
    # It is always good to clear your targets after planning with poses.
    # Note: there is no equivalent function for clear_joint_value_targets()
    move_group.clear_pose_targets()

    # For testing:
    # Note that since this section of code will not be included in the tutorials
    # we use the class variable rather than the copied state variable
    current_pose = move_group.get_current_pose().pose
    return all_close(pose_goal, current_pose, 0.01)

def go_to_pick_ee_pose(move_group, object_pose):
    #this function is different from go_to_ee_pose because the end effector has to be vertical to the object in order to grasp
    robot_pose = rospy.wait_for_message('/odom', Odometry).pose.pose

    
    #robot pose
    rx = robot_pose.position.x
    ry = robot_pose.position.y

    #convert quaternion to yaw value
    w = rospy.wait_for_message('/odom', Odometry).pose.pose.orientation.w
    x = rospy.wait_for_message('/odom', Odometry).pose.pose.orientation.x
    y = rospy.wait_for_message('/odom', Odometry).pose.pose.orientation.y
    z = rospy.wait_for_message('/odom', Odometry).pose.pose.orientation.z

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    theta = atan2(siny_cosp, cosy_cosp)


    #object pose
    dx = object_pose[0]
    dy = object_pose[1]
    pose_goal = geometry_msgs.msg.Pose()

    
    #rotate the gripper so that it can grasp
    roll = 0
    pitch = pi/2
    yaw = 0

    #convert roll pitch yaw into quaternion
    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)

    pose_goal.orientation.w = cr * cp * cy + sr * sp * sy
    pose_goal.orientation.x = sr * cp * cy - cr * sp * sy
    pose_goal.orientation.y = cr * sp * cy + sr * cp * sy
    pose_goal.orientation.z = cr * cp * sy - sr * sp * cy 
    
  
    pose_goal.position.x = ((dy-ry)*tan(theta) - (rx-dx))/(cos(theta) + sin(theta)*tan(theta))
    pose_goal.position.y = (dy - ry - (dx - rx)*tan(theta))/(sin(theta)*tan(theta) + cos(theta))

    pose_goal.position.z = 1.3

    move_group.set_pose_target(pose_goal)

    ## Now, we call the planner to compute the plan and execute it.

    plan = move_group.go(wait=True)
    # Calling `stop()` ensures that there is no residual movement
    move_group.stop()
    # It is always good to clear your targets after planning with poses.
    # Note: there is no equivalent function for clear_joint_value_targets()
    move_group.clear_pose_targets()

    # For testing:
    # Note that since this section of code will not be included in the tutorials
    # we use the class variable rather than the copied state variable
    current_pose = move_group.get_current_pose().pose
    return all_close(pose_goal, current_pose, 0.01)

def reach_object(move_group):
   
    pose_goal = geometry_msgs.msg.Pose()
    current_pose = move_group.get_current_pose().pose
    pose_goal = current_pose

    pose_goal.position.z = 1.22
    move_group.set_pose_target(pose_goal)

    ## Now, we call the planner to compute the plan and execute it.
    
    plan = move_group.go(wait=True)

    #print (hello)
    #print (plan)
    # Calling `stop()` ensures that there is no residual movement
    move_group.stop()
    # It is always good to clear your targets after planning with poses.
    # Note: there is no equivalent function for clear_joint_value_targets()
    move_group.clear_pose_targets()

    # For testing:
    # Note that since this section of code will not be included in the tutorials
    # we use the class variable rather than the copied state variable
    current_pose = move_group.get_current_pose().pose
    return all_close(pose_goal, current_pose, 0.01)

def close_gripper():
    g = GripperCommandGoal()
    g.command.position = 0.33
    g.command.max_effort = -1
    rospy.loginfo("Sending close goal...")
    client.send_goal(g)
    client.wait_for_result()

def open_gripper():
    g = GripperCommandGoal()
    g.command.position = 0
    g.command.max_effort = -1
    rospy.loginfo("Sending open goal...")
    client.send_goal(g)
    client.wait_for_result()
  
def move_to_goal(xGoal,yGoal,orientation):
    ac = actionlib.SimpleActionClient("move_base", MoveBaseAction)

    while(not ac.wait_for_server(rospy.Duration.from_sec(5.0))):
        rospy.loginfo("Waiting for move_base action server to respond")

    goal = MoveBaseGoal()

    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

    goal.target_pose.pose.position = Point(xGoal,yGoal,0)
    goal.target_pose.pose.orientation.x = 0.0
    goal.target_pose.pose.orientation.y = 0.0
    goal.target_pose.pose.orientation.z = orientation[2]
    goal.target_pose.pose.orientation.w = orientation[3]
    rospy.loginfo("Sending goal...")
    ac.send_goal(goal)

    ac.wait_for_result(rospy.Duration(60))

if __name__ == '__main__':
  try:
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('pick_n_place', anonymous=True)

    robot = moveit_commander.RobotCommander()

    scene = moveit_commander.PlanningSceneInterface()

    group_name = "manipulator"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                   moveit_msgs.msg.DisplayTrajectory,
                                                   queue_size=20)

    #robot_pose_sub = rospy.Subscriber('/odom', Odometry, get_pose)
    #robot_pose = rospy.wait_for_message('/odom', Odometry).pose.pose
    
	
    #create client for the gripper
    client = actionlib.SimpleActionClient('gripper_controller/gripper_cmd', GripperCommandAction)
    print "Waiting for the gripper server..."
    client.wait_for_server()
    print "Connected to gripper server"


    

    object_count = 3
    for i in range(0, object_count):
        move_to_goal(pick_loc[0], pick_loc[1], pick_orien)
        go_to_pick_ee_pose(move_group, object_loc[i])
        open_gripper()
        reach_object(move_group)
        close_gripper()
        go_to_init_joints(move_group)
        move_to_goal(place_loc[i][0], place_loc[i][1], place_orien[i])
        go_to_ee_pose(move_group, place_ee_loc[i])
        open_gripper()
        go_to_init_joints(move_group)
  except KeyboardInterrupt:
    rospy.signal_shutdown("KeyboardInterrupt")
    raise

