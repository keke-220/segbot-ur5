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

class arm_client(object):
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)

        robot = moveit_commander.RobotCommander()

        scene = moveit_commander.PlanningSceneInterface()

        group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(group_name)

        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                       moveit_msgs.msg.DisplayTrajectory,
                                                       queue_size=20)

        #create client for the gripper
        client = actionlib.SimpleActionClient('gripper_controller/gripper_cmd', GripperCommandAction)
        print ("Waiting for the gripper server...")
        client.wait_for_server()
        print ("Connected to gripper server")

    def all_close(self, goal, actual, tolerance):
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
            return self.all_close(goal.pose, actual.pose, tolerance)

        elif type(goal) is geometry_msgs.msg.Pose:
            return self.all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

        return True

    def is_plan_found(self, robot_pose, robot_orien, ee_pose):
        '''return a bool value indicating whether a valid plan can be found. The plan will not be excuted.
        '''
        rx = robot_pose.x
        ry = robot_pose.y
        w = robot_orien.w
        x = robot_orien.x
        y = robot_orien.y
        z = robot_orien.z
        
        #convert robot's orientation (quaternion) to yaw value
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        theta = atan2(siny_cosp, cosy_cosp)

        #end effector pose
        dx = ee_pose.x
        dy = ee_pose.y
        pose_goal = geometry_msgs.msg.Pose()
        current_pose = self.move_group.get_current_pose().pose

        #set goal orientation to the orien of current pose
        pose_goal.orientation = current_pose.orientation
        pose_goal.position.x = ((dy-ry)*tan(theta) - (rx-dx))/(cos(theta) + sin(theta)*tan(theta))
        pose_goal.position.y = (dy - ry - (dx - rx)*tan(theta))/(sin(theta)*tan(theta) + cos(theta))
        #set the height of goal position
        #pose_goal.position.z = 1.3
        pose_goal.position.z = ee_pose.z

        #self.move_group.set_pose_target(pose_goal)

        ## Now, we call the planner to compute the plan and execute it.
        #print(self.move_group.plan(pose_goal).joint_trajectory.points)
        plan_points = self.move_group.plan(pose_goal).joint_trajectory.points
        if len(plan_points) == 0:
            print ("Plan not found. ")
            return False
        else:
            print ("At least one valid plan has been found. ")
            return True
 
    def move_to_ee_pose(self, robot_pose, robot_orien, ee_pose):
        '''Let the end effector go to a specific position. ee_pose is an absolute position. Goal orientation is set to the current orientation of the end effector. '''
        rx = robot_pose.x
        ry = robot_pose.y
        w = robot_orien.w
        x = robot_orien.x
        y = robot_orien.y
        z = robot_orien.z
        
        #convert robot's orientation (quaternion) to yaw value
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        theta = atan2(siny_cosp, cosy_cosp)

        #end effector pose
        dx = ee_pose.x
        dy = ee_pose.y
        pose_goal = geometry_msgs.msg.Pose()
        current_pose = self.move_group.get_current_pose().pose
        
        #set goal orientation to the orien of current pose
        pose_goal.orientation = current_pose.orientation
        pose_goal.position.x = ((dy-ry)*tan(theta) - (rx-dx))/(cos(theta) + sin(theta)*tan(theta))
        pose_goal.position.y = (dy - ry - (dx - rx)*tan(theta))/(sin(theta)*tan(theta) + cos(theta))
        
        #set the height of goal position
        #pose_goal.position.z = 1.3
        pose_goal.position.z = ee_pose.z

        self.move_group.set_pose_target(pose_goal)

        ## Now, we call the planner to compute the plan and execute it.
        #print(self.move_group.plan(pose_goal).joint_trajectory.points)
        plan = self.move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        self.move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        self.move_group.clear_pose_targets()

        # For testing:
        # Note that since this section of code will not be included in the tutorials
        # we use the class variable rather than the copied state variable
        current_pose = self.move_group.get_current_pose().pose
        return self.all_close(pose_goal, current_pose, 0.01)


if __name__ == '__main__':
    rospy.init_node('arm_client', anonymous=True)
    test = arm_client()
    robot_pose = rospy.wait_for_message('/odom', Odometry).pose.pose.position
    robot_orien = rospy.wait_for_message('/odom', Odometry).pose.pose.orientation
    print (robot_orien)
    goal_pose = Point(0, 0.81, 1.3)
    test.is_plan_found(Point(0, 0, 0), robot_orien, goal_pose)
