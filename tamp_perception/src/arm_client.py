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
        self.init_joints = [0, -2.2, 2.2, -1.57, -1.57, 0]
        
        moveit_commander.roscpp_initialize(sys.argv)

        robot = moveit_commander.RobotCommander()

        scene = moveit_commander.PlanningSceneInterface()

        group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(group_name)

        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                       moveit_msgs.msg.DisplayTrajectory,
                                                       queue_size=20)

        #create client for the gripper
        self.client = actionlib.SimpleActionClient('gripper_controller/gripper_cmd', GripperCommandAction)
        print ("Waiting for the gripper server...")
        self.client.wait_for_server()
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

    def go_to_init_joints(self):


        joint_goal = self.move_group.get_current_joint_values()
        joint_goal[0] = self.init_joints[0]
        joint_goal[1] = self.init_joints[1]
        joint_goal[2] = self.init_joints[2]
        joint_goal[3] = self.init_joints[3]
        joint_goal[4] = self.init_joints[4]
        joint_goal[5] = self.init_joints[5]

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        self.move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        self.move_group.stop()

        ## END_SUB_TUTORIAL

        # For testing:
        current_joints = self.move_group.get_current_joint_values()
        return self.all_close(joint_goal, current_joints, 0.01)



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

    def close_gripper(self):
        g = GripperCommandGoal()
        g.command.position = 0.33
        g.command.max_effort = -1
        rospy.loginfo("Sending close goal...")
        self.client.send_goal(g)
        self.client.wait_for_result()

    def open_gripper(self):
        g = GripperCommandGoal()
        g.command.position = 0
        g.command.max_effort = -1
        rospy.loginfo("Sending open goal...")
        self.client.send_goal(g)
        self.client.wait_for_result()
    
    def lift(self, height):

        pose_goal = geometry_msgs.msg.Pose()
        current_pose = self.move_group.get_current_pose().pose
        pose_goal = current_pose

        pose_goal.position.z = height

        self.move_group.set_pose_target(pose_goal)

        ## Now, we call the planner to compute the plan and execute it.

        plan = self.move_group.go(wait=True)

        #print (hello)
        #print (plan)
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


    def reach_object(self):

        pose_goal = geometry_msgs.msg.Pose()
        current_pose = self.move_group.get_current_pose().pose
        pose_goal = current_pose
        pose_goal.position.z = 1.22
        self.move_group.set_pose_target(pose_goal)

        ## Now, we call the planner to compute the plan and execute it.

        plan = self.move_group.go(wait=True)

        #print (hello)
        #print (plan)
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

    def get_absolute_pos(self, robot_pose, robot_orien, ee_pose):
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
        return pose_goal.position

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

    def move_to_ee_pose_verticle(self, robot_pose, robot_orien, ee_pose):
        '''Let the end effector go to a specific position. ee_pose is an absolute position. Goal orientation is verticle to the table '''
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
        
        #set goal orientation to be verticle to the table
        #rotate the gripper so that it can grasp
        roll = 0
        pitch = 1.57
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
    #robot_pose = rospy.wait_for_message('/odom', Odometry).pose.pose.position
    #robot_orien = rospy.wait_for_message('/odom', Odometry).pose.pose.orientation
    #print (robot_orien)
    #goal_pose = Point(0, 0.81, 1.3)
    #test.is_plan_found(Point(0, 0, 0), robot_orien, goal_pose)
    test.close_gripper()
