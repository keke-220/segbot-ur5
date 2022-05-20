#!/usr/bin/env python
import os
import rospy
import actionlib
import time
import random
import dynamic_reconfigure.cfg
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionResult
from math import radians, degrees, pi, sin, cos, sqrt
from actionlib_msgs.msg import GoalStatusArray
from geometry_msgs.msg import Point, Quaternion, PoseStamped
from nav_msgs.msg import Path
from nav_msgs.srv import GetPlan
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel, GetModelState, SetModelState
from gazebo_msgs.msg import ModelState

class navigator():
    
    def __init__(self, pose, orien):

        os.system("rosservice call /move_base/clear_costmaps")
        self.ac = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        while(not self.ac.wait_for_server(rospy.Duration.from_sec(5.0))):
            rospy.loginfo("Waiting for move_base action server to respond")
        #move to origin to enable action client to have previous status value
        testgoal = MoveBaseGoal()
        testgoal.target_pose.header.frame_id = "map"
        testgoal.target_pose.header.stamp = rospy.Time.now()

        testgoal.target_pose.pose.position = pose
        testgoal.target_pose.pose.orientation = orien

        self.ac.send_goal(testgoal)
        self.ac.wait_for_result(rospy.Duration(60))
        print ("Test result: action client works fine.")
 
    #we don't consider orientation in this virtual navigator   


    def move_to_goal(self, p, o):
        '''take point and Quaternion objects as inputs'''
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        goal.target_pose.pose.position = p
        goal.target_pose.pose.orientation = o
        
        if p != None:
            self.ac.send_goal(goal)
            self.ac.wait_for_result(rospy.Duration(60))
            print ("Robot stopped. ")
            state = rospy.wait_for_message('/move_base/status', GoalStatusArray)

        current_state = state.status_list[-1].status
        if current_state == 4:
            print ("Goal cannot be reached. ")
            return False
        else:
            print ("Goal reached. ")
            return True


    def dist(self, p1, p2):
        return ((((p2[0] - p1[0] )**2) + ((p2[1]-p1[1])**2) )**0.5)

    def make_plan(self, p1, p2):
        #store the robot current state
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        old_robot_state = model_coordinates("segbot","")
        o = old_robot_state.pose.orientation
        p = old_robot_state.pose.position

        start = PoseStamped()
        start.header.frame_id = 'map'
        start.header.stamp = rospy.Time.now()
        start.pose.position = p1
        start.pose.orientation = o

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = rospy.Time.now()
        goal.pose.position = p2
        goal.pose.orientation = o

        planner = rospy.ServiceProxy('/move_base/NavfnROS/make_plan', GetPlan)
        srv = GetPlan()
        srv.start = start
        srv.goal = goal
        srv.tolerance = 0.2

        plan = planner(start, goal, 0.2)
        #print (type(plan))
        
        poses = []
        for item in plan.plan.poses:
            poses.append((item.pose.position.x, item.pose.position.y))
        #print (poses)
        total_cost = 0
        for i in range(0, len(poses)-1):
            total_cost += self.dist(poses[i], poses[i+1])
        if total_cost == 0:
            total_cost = None
        #print (total_cost) 
        return total_cost
   


if __name__ == '__main__':
    try:
        rospy.init_node('navigator', anonymous=False)
        p1 = Point(0,3,0)
        p2 = Point(0,1.5,0)
        p3 = Point(3,5,0)
        p4 = Point(-4, 5, 0)
        p5 = Point(1.5, 7, 0)
        o = Quaternion(0,0,0,1)
        #print(vars(dynamic_reconfigure.cfg) )
        #cf = dynamic_reconfigure.boolParameter() 
        #rospy.ServiceProxy('/move_base/global_costmap/obstacle_layer_set_parameters', req, res)
        test = navigator(p1, o)
        #test.vir_nav(p, o)
        #test.sample_goal(p, 0.2)
        #test.move_to_goal(p, o)
        print(test.make_plan(p1, p2))
        print(test.make_plan(p2, p3))
        print(test.make_plan(p3, p4))
        print(test.make_plan(p4, p5))

        #test.make_plan(Point(0,1.05,0), Point(0, 6.5,0))
    except rospy.ROSInterruptException:
        rospy.loginfo("nav node terminated")


