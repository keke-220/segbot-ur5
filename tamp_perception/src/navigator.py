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

    def vir_nav(self, p, o):
        ''' This is a virtual navigation class. Given a goal, we don't let the robot to move physically. Instead, we want to know if a global plan can be generated. Return Ture if a plan is found, return False intead.
        '''
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        goal.target_pose.pose.position = p
        goal.target_pose.pose.orientation = o
        
        self.ac.send_goal(goal)
        
        #keep querying current state of the move_base to get if plan is valid or not in 5 seconds.
        #Possibly it is not the best to way to know if a plan is found.
        while True:
            time.sleep(0.1)
            state = rospy.wait_for_message('/move_base/status', GoalStatusArray)
            current_state = state.status_list[-1].status
            state_num = len(state.status_list)
            if state_num == 1:
                if current_state == 1 or current_state == 3:
                    print ("plan is valid")
                    self.ac.cancel_goal()
                    return True
                elif current_state == 4:
                    print ("cannot find a plan")
                    return False

    def sample_goal(self, p, tolerance):
        '''This function is used for virtual navigation. It will sample an actual goal position within the range of robot's navigation goal tolerance'''
        rd = random.uniform(0, 1)
        r = tolerance * sqrt(rd)
        theta = rd * 2 * pi
        x = p.x + r * cos(theta)
        y = p.y + r * sin(theta)
        return Point(x, y, 0)
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
   


    def get_cost(self, p1, p2):
        #print ("clearing costmaps...")
        #rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
        os.system("rosservice call /move_base/clear_costmaps")

        #store the robot current state
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        old_robot_state = model_coordinates("segbot","")
        o = old_robot_state.pose.orientation
        p = old_robot_state.pose.position
        #move robot to the starting point if it is too far from the current pos
        #print ("Calculating navigation cost...")
        if self.dist((p.x, p.y),(p1.x, p1.y)) >= 1.5:
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = rospy.Time.now()

            goal.target_pose.pose.position = p1
            goal.target_pose.pose.orientation = o

            self.ac.send_goal(goal)
            self.ac.wait_for_result(rospy.Duration(60))

        state = rospy.wait_for_message('/move_base/status', GoalStatusArray)

        temp_state = state.status_list[-1].status
        if temp_state == 4:
            print ("Goal cannot be reached. ")
            total_cost = None
        else: #reach starting point
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = rospy.Time.now()

            goal.target_pose.pose.position = p2
            goal.target_pose.pose.orientation = o

            self.ac.send_goal(goal)
            
            #while True:
            global_plan = rospy.wait_for_message('/move_base/NavfnROS/plan', Path)
            #print (global_plan)
            self.ac.cancel_goal()
            #print (len(global_plan.poses))
            poses = []
            for item in global_plan.poses:
                poses.append((item.pose.position.x, item.pose.position.y))
            #print (poses)
            total_cost = 0
            for i in range(0, len(poses)-1):
                total_cost += self.dist(poses[i], poses[i+1])
            if total_cost == 0:
                total_cost = None
        print (total_cost) 
        return total_cost

    def imagine_cost(self, p1, p2): #segbot need to be teleoprated to a start position p1

        #store the robot current state
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        old_robot_state = model_coordinates("segbot","")
    
        #move the camera to the start position
        new_robot_state = ModelState()
        new_robot_state.model_name = "segbot"
        new_robot_state.pose.orientation = old_robot_state.pose.orientation
        new_robot_state.pose.position = p1
        set_state(new_robot_state)
        time.sleep(1)

        #calculate cost from new state
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        goal.target_pose.pose.position = Point(6, 0, 0)
        goal.target_pose.pose.orientation = old_robot_state.pose.orientation

        self.ac.send_goal(goal)
        
        #while True:
        global_plan = rospy.wait_for_message('/move_base/NavfnROS/plan', Path)
        #print (global_plan)
        self.ac.cancel_goal()
        #print (len(global_plan.poses))
        poses = []
        for item in global_plan.poses:
            poses.append((item.pose.position.x, item.pose.position.y))
        #print (poses)
        total_cost = 0
        for i in range(0, len(poses)-1):
            total_cost += self.dist(poses[i], poses[i+1])
        #print (total_cost) 

        #move the robot back the old position
        new_robot_state = ModelState()
        new_robot_state.model_name = "segbot"
        new_robot_state.pose.orientation = old_robot_state.pose.orientation
        new_robot_state.pose.position = old_robot_state.pose.position
        set_state(new_robot_state)
        time.sleep(1)

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


