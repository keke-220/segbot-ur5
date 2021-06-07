#!/usr/bin/env python

import rospy
import actionlib
import time
import random
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionResult
from math import radians, degrees, pi, sin, cos, sqrt
from actionlib_msgs.msg import GoalStatusArray
from geometry_msgs.msg import Point, Quaternion


class navigator():
    
    def __init__(self):
        self.ac = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        while(not self.ac.wait_for_server(rospy.Duration.from_sec(5.0))):
            rospy.loginfo("Waiting for move_base action server to respond")
        #move to origin to enable action client to have previous status value
        testgoal = MoveBaseGoal()
        testgoal.target_pose.header.frame_id = "map"
        testgoal.target_pose.header.stamp = rospy.Time.now()

        testgoal.target_pose.pose.position = Point(0,0,0)
        testgoal.target_pose.pose.orientation = Quaternion(0,0,0,1)

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
                
if __name__ == '__main__':
    try:
        rospy.init_node('navigator', anonymous=False)
        p = Point(0,5.4,0)
        o = Quaternion(0,0,0,1)

        
        test = navigator()
        #test.vir_nav(p, o)
        #test.sample_goal(p, 0.2)
        test.move_to_goal(p, o)
    except rospy.ROSInterruptException:
        rospy.loginfo("nav node terminated")


