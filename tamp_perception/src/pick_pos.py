#!/usr/bin/env python

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionResult
from math import radians, degrees
from actionlib_msgs.msg import *
from geometry_msgs.msg import Point
from sound_play.libsoundplay import SoundClient

pick_loc = [0, 3]
pick_orien = [0.0, 0.0, 0, 1]


#goal_loc = [-4.08572691132,-5.66240519014]
#goal_orientation = [0.0,0.0,-0.878726986297,0.477324715004]

class nav():
    def __init__(self):

        self.goalReached = self.moveToGoal(pick_loc[0], pick_loc[1],pick_orien)
	#soundhandle = SoundClient()
	#soundhandle.say("Welcome to the lab")
    def shutdown(self):
        rospy.loginfo("Exiting...")
        rospy.sleep()

    def moveToGoal(self,xGoal,yGoal,orientation):
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
         
        #ac.wait_for_result(rospy.Duration(60))
        #print (rospy.wait_for_message('/move_base/result', MoveBaseActionResult))
        #if not find_plan:
        #    print ("plan not found")

if __name__ == '__main__':
    try:
        rospy.init_node('pick_pos', anonymous=False)
        nav()

	#soundhandle = SoundClient()
	#soundhandle.say("Welcome to the lab")
        #rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("nav node terminated")
