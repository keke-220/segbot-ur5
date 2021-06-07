#!/usr/bin/env python
import rospy
import time
import os
import json
import random
import datetime
from sensor_msgs.msg import Image
from chair_sampler import chair_sampler
from navigator import navigator
from arm_client import arm_client 
from camera_processor import camera_processor
from geometry_msgs.msg import Point, Quaternion
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

"""This moviet version requires python2. 
"""

"""Camera position is set to (0,6,0) in the gazebo env. 
   Camera's horizontal fov is set to be 1.047.
   Unloading object position is at the center of the image.
   The current FOV setting enables the camera to capture a 4x4 square.
"""
start = datetime.datetime.now()

camera_x = 0
camera_y = 6

image_h = 64
image_w = 64


rospy.init_node('data_collector', anonymous=False)

#create dir for storing data
path = 'data'
image_path = path + '/images'
if not os.path.exists(path):
    os.mkdir(path)
if not os.path.exists(image_path):
    os.mkdir(image_path)
annotation_file = path + '/annotations.txt'
scene_file = path+'/scenes.txt'
#the number of different scenes (with different chairs) we would like to sample
scene_num = 6

#the number of robot's unloading positions in each scene we would like to sample
robot_pose_num = 100

#the number of chairs in one side of the table
chair_num = 7

#unloading position range
x_range = [-1, 1]
y_range = [4.9, 5.6]  #don't sample on the table, but we can sample on the chairs

ee_goal = Point(0, 6, 1.3)

cp = camera_processor('/top_down_cam/image_raw')
cs = chair_sampler(chair_num)
nav = navigator()
ac = arm_client()

annotations = []
img_infos = []

for i in range(0, scene_num):

    print ("Generating scene id: " + str(i+1))
    cs.spawn()
    time.sleep(1)
    #save scene image to dataset
    cp.save_image(image_path+'/'+str(i+1)+'.jpg')
    print ("Scene image saved. ")
    
    #save the config of scene which enable us to produce more instances in the future
    chair_pose = cs.get_positions()
    chair_orien = cs.get_oriens()

    img_info = {"image_id": i+1,
                "chair_pose": chair_pose,
                "chair_orien": chair_orien}
    img_infos.append(img_info)
    with open(scene_file, 'w') as outfile:
        json.dump(img_infos, outfile)
    

    for j in range(0, robot_pose_num):
        print ("Sample position id: " + str(j+1)+ " from scene id: " + str(i+1))
        #sample a position in the range of the camera
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])

        #calculate the relative pixelwise position of the robot pose
        #the bottom left corner is the origin
        x_im = (x+2) * image_h/4
        y_im = image_w - ((y-4) * image_w/4)
        
        print ("Move the robot to the sampled goal position... ")
        #navigate to the goal position (Orientation doesn't matter)
        is_nav_success = nav.move_to_goal(Point(x, y, 0), Quaternion(0,0,0,1))

        #create a annotation dict for this instance
        ant = {'image_id': i+1,
               'robot_pose': [x_im, y_im],
               'unloading_result': False}
        result = False
        #label is trun only if it navigates to the correct position and generate an unloading plan
        if is_nav_success == True:
            robot_pose = rospy.wait_for_message('/odom', Odometry).pose.pose.position
            robot_orien = rospy.wait_for_message('/odom', Odometry).pose.pose.orientation
            if ac.is_plan_found(robot_pose, robot_orien, ee_goal):
                result = True
        #create a annotation dict for this instance
        ant = {'image_id': i+1,
               'robot_pose': [x_im, y_im],
               'unloading_result': result}
        print ("Unloading behavior result: " + str(ant['unloading_result']))
        annotations.append(ant)

        with open(annotation_file, 'w') as outfile:
            json.dump(annotations, outfile)
        
        #move robot back to its an open space
        print ("Move the robot back to an open space... ")
        nav.move_to_goal(Point(0,0,0), Quaternion(0,0,0,1))

   
    #delete all chairs
    cs.delete_all()
    #clear the map
    rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
#print (annotations)

end = datetime.datetime.now()
print (end-start)
