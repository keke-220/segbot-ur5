#!/usr/bin/env python
import rospy
import time
import os
import json
import random
import datetime
#from math import dist
from math import sqrt, atan2
from sensor_msgs.msg import Image
from chair_sampler import chair_sampler
from navigator import navigator
from arm_client import arm_client 
from camera_processor import camera_processor
from geometry_msgs.msg import Point, Quaternion, Pose
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel, GetModelState, SetModelState
from gazebo_msgs.msg import ModelState 
"""This moviet version requires python2. 
"""

"""Camera position is set to (0,6,0) in the gazebo env. 
   Camera's horizontal fov is set to be 1.047.
   Unloading object position is at the center of the image.
   The current FOV setting enables the camera to capture a 4x4 square.
"""

"""This data collector collects data for every pixels in the camera image. 
   In order to save time, pixel_data_collector has been optimized in sampling goal positions compareing to
   data_collector.
   Vitual navigator is used for this collector.
"""
"""
0.82 is the max relative distance that the robot arm can reach in the current setting
y = 5.45 is the  max navigation point
"""

def dist(p1, p2):
    return ((((p2[0] - p1[0] )**2) + ((p2[1]-p1[1])**2) )**0.5)

def point_to_pixel(point, origin, size, im_size):
    #Input a xy coordinate, output the corresponding image pixel point
    x = (point[0]-origin[0]+size[0]/2) * im_size[0] / size[0]
    y = (-point[1]+origin[1]+size[1]/2) *im_size[1] / size[1]
    return [int(x), int(y)]

def pixel_to_point(pixel, origin, size, im_size):
    x = float(pixel[0] * size[0]) / float(im_size[0]) + origin[0] - size[0]/2
    y = (-1)* (float(pixel[1] * size[1]) / float(im_size[1]) - origin[1] - size[1]/2)
    return [x, y]

def is_goal_in_chair(goal, test_spawner, model_coordinates, test_remover):
    #identify if sampled goal position is in collision with chairs in the scene
    #The idea is to spawn a test object in the goal position and see if there is collision with the chairs
    test_spawner(model_name = 'test_stick',
            model_xml = open("/home/xiaohan/.gazebo/models/wood_stick_1cm/model.sdf", 'r').read(),
            robot_namespace = "/stick",
            initial_pose = Pose(position=Point(goal[0], goal[1],0),orientation=Quaternion(0,0,0,1)),
            reference_frame = "world")
    #print ("object spawened!")
    object_coordinates = model_coordinates("test_stick", "")
    result = True

    #time.sleep(1)

    if object_coordinates.pose.position.x - goal[0] < 0.001 and object_coordinates.pose.position.y - goal[1] < 0.001:  
        result = False
    if result:
        print ("collision!")   
    test_remover(model_name='test_stick')
    return result
    
def sample_actual_goal_normal(goal, tolerance):
    #x_range = [goal[0]-tolerance, goal[0]+tolerance]
    #y_range = [goal[1]-tolerance, goal[1]+tolerance]
    mu = 0
    sigma = tolerance/2
    x = goal[0]+random.gauss(mu, sigma)
    y = goal[1]+random.gauss(mu, sigma)
    print ("Sampled position: " + str([x,y]))
    return [x,y]

def quat_to_yaw(x, y, z, w):
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    theta = atan2(siny_cosp, cosy_cosp)
    return (theta)



def sample_actual_goal_random(goal, tolerance):
    x_range = [goal[0]-tolerance, goal[0]+tolerance]
    y_range = [goal[1]-tolerance, goal[1]+tolerance]
    while True:
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        if ((((x-goal[0])**2) + ((y-goal[1])**2))**0.5) <= tolerance:
            return [x,y]

#some objects used for the function is_goal_in_chair
test_spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
test_remover = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)

size = [4, 4] 
#w, h, indicating the actual domain size of the camera view

start = datetime.datetime.now()

camera_pos = [0, 6]

im_size = [64,64]  #w, h
"""
test = [3,8]
point = pixel_to_point(test,camera_pos, size, im_size)
print (point)
print (point_to_pixel())
"""


augmented_chair_r = 0.3

rospy.init_node('data_collector', anonymous=False)

#create dir for storing data
path = 'data'
image_path = path + '/images'
if not os.path.exists(path):
    os.mkdir(path)
if not os.path.exists(image_path):
    os.mkdir(image_path)
annotation_file = path + '/annotations.txt'
scene_file = 'scenes.txt'
#the number of different scenes (with different chairs) we would like to sample
scene_num = 1

#the number of robot's unloading positions in each scene we would like to sample
robot_pose_num = 100
trial_per_pixel = 5

#the number of chairs in one side of the table
chair_num = [20]

#unloading position range
#x_range = [-0.81, 0.81] #caculated from the current setting
#y_range = [4.98, 5.45]
x_range = [-1, 1]
y_range = [4.4, 5.4]

#derive pixel range from xy position range:

#im_x_max = (x_range[1]+(w/2))/w * image_w
#im_int = int(im_x_max)
#im_x_range = [image_w/2-(im_int-(image_w/2)), im_int]

#im_y_max = int((camera_y+2-y_range[1])/4 * image_h) + 1

im_x_range = [point_to_pixel([x_range[0], 0], camera_pos, size, im_size)[0] + 1, point_to_pixel([x_range[1], 0], camera_pos, size, im_size)[0]]
im_y_range = [point_to_pixel([0, y_range[1]], camera_pos, size, im_size)[1], point_to_pixel([0, y_range[0]], camera_pos, size, im_size)[1]]

tolerance = 0.2
sample_threshold = 5

max_reach = 1.1
max_reach_im = float(max_reach)/float(size[1]) *im_size[1]
min_reach = 0.5
min_reach_im = float(min_reach)/float(size[1]) *im_size[1]

ee_goal = Point(0, 6, 1.3)

cp = camera_processor('/top_down_cam/image_raw')
nav = navigator(Point(0,2,0), Quaternion(0,0,0,1))
ac = arm_client()

annotations = []
img_infos = []


default_state = model_coordinates("distorted_camera","")
camera_state = ModelState()
camera_state.model_name = "distorted_camera"
camera_state.pose = default_state.pose
set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)


for c_num in chair_num:

    cs = chair_sampler(c_num)
    for i in range(0, scene_num):

        #cs.spawn_both_sides()
        
        #map_id = 'c20_0'
        #scene_file = 'grid_maps/normal/scenes.txt'
        #cs.reconstruct_env(scene_file, map_id)
        
        chair_pose = []
        chair_orien = []
        for j in range(0, c_num):
            c_pose = model_coordinates("chair_2_"+str(j),"").pose.position
            c_orien = model_coordinates("chair_2_"+str(j),"").pose.orientation
            chair_pose.append([c_pose.x, c_pose.y])
            chair_orien.append(quat_to_yaw(c_orien.x, c_orien.y, c_orien.z, c_orien.w))

        
        #save the config of scene which enable us to produce more instances in the future
        #chair_pose = cs.get_positions()
        #chair_orien = cs.get_oriens()

        print (chair_pose)
        
        img_info = {"image_id": "c"+str(c_num)+"_"+str(i),
                    "chair_pose": chair_pose,
                    "chair_orien": chair_orien}
        img_infos.append(img_info)
        with open(scene_file, 'w') as outfile:
            json.dump(img_infos, outfile)

        os.system("rosservice call /gazebo_2Dmap_plugin/generate_map") 
        command = "rosrun map_server map_saver -f " + "c" + str(c_num) + "_" + str(i)  + " /map:=/map2d"
        os.system(command)

        #delete all chairs
        #cs.delete_all()

end = datetime.datetime.now()
print (end-start)
