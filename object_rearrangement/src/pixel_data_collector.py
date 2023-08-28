#!/usr/bin/env python
import rospy
import time
import os
import json
import random
import datetime
#from math import dist
from math import sqrt
from sensor_msgs.msg import Image
from PIL import Image as PIL_IM
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

#camera_pos = [0, 6]

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
scene_file = path+'/scenes.txt'
#the number of different scenes (with different chairs) we would like to sample
scene_num = 50

#the number of robot's unloading positions in each scene we would like to sample
robot_pose_num = 100
trial_per_pixel = 5

#the number of chairs in one side of the table
chair_num = 10

#unloading position range
#this is for object being in the middle
x_range = [-1.1, 1.1]
y_range = [4.2, 5.4]

#derive pixel range from xy position range:

#im_x_max = (x_range[1]+(w/2))/w * image_w
#im_int = int(im_x_max)
#im_x_range = [image_w/2-(im_int-(image_w/2)), im_int]

#im_y_max = int((camera_y+2-y_range[1])/4 * image_h) + 1

#im_x_range = [point_to_pixel([x_range[0], 0], camera_pos, size, im_size)[0] + 1, point_to_pixel([x_range[1], 0], camera_pos, size, im_size)[0]]
#im_y_range = [point_to_pixel([0, y_range[1]], camera_pos, size, im_size)[1], point_to_pixel([0, y_range[0]], camera_pos, size, im_size)[1]]

tolerance = 0.2
sample_threshold = 5

max_reach = 1.1
max_reach_im = float(max_reach)/float(size[1]) *im_size[1]
min_reach = 0.5
min_reach_im = float(min_reach)/float(size[1]) *im_size[1]



cp = camera_processor('/top_down_cam/image_raw')
cs = chair_sampler(chair_num, [0,6], 13.5, 0.4)
nav = navigator(Point(0,2,0), Quaternion(0,0,0,1))
ac = arm_client()

annotations = []
img_infos = []


#Generate a grey wall in front of the robot to protect it 
test_spawner(model_name = 'grey_wall_protect',
        model_xml = open("/home/xiaohan/.gazebo/models/grey_wall/model.sdf", 'r').read(),
        robot_namespace = "/protect_wall",
        initial_pose = Pose(position=Point(0,3,0),orientation=Quaternion(0,0,0,1)),
        reference_frame = "world")

default_state = model_coordinates("distorted_camera","")
camera_state = ModelState()
camera_state.model_name = "distorted_camera"
camera_state.pose = default_state.pose
set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

#more general model can adapt different unloading positions on the table
#obj_num = 2
#camera_pos_set = [5.8, 6, 6.2]
for i in range(0, scene_num):
    #random object unloading positions
    obj_pos = random.uniform(5.8, 6.2)
    camera_pos = [0, obj_pos]
    #camera_pos = [0, camera_pos_set[i]]
    ee_goal = Point(0, camera_pos[1], 1.3)
    im_x_range = [point_to_pixel([x_range[0], 0], camera_pos, size, im_size)[0] + 1, point_to_pixel([x_range[1], 0], camera_pos, size, im_size)[0]]
    im_y_range = [point_to_pixel([0, y_range[1]], camera_pos, size, im_size)[1], point_to_pixel([0, y_range[0]], camera_pos, size, im_size)[1]]

    #print(point_to_pixel([0, obj_pose], camera_pos, size, im_size))
    #exit()
    #create candidate set of pixels from im_x_range and im_y_range
    pixel_set = []
    for pi in range(im_x_range[0], im_x_range[1]):
        for pj in range(im_y_range[0], im_y_range[1]):
            distance = dist([pi,pj], [im_size[0]/2, im_size[1]/2])
            if distance <= max_reach_im and distance >= min_reach_im:
                pixel_set.append([pi,pj])
    print (len(pixel_set))
    #back to original
    set_state(camera_state)
    #move as the object pose
    new_state = ModelState()
    new_state.model_name = "distorted_camera"
    new_state.pose = default_state.pose
    #print (new_state.pose)
    new_state.pose.position.y = camera_pos[1]
    set_state(new_state)

    print ("Generating scene id: " + str(i+1))
    cs.spawn()
    time.sleep(1)
    #save scene image to dataset

    test_remover(model_name='test_stick') 

    cp.save_image(image_path+'/'+str(i+1)+'.jpg')
    #im1 = PIL_IM.open(image_path+'/'+str(i+1)+'.jpg')
    #width, height = im1.size
    #left = 0
    #right = width
    #top = height/2
    #bottom = height
    #im2 = im1.crop((left, top, right, bottom))
    #im2.save(image_path+'/'+str(i+1)+'.jpg')   
    print ("Scene image saved. ")
    
    #save the config of scene which enable us to produce more instances in the future
    chair_pose = cs.get_positions()
    chair_orien = cs.get_oriens()
    print (chair_pose)
    
    #img_info = {"image_id": i+1,
    #            "chair_pose": chair_pose,
    #            "chair_orien": chair_orien}
    #img_infos.append(img_info)
    #with open(scene_file, 'w') as outfile:
    #    json.dump(img_infos, outfile)


    #provide a basic method for filtering out goal positions that has collision with chairs
    delete_list = []
    for pixel_index in range(0, len(pixel_set)):
        goal = pixel_to_point(pixel_set[pixel_index], camera_pos, size, im_size)
        delete_signal = False
        for p in chair_pose:
            if dist(goal, p) <= augmented_chair_r:
                delete_signal = True
        if delete_signal == True:
            delete_list.append(pixel_set[pixel_index])
    for item in delete_list:
        pixel_set.remove(item)
            
    print (len(pixel_set))


    #for j in range(0, robot_pose_num):
    for j in range(0, len(pixel_set)):
        print ('\n')
        print ("Sample position id: " + str(j+1)+ " from scene id: " + str(i+1))
        goal = pixel_to_point(pixel_set[j], camera_pos, size, im_size)
         
        #we only care about goal positions that can be sent by navigator
        if not is_goal_in_chair(goal, test_spawner, model_coordinates, test_remover):
            
            k = 0 #number of valid trials
            #sample_n = 0 #number of samplings for each trail

            while k < trial_per_pixel:
                #sample_n += 1
                #sample and calculate xy coordinate from pixel point
                print ("Trial: " + str(k+1))  

                #print ("Move the robot to the sampled goal position... ")
                #navigate to the goal position (Orientation doesn't matter)
                #is_nav_success = nav.move_to_goal(Point(goal[0], goal[1], 0), Quaternion(0,0,0,1))
                print ("Sample a robot actual position around the goal position...")

                actual_goal = sample_actual_goal_normal(goal, tolerance)
                
                #print (goal[0])
                #print (actual_goal[0])

                result = False
                #label is trun only if it navigates to the correct position and generate an unloading plan
                if not is_goal_in_chair(actual_goal, test_spawner, model_coordinates, test_remover):
                    #robot_pose = rospy.wait_for_message('/odom', Odometry).pose.pose.position
                    #robot_orien = rospy.wait_for_message('/odom', Odometry).pose.pose.orientation
                    #robot is always set to face the banquet table
                    #p = pixel_to_point(pixel_set[j],camera_pos,size,im_size)

                    if ac.is_plan_found(Point(actual_goal[0], actual_goal[1], 0), Quaternion(0, 0, sqrt(2)/2, sqrt(2)/2), ee_goal):
                        result = True
                #create a annotation dict for this instance
                ant = {'image_id': i+1,
                       'robot_pose': pixel_set[j],
                       'unloading_result': result}
                print ("Unloading behavior result: " + str(ant['unloading_result']))
                annotations.append(ant)

                with open(annotation_file, 'w') as outfile:
                    json.dump(annotations, outfile)
                
                #move robot back to its an open space
                #print ("Move the robot back to an open space... ")
                #nav.move_to_goal(Point(0,3,0), Quaternion(0,0,0,1))
                
                test_remover(model_name='test_stick')
                k += 1
                #sample_n = 0

    #delete all chairs
    cs.delete_all()
    #clear the map
    rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
#print (annotations)

end = datetime.datetime.now()
print (end-start)
