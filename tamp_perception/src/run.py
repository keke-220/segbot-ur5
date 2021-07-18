#!/usr/bin/env python
import rospy
import time
import os
import json
import random
import datetime
import shutil
import PIL.Image as IM
from math import sqrt, cos, sin
from sensor_msgs.msg import Image
from chair_sampler import chair_sampler
from navigator import navigator
from heatmap import heatmap
from arm_client import arm_client
from camera_processor import camera_processor
from geometry_msgs.msg import Point, Quaternion, Pose
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel, GetModelState, SetModelState
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty


def dist(p1, p2):
    return ((((p2[0] - p1[0] )**2) + ((p2[1]-p1[1])**2) )**0.5)

def point_to_pixel(point, origin, size, im_size):
    #Input a xy coordinate, output the corresponding image pixel point
    x = (point[0]-origin[0]+size[0]/2) * im_size[0] / size[0]
    y = (-point[1]+origin[1]+size[1]/2) *im_size[1] / size[1]
    return [int(x), int(y)]

def pixel_to_point(pixel, origin, size, im_size):
    #size: [4,4] im_size = [64,64]
    x = float(pixel[0] * size[0]) / float(im_size[0]) + origin[0] - size[0]/2
    y = (-1)* (float(pixel[1] * size[1]) / float(im_size[1]) - origin[1] - size[1]/2)
    return [x, y]

def euler_to_quat(roll, pitch, yaw):
    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return Quaternion(x, y, z, w)

def go_to_loading(nav):
    loading_pos = Point(0,1.05, 0)
    loading_orien = euler_to_quat(0, 0, -1.57)
    nav.move_to_goal(loading_pos, loading_orien)

def go_to_unloading(nav, max_side, goal):
    loading_pos = Point(goal[0], goal[1], 0)
    if max_side == "top":
        loading_pos = Point(goal[0], goal[1], 0)  ###localization issue, test for pushing the robot to the table
        loading_orien = euler_to_quat(0, 0, -1.57)
    else:
        loading_pos = Point(goal[0], goal[1], 0)
        loading_orien = euler_to_quat(0, 0, 1.57)
    nav.move_to_goal(loading_pos, loading_orien)

def open_gripper():
    g = GripperCommandGoal()
    g.command.position = 0
    g.command.max_effort = -1
    rospy.loginfo("Sending open goal...")
    client.send_goal(g)
    client.wait_for_result()

def load_object(ac, object_pose):
    robot_pose = rospy.wait_for_message('/odom', Odometry).pose.pose.position
    robot_orien = rospy.wait_for_message('/odom', Odometry).pose.pose.orientation

    ac.move_to_ee_pose(robot_pose, robot_orien, Point(object_pose[0], object_pose[1], 1.3))
    ac.open_gripper()
    ac.reach_object()
    ac.close_gripper()
    ac.go_to_init_joints()

def unload_object(ac, unloading_point):
    robot_pose = rospy.wait_for_message('/odom', Odometry).pose.pose.position
    robot_orien = rospy.wait_for_message('/odom', Odometry).pose.pose.orientation

    ac.move_to_ee_pose(robot_pose, robot_orien, Point(unloading_point[0], unloading_point[1], 1.3))
    ac.open_gripper()
    ac.go_to_init_joints()

def get_utility(cost, feasibility):
    reward = 40
    return (reward * feasibility - cost)

def predict(filename):
    #split and save the image into two images of both sides
    im = IM.open(filename)
    width, height = im.size
    left = 0
    right = width
    top = height/2
    bottom = height
    temp_image_path = "log/temp_image/images/"
    temp_annotation_path = "log/temp_image/annotations/"
    temp_result_path = "log/temp_result/"
    result_path = "log/results/"
    os.mkdir("log/temp_image")
    os.mkdir(temp_image_path)
    os.mkdir(temp_result_path)
    os.mkdir(temp_annotation_path)

    im_bot = im.crop((left, top, right, bottom))
    im_bot.save(temp_image_path+"bot_"+filename.split("/")[-1])

    im_top = im.crop((left, 0, right, top))
    im_top = im_top.rotate(180)
    im_top.save(temp_image_path+"top_"+filename.split("/")[-1])

    #duplicate some annotation images due to FCN code restriction
    im_an = IM.open("log/test.png")
    im_an.save(temp_annotation_path+"top_"+filename.split("/")[-1].split(".")[0] + ".png")
    im_an.save(temp_annotation_path+"bot_"+filename.split("/")[-1].split(".")[0] + ".png")
    
    #predict and save image to temp dir
    while not os.path.exists(temp_result_path+"pred_0.png"):        
        os.system('python predict.py')

    #save result to results dir
    im_res_bot = IM.open(temp_result_path+"pred_0.png")
    im_res_top = IM.open(temp_result_path+"pred_1.png")
    im_res_bot.save(result_path+"bot_"+filename.split("/")[-1])
    im_res_top.save(result_path+"top_"+filename.split("/")[-1])

    shutil.rmtree("log/temp_image")
    shutil.rmtree(temp_result_path)

def main():
    
    im_actual_size = [4,4]
    im_size = [64,64]
    

    rospy.init_node('run', anonymous=False)
    if not os.path.isdir('log/images'):
        os.mkdir("log/images")
    if not os.path.isdir('log/results'):
        os.mkdir("log/results")
    object_pose = [0, 0.75]
    unloading_range = [-6, 6]
    chair_num = 0
    sample_n = 1
    trial_num = 100
    pick_n = 1 #how many objects to pick and place in a trial
    success_range = 0.2 #success unloading range
    cp = camera_processor('/top_down_cam/image_raw')
    cs = chair_sampler(chair_num)
    hm = heatmap()
    ac = arm_client()
    ac.open_gripper()
    ac.go_to_init_joints()


    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    old_camera_state = model_coordinates("distorted_camera","")

    test_spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
    test_remover = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)

    #define some counters
    success_trial_num = 0
    overall_results = []

    for trial_idx in range(0, trial_num):
        init_pose = Point(0, 2, 0)
        init_orien = euler_to_quat(0, 0, -1.57)
        nav = navigator(init_pose, init_orien)
        print ("Generating scene id: " + str(trial_idx+1))
        cs.spawn_both_sides()
        time.sleep(1)
        
        #generate an object
        test_spawner(model_name = 'target_object',
            model_xml = open("/home/xiaohan/.gazebo/models/wood_cube_7_5cm/model.sdf", 'r').read(),
            robot_namespace = "/object",
            initial_pose = Pose(position=Point(object_pose[0],object_pose[1],1.1),orientation=Quaternion(0,0,0,1)),
            reference_frame = "world")

        #sample a random unloading object goal
        unloading_point = (random.uniform(unloading_range[0], unloading_range[1]), 6)
        print ("Unloading position: " + str(unloading_point))
        
        #move the camera at unloading point
        new_camera_state = ModelState()
        new_camera_state.model_name = "distorted_camera"
        new_camera_state.pose.orientation = old_camera_state.pose.orientation
        new_camera_state.pose.position = Point(unloading_point[0], unloading_point[1] , 3.46489)
        set_state(new_camera_state)
        time.sleep(1)

        #save the top down image
        input_image_name = "log/images"+'/'+str(trial_idx+1)+'.jpg'
        cp.save_image(input_image_name)
        
        #make prediction on the top down image, return two files of both sides
        predict(input_image_name)

        #get utility
        predicted_results = ["top_"+input_image_name.split('/')[-1], "bot_"+input_image_name.split('/')[-1]]
        max_utility = (-1)*float('inf')

        #testing one side
        predicted_results = ["top_test.jpg", "bot_test.jpg"]
        
        for result in predicted_results:
            unloading_robot_pose = [0,0]
            mirror_unloading_robot_pose = [0,0]
            side = result.split('_')[0]
            feasibility = hm.get_feasibility("log/results/"+result, sample_n)
            if side == "bot":
                unloading_robot_pixel = hm.sample_pixel("log/results/"+result, sample_n)
                if unloading_robot_pixel:
                    unloading_robot_pixel = (unloading_robot_pixel[0], unloading_robot_pixel[1]+im_size[1]/2)
                    unloading_robot_pose = pixel_to_point(unloading_robot_pixel, unloading_point, im_actual_size, im_size)
                    cost = nav.get_cost(Point(unloading_robot_pose[0], unloading_robot_pose[1], 0),Quaternion(0,0,0,1))
                    #cost = nav.get_cost(Point(unloading_robot_pose[0], 3.7, 0),Quaternion(0,0,0,1))
                else:
                    cost = float('inf')
            else:
                #if we use top side of the image, the goal pose have to be mirrored by the center of the camera
                mirror_unloading_robot_pixel = hm.sample_pixel("log/results/"+result, sample_n)
                if mirror_unloading_robot_pixel:
                    mirror_unloading_robot_pixel = (mirror_unloading_robot_pixel[0], mirror_unloading_robot_pixel[1]+im_size[1]/2)
                    mirror_unloading_robot_pose = pixel_to_point(mirror_unloading_robot_pixel, unloading_point, im_actual_size, im_size)
                    #print(mirror_unloading_robot_pose)
                    #print(unloading_point)
                    unloading_robot_pose[0] = 2*unloading_point[0]-mirror_unloading_robot_pose[0]
                    unloading_robot_pose[1] = 2*unloading_point[1]-mirror_unloading_robot_pose[1]
                    #unloading_robot_pixel = (im_size[0]-unloading_robot_pixel[0], im_size[1]/2-unloading_robot_pixel[1])
                    #unloading_robot_pose = pixel_to_point(unloading_robot_pixel, unloading_point, im_actual_size, im_size)
                    #print (unloading_robot_pose)

                   # mirror_pose = [2*unloading_point[0]-unloading_robot_pose[0], 2*unloading_point[1]-unloading_robot_pose[1]]
                    #unloading_robot_pose = mirror_pose
                    cost = nav.get_cost(Point(unloading_robot_pose[0], unloading_robot_pose[1], 0),Quaternion(0,0,0,1))
                    #cost = nav.get_cost(Point(unloading_robot_pose[0], 8.3, 0),Quaternion(0,0,0,1))
                else:
                    cost = float('inf')
            
            ut = get_utility(cost, feasibility)
            if ut > max_utility:
                max_utility = ut
                max_side = side
                nav_goal = unloading_robot_pose
                max_cost = cost
        print ("Choosing side: " + max_side)
        print ("Navigation goal: " + str(nav_goal))
                


        go_to_loading(nav)
        load_object(ac, object_pose)
        go_to_unloading(nav, max_side, nav_goal)

        if max_side == "top":   ###### an unloading policy change for top side
            top_unloading = [0,0]
            top_unloading[0] = unloading_point[0]
            top_unloading[1] = unloading_point[1]+0.15
            unload_object(ac, top_unloading)
        else:
            unload_object(ac, unloading_point)
        
        #check if the object is successfully unloaded to the target position within a specific range
        time.sleep(1)
        object_state = model_coordinates("target_object","")
        object_current_pose = [object_state.pose.position.x, object_state.pose.position.y]

        is_success = False
        if dist(object_current_pose, unloading_point) <= success_range:
            success_trial_num += 1
            is_success = True
            print ("What a successful unloading!!!")

        result = {}
        result["trial_id"] = trial_idx
        result["success"] = is_success
        result["utility"] = max_utility
        result["unloading_area"] = max_side
        result["nav_cost"] = max_cost #the cost from its thinking position

        overall_results.append(result)

        #reinitialize
        cs.delete_all()
        test_remover(model_name="target_object")

        with open("test_results.txt", 'w') as outfile:
            json.dump(overall_results, outfile)
    print (success_trial_num)


         

if __name__ == "__main__":
    main()
