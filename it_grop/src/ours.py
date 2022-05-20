#!/usr/bin/env python
import roslaunch
import rospy
import time
import os
import json
import random
import datetime
import shutil
import PIL.Image as IM
import copy
from math import sqrt, cos, sin, pi, acos, tan, atan2
from sensor_msgs.msg import Image
from chair_sampler import chair_sampler
from scene_sampler import scene_sampler
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
from voronoi import voronoi


def dist(p1, p2):
    return ((((p2[0] - p1[0] )**2) + ((p2[1]-p1[1])**2) )**0.5)

def point_to_pixel(point, origin, size, im_size):
    #Input a xy coordinate, output the corresponding image pixel point
    x = (point[0]-origin[0]+size[0]/2) * im_size[0] / size[0]
    y = (-point[1]+origin[1]+size[1]/2) *im_size[1] / size[1]
    return (int(x), int(y))

def pixel_to_point(pixel, origin, size, im_size):
    #size: [4,4] im_size = [64,64]
    x = float(pixel[0] * size[0]) / float(im_size[0]) + origin[0] - size[0]/2
    y = (-1)* (float(pixel[1] * size[1]) / float(im_size[1]) - origin[1] - size[1]/2)
    return (x, y)

def quat_to_yaw(x, y, z, w):
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    theta = atan2(siny_cosp, cosy_cosp)
    return (theta)

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

def go_to_loading(nav, loading_pose):
    loading_orien = euler_to_quat(0, 0, -1.57)
    nav.move_to_goal(loading_pose, loading_orien)

def go_to_unloading(nav, max_side, goal):
    loading_pos = goal
    is_there = False
    if goal!=None: 
        if max_side == "top":
            loading_pos = Point(goal.x, goal.y-0.06, 0)  ###localization issue, test1 for pushing the robot to the table
            loading_orien = euler_to_quat(0, 0, -1.57)
        else:
            #loading_pos = Point(goal[0], goal[1], 0)
            loading_orien = euler_to_quat(0, 0, 1.57)
        is_there = nav.move_to_goal(loading_pos, loading_orien)
    return is_there

def open_gripper():
    g = GripperCommandGoal()
    g.command.position = 0
    g.command.max_effort = -1
    rospy.loginfo("Sending open goal...")
    client.send_goal(g)
    client.wait_for_result()

def load_object(ac, object_pose, goal_pose):
    robot_pose = rospy.wait_for_message('/odom', Odometry).pose.pose.position
    robot_orien = rospy.wait_for_message('/odom', Odometry).pose.pose.orientation

    ac.move_to_ee_pose(robot_pose, robot_orien, Point(object_pose[0], object_pose[1], 1.3))
    ac.open_gripper()
    ac.reach_object()
    ac.close_gripper()
    ac.lift(1.32)
    #ac.go_to_init_joints()
    ac.move_to_ee_pose(Point(0,0,0), euler_to_quat(0,0,0), Point(goal_pose[0], goal_pose[1], 1.32))
    ac.open_gripper()
    ac.lift(1.4)
    ac.go_to_init_joints()

def unload_object(ac, object_pose, unloading_point):
    robot_pose = rospy.wait_for_message('/odom', Odometry).pose.pose.position
    robot_orien = rospy.wait_for_message('/odom', Odometry).pose.pose.orientation

    ac.move_to_ee_pose(robot_pose, robot_orien, Point(object_pose[0], object_pose[1], 1.35))
    ac.open_gripper()
    ac.lift(1.3)
    ac.close_gripper()
    ac.lift(1.35)
    ac.move_to_ee_pose(robot_pose, robot_orien, Point(unloading_point[0], unloading_point[1], 1.3))
    return
    ac.open_gripper()
    #ac.lift(1.32)
    #ac.go_to_init_joints()

def get_utility(cost, feasibility):
    reward = 40
    return (reward * feasibility - cost)
    #return (reward - cost)

def get_feasibility(filename, pixel):
    img = Image.open(filename)
    pixels = img.load()
    return pixels[pixel[0], pixel[1]]

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
   
    orien_bot = filename.split('_')[4][1]
    orien_top = filename.split('_')[4][0]

    #bot image
    os.mkdir("log/temp_image")
    os.mkdir(temp_image_path)
    os.mkdir(temp_result_path)
    os.mkdir(temp_annotation_path)
    
    im_bot = im.crop((left, top, right, bottom))
    im_bot.save(temp_image_path+str(orien_bot)+'_'+filename.split("/")[-1])
    #print(temp_image_path+str(orien_bot)+'_'+filename.split("/")[-1])
    #return
    #duplicate some annotation images due to FCN code restriction
    im_an = IM.open("log/test.png")
    im_an.save(temp_annotation_path+str(orien_bot)+'_'+filename.split("/")[-1].split(".")[0] + ".png")
    #predict and save image to temp dir
    while not os.path.exists(temp_result_path+"pred_0.png"):        
        os.system('python predict.py')

    im_res_bot = IM.open(temp_result_path+"pred_0.png") 
    im_res_bot.save(result_path+str(orien_bot)+'_'+filename.split("/")[-1])
    shutil.rmtree("log/temp_image")
    shutil.rmtree(temp_result_path)

  
    #top image
    os.mkdir("log/temp_image")
    os.mkdir(temp_image_path)
    os.mkdir(temp_result_path)
    os.mkdir(temp_annotation_path)
    
    im_top = im.crop((left, 0, right, top))
    im_top = im_top.rotate(180)
    im_top.save(temp_image_path+str(orien_top)+'_'+filename.split("/")[-1])
    #duplicate some annotation images due to FCN code restriction
    im_an = IM.open("log/test.png")
    im_an.save(temp_annotation_path+str(orien_top)+'_'+filename.split("/")[-1].split(".")[0] + ".png")
    #im_an.save(temp_annotation_path+str(orien_bot)+'_'+filename.split("/")[-1].split(".")[0] + ".png")
    #predict and save image to temp dir
    while not os.path.exists(temp_result_path+"pred_0.png"):        
        os.system('python predict.py')

   
    #save result to results dir
    #if orien_bot == 'b':
    im_res_top = IM.open(temp_result_path+"pred_0.png")
    im_res_top.save(result_path+str(orien_top)+'_'+filename.split("/")[-1])
    """
    if orien_bot == 'l':
        im_res_bot = IM.open(temp_result_path+"pred_1.png")
        im_res_top = IM.open(temp_result_path+"pred_0.png")
        im_res_bot.save(result_path+str(orien_bot)+'_'+filename.split("/")[-1])
        im_res_top.save(result_path+str(orien_top)+'_'+filename.split("/")[-1])
    """


    shutil.rmtree("log/temp_image")
    shutil.rmtree(temp_result_path)

#points that are overlapping with the table causing to be infeasible
def in_table(p,table_p):
    width = 6
    if abs(p[0]-table_p[0]) <= width and  abs(p[1]-table_p[1]) <= width:
        return True
    return False

def filter_infeasible_with_table(filename, tables):
    im = IM.open(filename)
    pixels = im.load()
    w, h = im.size
    for i in range(0, w):
        for j in range(0, h):
            for t in tables:
                t_p = point_to_pixel(t,(0,0),(10,10), (160,160))
                if in_table((i,j), t_p):
                    pixels[i,j] = 0
                    break
    im.save("global_heatmap.png")

def capture_image(old_camera_state, target, filename, rotation):
    new_camera_state = ModelState()
    new_camera_state.model_name = "distorted_camera"
    new_camera_state.pose.orientation = euler_to_quat(0, 1.57, 1.57)
    if rotation:
        new_camera_state.pose.orientation = euler_to_quat(1.57,1.57,1.57)
    new_camera_state.pose.position = Point(target[0], target[1], 3.46489)
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    cp = camera_processor('/top_down_cam/image_raw')
    set_state(new_camera_state)
    time.sleep(1)

    #save the top down image
    #input_image_name = "log/images"+'/tr'+str(trial_idx)+'_ta'+str(task_idx)+'_p'+str(pick_idx)+'.jpg'
    input_image_name = filename
    cp.save_image(input_image_name)

def capture_env(old_camera_state, target, env_name):
    new_camera_state = ModelState()
    new_camera_state.model_name = "distorted_camera"
    new_camera_state.pose.orientation = euler_to_quat(0, 1.57, 1.57)
    new_camera_state.pose.position = Point(target[0], target[1], 3.46489*2.5)
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    cp = camera_processor('/top_down_cam/image_raw')
    set_state(new_camera_state)
    time.sleep(1)
    cp.save_image(env_name)


def sim_point(p):
    x = p[0]+20
    return (x, p[1])

#action is a list of table id(s)
def visualize_heatmap(feas, state, action):
    im = IM.new('L', (160,160))
    pixels = im.load()
     
    #unloading one object at one table
    for k, v in feas[action[0]].items():
        if k in state[action[0]]:
            pixels[k[0],k[1]] = v

    if len(action) > 1:
        for t in action[1:]:
            for i in range(0, 160):
                for j in range(0, 160):
                    if (i, j) in feas[t].keys():
            #for k, v in feas[t].items():
                        pixels[i, j] = int(pixels[i,j]*feas[t][(i,j)]/255)
                    else:
                        pixels[i, j] = 0
    #for k, v in d.items():
    #    p = point_to_pixel(k, [0,0], [10,10], [160, 160])
    #    pixels[p[0],p[1]] = v
    im.save("global_heatmap.png")

def combine_state_space(state, com):
    new_state = copy.deepcopy(state)
    for c in com:
        for s in c[1:]:
            for p in state[s]:
                new_state[c[0]].append(p)
        for s in c[1:]:
            new_state[s] = new_state[c[0]]
    return new_state



def main():
    
    im_actual_size = [4,4]
    im_size = [64,64]
    rospy.init_node('run', anonymous=False)

    if not os.path.isdir('log/images'):
        os.mkdir("log/images")
    if not os.path.isdir('log/results'):
        os.mkdir("log/results")
    object_pose = [[0, 0.75], [0.2, 0.75], [-0.2, 0.75]] #manually define for 3 objects
    #object_pose = [0, 0.75]
    loaded_object_pose = [0.15, -0.3]
    unloading_range = [-6, 6]
    
    chair_num = 6 #this param is only for randomly generated scenes
    table_num = 7

    random_sample = False # sample baseline1
    sample_n = 1
    trial_num = 1  #scene change after each trial
    task_num = 1  #for each scene, the number of tested tasks
    pick_n = 3 #how many objects to pick and place in a trial
    success_range = 0.3 #success unloading range
    cp = camera_processor('/top_down_cam/image_raw')
    cs = scene_sampler(chair_num, table_num)
    sim_cs = scene_sampler(chair_num, 1)
    hm = heatmap()
    ac = arm_client()
    ac.open_gripper()
    ac.go_to_init_joints()
    loading_pose = Point(0, 1.05, 0)

    vo = voronoi(160, 160, 10, 10, [0,0])

    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    old_camera_state = model_coordinates("distorted_camera","")

    test_spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
    test_remover = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
    
    # robot initial position and orientation
    init_pose = Point(0, -4, 0)
    init_orien = euler_to_quat(0, 0, -1.57)
    
    #define some counters
    total_success_trial_num = 0
    overall_results = []

#=========================================
    for trial_idx in range(0, trial_num):

        nav = navigator(init_pose, init_orien)
         
        #randomly generate chairs for each scene
        print ("Generating scene id: " + str(trial_idx+1))
        cs.spawn_table()
        cs.spawn_chair()
        
        capture_env(old_camera_state, (0, 0), "log/envs/trial_"+str(trial_idx)+'.jpg')

        table_positions = cs.get_positions_table()
        chair_positions = cs.get_positions_chair()
        chair_oriens = cs.get_oriens_chair()
        
        # construct similar env next to the original one for getting top-down observations
        sim_chair_positions = []
        for p in chair_positions:
            sim_chair_positions.append(sim_point(p))
        sim_table_positions = []
        for p in table_positions:
            sim_table_positions.append(sim_point(p))
        sim_cs.spawn_chair_no_sample(sim_chair_positions, chair_oriens)

        sim_table_state = model_coordinates("table_cube_3","")

        for t in range(table_num):
            #vertical
            new_table_state = ModelState()
            new_table_state.model_name = "table_cube_3"
            new_table_state.pose.orientation = euler_to_quat(0,0,1.57)
            new_table_state.pose.position = Point(sim_table_positions[t][0], sim_table_positions[t][1], sim_table_state.pose.position.z)
            set_state(new_table_state)

            image_filename = "log/images/trial_" + str(trial_idx) + "_table_" + str(t) + "_rl" + '.jpg'
            capture_image(old_camera_state, (sim_table_positions[t][0], sim_table_positions[t][1]), image_filename, True)
            predict(image_filename)
 
            #horizontal
            new_table_state = ModelState()
            new_table_state.model_name = "table_cube_3"
            new_table_state.pose.orientation = euler_to_quat(0,0,0)
            new_table_state.pose.position = Point(sim_table_positions[t][0], sim_table_positions[t][1], sim_table_state.pose.position.z)
            set_state(new_table_state)

            image_filename = "log/images/trial_" + str(trial_idx) + "_table_" + str(t) + "_tb" + '.jpg'
            capture_image(old_camera_state, (sim_table_positions[t][0], sim_table_positions[t][1]), image_filename, False)
            predict(image_filename)

        #get voronoi graph   return -> key: pixel point -> value: 0-num_of_table
        pixel_state_dict = vo.generate_voronoi(table_positions)
        
        point_state_dict = {}
        for k in pixel_state_dict.keys():
            point_state_dict[pixel_to_point(k, (0,0), (10,10), (160,160))] = pixel_state_dict[k]
        
        state_point_dict = {}
        for t in range(table_num):
            state_point_dict[t] = []
        for k in point_state_dict.keys():
            state_point_dict[point_state_dict[k]].append(k)

        state_pixel_dict = {}
        for k, v in state_point_dict.items():
            state_pixel_dict[k] = []
            for p in v:
                state_pixel_dict[k].append(point_to_pixel(p, (0,0), (10,10), (160,160)))
                
    
        pixel_feas_dict = {}
        for t in range(table_num):
            pixel_feas_dict[t] = {}
            l_image_filename = "log/results/l_trial_" + str(trial_idx) + "_table_" + str(t) + "_rl" + '.jpg'
            r_image_filename = "log/results/r_trial_" + str(trial_idx) + "_table_" + str(t) + "_rl" + '.jpg'
            t_image_filename = "log/results/t_trial_" + str(trial_idx) + "_table_" + str(t) + "_tb" + '.jpg'
            b_image_filename = "log/results/b_trial_" + str(trial_idx) + "_table_" + str(t) + "_tb" + '.jpg'
            #bottom image
            b_im = IM.open(b_image_filename)
            im_w, im_h = b_im.size
            pixels = b_im.load()
            for i in range(0, im_w):
                for j in range(0, im_h):
                    ori_p = pixel_to_point((i,j+im_h), table_positions[t], im_actual_size, im_size)
                    new_p = (ori_p[0], ori_p[1])
                    new_pixel = point_to_pixel(new_p, (0,0), (10,10), (160,160))
                    if pixels[i,j] >0:
                        if new_pixel in pixel_feas_dict[t].keys():
                            pixel_feas_dict[t][new_pixel] = (pixel_feas_dict[t][new_pixel]+pixels[i,j])/2
                        else:
                            pixel_feas_dict[t][new_pixel] = pixels[i,j]


            #top image
            t_im = IM.open(t_image_filename)
            im_w, im_h = t_im.size
            pixels = t_im.load()
            for i in range(0, im_w):
                for j in range(0, im_h):
                    ori_p = pixel_to_point((i,j+im_h), table_positions[t], im_actual_size, im_size)
                    new_p = (2*table_positions[t][0]-ori_p[0], 2*table_positions[t][1]-ori_p[1])
                    new_pixel = point_to_pixel(new_p, (0,0), (10,10), (160,160))
                    if pixels[i,j] >0:
                        if new_pixel in pixel_feas_dict[t].keys():
                            pixel_feas_dict[t][new_pixel] = (pixel_feas_dict[t][new_pixel]+pixels[i,j])/2
                        else:
                            pixel_feas_dict[t][new_pixel] = pixels[i,j]

            #left image
            l_im = IM.open(l_image_filename)
            im_w, im_h = l_im.size
            pixels = l_im.load()
            for i in range(0, im_w):
                for j in range(0, im_h):
                    ori_p = pixel_to_point((i,j+im_h), table_positions[t], im_actual_size, im_size)
                    new_p = (table_positions[t][0]-table_positions[t][1]+ori_p[1],table_positions[t][1]+table_positions[t][0]-ori_p[0])
                    new_pixel = point_to_pixel(new_p, (0,0), (10,10), (160,160))
                    if pixels[i,j] >0:
                        if new_pixel in pixel_feas_dict[t].keys():
                            pixel_feas_dict[t][new_pixel] = (pixel_feas_dict[t][new_pixel]+pixels[i,j])/2
                        else:
                            pixel_feas_dict[t][new_pixel] = pixels[i,j]


            
            #right image
            r_im = IM.open(r_image_filename)
            im_w, im_h = r_im.size
            pixels = r_im.load()
            for i in range(0, im_w):
                for j in range(0, im_h):
                    ori_p = pixel_to_point((i,j+im_h), table_positions[t], im_actual_size, im_size)
                    new_p = (table_positions[t][0]+table_positions[t][1]-ori_p[1],table_positions[t][1]-table_positions[t][0]+ori_p[0])
                    new_pixel = point_to_pixel(new_p, (0,0), (10,10), (160,160))
                    if pixels[i,j] >0:
                        if new_pixel in pixel_feas_dict[t].keys():
                            pixel_feas_dict[t][new_pixel] = (pixel_feas_dict[t][new_pixel]+pixels[i,j])/2
                        else:
                            pixel_feas_dict[t][new_pixel] = pixels[i,j]

        action_table = [4]

        #combined_state = combine_state_space(state_pixel_dict, [[0,1]])

        visualize_heatmap(pixel_feas_dict, state_pixel_dict, action_table)               
        #visualize_heatmap(pixel_feas_dict, combined_state, action_table)               
        filter_infeasible_with_table('global_heatmap.png', table_positions)






        

            

        #make prediction on the top down image, return two files of both sides
        #predict(input_image_name)

   
        time.sleep(3)
        cs.delete_all()
        sim_cs.delete_all()
    return












#==========================================

    #with open("unloading_pos.txt", 'r') as f:
    #    unloading_dict = json.load(f)

    for trial_idx in range(0, trial_num):

        nav = navigator(init_pose, init_orien)
         
        """ 
        #randomly generate chairs for each scene
        print ("Generating scene id: " + str(trial_idx+1))
        cs.spawn_both_sides()
        time.sleep(1)
        """
         
        diff = 'hard'
        map_dir = 'grid_maps/'+diff+'/'
        #map_id = 'c15_0'
        
        #change the chair num accroding to the difficulty of scenes
        if diff == 'normal':
            chair_num = 20
        elif diff == 'easy':
            chair_num = 15
        elif diff == 'hard':
            chair_num = 25
        trial_idx = 9
        map_id = 'c'+ str(chair_num)+ '_' + str(trial_idx)
        
        #load pre-saved map:
        package = 'map_server'
        executable = 'map_server'
        arg = '$(find tamp_perception)/src/'+map_dir+map_id+'.yaml'
        node = roslaunch.core.Node(package, executable, args=arg)
        launch = roslaunch.scriptapi.ROSLaunch()
        launch.start()
        process = launch.launch(node)
        
        #load pre-saved scene:
        scene_file = map_dir+"scenes.txt"
        cs.reconstruct_env(scene_file, map_id)
        
        
        for task_idx in range(0, task_num):

            task_result = {}
            task_result['scene_id'] = map_id
            task_result['task_id'] = task_idx
            print ("################### TASK: " + str(task_idx) + " ###################")
    
            nav = navigator(init_pose, init_orien)
 


            # Planning...
            
            #test: remove object when navigating
            for pick_idx in range(0, pick_n):
                test_remover(model_name="target_object"+str(pick_idx))
 

            unloading_points = []

            #planning-step1: motion planner generates a cost function, mainly for navigation costs

            cost_function = {}
            cost_function['load'] = 1
            cost_function['unload'] = 1
            cost_function['navigate_to'] = {}
            
            access_loc = {} #a dict for storing symbolic navigation goal and 2d space navigation goal
            access_loc['src'] = loading_pose

            feasibility_loc = {}

            #sample 2d space navigation goal using FCN
            for pick_idx in range(0, pick_n):
                #sample a random unloading object goal
                #unloading_point = (random.uniform(unloading_range[0], unloading_range[1]), 6)
                #read unloading object goal from file
                unloading_point = (unloading_dict[map_id][3*task_idx+pick_idx], 6)
                unloading_points.append(unloading_point)
                #print ("Unloading position: " + str(unloading_point))
                
                #move the camera at unloading point
                new_camera_state = ModelState()
                new_camera_state.model_name = "distorted_camera"
                new_camera_state.pose.orientation = old_camera_state.pose.orientation
                new_camera_state.pose.position = Point(unloading_point[0], unloading_point[1] , 3.46489)
                set_state(new_camera_state)
                time.sleep(1)

                #save the top down image
                input_image_name = "log/images"+'/tr'+str(trial_idx)+'_ta'+str(task_idx)+'_p'+str(pick_idx)+'.jpg'
                cp.save_image(input_image_name)
                
                #make prediction on the top down image, return two files of both sides
                predict(input_image_name)

                #get utility
                predicted_results = ["top_"+input_image_name.split('/')[-1], "bot_"+input_image_name.split('/')[-1]]

                #predicted_results = ["top_"+input_image_name.split('/')[-1], "bot_test.jpg"]
                #testing one side
                if random_sample == True:
                    predicted_results = ["top_random_test.jpg", "bot_random_test.jpg"]
                    
              
                
                for result in predicted_results:
                    unloading_robot_pose = None
                    mirror_unloading_robot_pose = [0,0]
                    side = result.split('_')[0]
                    feasibility = hm.get_feasibility("log/results/"+result, sample_n)
                    if side == "bot":
                        if random_sample == True:
                            unloading_robot_pixel = hm.random_sample_pixel('log/results/'+result, sample_n)
                        else:
                            unloading_robot_pixel = hm.sample_pixel("log/results/"+result, sample_n)
                        if unloading_robot_pixel:
                            unloading_robot_pixel = (unloading_robot_pixel[0], unloading_robot_pixel[1]+im_size[1]/2)
                            xy_pose = pixel_to_point(unloading_robot_pixel, unloading_point, im_actual_size, im_size)
                            unloading_robot_pose =  Point(xy_pose[0], xy_pose[1], 0)

                        access_loc['l'+str(pick_idx)+'_bot'] = unloading_robot_pose
                        feasibility_loc['l'+str(pick_idx)+'_bot'] = feasibility

                    else:    
                        if random_sample == True:
                            mirror_unloading_robot_pixel = hm.random_sample_pixel('log/results/'+result, sample_n)
                        else:

                        #if we use top side of the image, the goal pose have to be mirrored by the center of the camera
                            mirror_unloading_robot_pixel = hm.sample_pixel("log/results/"+result, sample_n)
                        if mirror_unloading_robot_pixel:
                            mirror_unloading_robot_pixel = (mirror_unloading_robot_pixel[0], mirror_unloading_robot_pixel[1]+im_size[1]/2)
                            mirror_unloading_robot_pose = pixel_to_point(mirror_unloading_robot_pixel, unloading_point, im_actual_size, im_size)
                            unloading_robot_pose = Point(2*unloading_point[0]-mirror_unloading_robot_pose[0], 2*unloading_point[1]-mirror_unloading_robot_pose[1], 0)

                        access_loc['l'+str(pick_idx)+'_top'] = unloading_robot_pose
                        feasibility_loc['l'+str(pick_idx)+'_top'] = feasibility
            
            #print (access_loc)
            #print (feasibility_loc)
            #calculate cost for each access points combination
            for k1 in feasibility_loc.keys():
                for k2 in feasibility_loc.keys():
                    k1_loc = k1.split('_')[0]
                    k2_loc = k2.split('_')[0]
                    if k1_loc != k2_loc:
                        v1 = access_loc[k1]
                        v2 = access_loc[k2]
                        if v1 == None or v2 == None:
                            cost = None
                        else:
                            cost = nav.make_plan(v1, v2)    
                        cost_function['navigate_to'][str(k1)+'-'+str(k2)] = cost


            for k2 in feasibility_loc.keys():
                v1 = access_loc['src']
                v2 = access_loc[k2]
                if v1 == None or v2 == None:
                    cost = None
                else:
                    cost = nav.make_plan(v1, v2)    
                cost_function['navigate_to']['src'+'-'+str(k2)] = cost

            #print (len(access_loc))
            #print (len(cost_function['navigate_to']))
            #print (cost_function['navigate_to'].keys())
            
            utility = {}
            utility['load'] = 1
            utility['unload'] = 1
            utility['navigate_to'] = {}
           
            for k, v in cost_function['navigate_to'].items():
                k1 = k.split('-')[0]
                k2 = k.split('-')[1]
                c = v
                if v == None:
                    c = 99
                utility['navigate_to'][k] = get_utility(c, feasibility_loc[k2])

            #print (utility['navigate_to'])
            utility['navigate_to']['linit-src'] = 1

            #planning step2: interact with task planner to choose a plan
            #read a pre-generated plans for specific task
            infile = open("plans.txt", 'r')
            lines = infile.readlines()
            i = 1
            max_utility = float('-inf')
            max_line = 0
            while i < len(lines):
                current_plan = lines[i]
                actions = lines[i].split(' ')
                total_utility = 0

                #cal total utility for each this plan
                for a in actions:
                    a_name = a.split('(')[0]
                    if a_name == 'load' or a_name == 'unload':
                        total_utility += utility[a_name]
                    else: #deal with navigation action
                        ls = a.split('(')[1].split(',')[0]
                        lg = a.split('(')[1].split(',')[1]
                        total_utility += utility[a_name][ls+'-'+lg]
                if total_utility > max_utility:
                    max_utility = total_utility
                    max_line = i
                i += 2
            #print (max_utility)
            #print (lines[max_line])
            optimal_plan = lines[max_line].split(' ')
            print (optimal_plan)
            
            # Execution...
            
            success_current_task = []


            go_to_loading(nav, loading_pose)

            unloading_time = []
            
            #Loading...
            """
            for pick_idx in range(0, pick_n):

                #generate an object
                test_spawner(model_name = 'target_object'+str(pick_idx),
                    model_xml = open("/home/xiaohan/.gazebo/models/wood_cube_7_5cm/model.sdf", 'r').read(),
                    robot_namespace = "/object",
                    initial_pose = Pose(position=Point(object_pose[pick_idx][0],object_pose[pick_idx][1],1.1),orientation=Quaternion(0,0,0,1)),
                    reference_frame = "world")


                load_object(ac, object_pose[pick_idx], loaded_object_pose[pick_idx]) 
            """        
    


            for a_idx in range(4, len(optimal_plan)): #already loaded object, start from the navigation action
                a = optimal_plan[a_idx]
                print ("#### Action being chosen: " + a)

                if a.split('(')[0] == 'unload':
                    object_idx = int(a.split('(')[1].split(',')[0][1])
                    unloading_point = unloading_points[object_idx]
                    #test: add object back when unloading: this walk around is because of the friction between object and serving plate
                    #step1: calculate absolute object (to be generated) pose
                    
                    temp_robot_pose = rospy.wait_for_message('/odom', Odometry).pose.pose.position
                    temp_robot_orien = rospy.wait_for_message('/odom', Odometry).pose.pose.orientation
                    rx = temp_robot_pose.x
                    ry = temp_robot_pose.y

                    theta = quat_to_yaw(temp_robot_orien.x, temp_robot_orien.y, temp_robot_orien.z, temp_robot_orien.w)

                    dx = loaded_object_pose[0]
                    dy = loaded_object_pose[1]

                    temp_unloaded = [0,0]
                    temp_unloaded[0] = rx+dx*cos(theta)-dy*sin(theta)
                    temp_unloaded[1] = ry+dy*cos(theta)+dx*sin(theta)
                    temp_orien = temp_robot_orien

                    #step2: add the object to the serving plate

                    test_spawner(model_name = 'target_object'+str(object_idx),
                        model_xml = open("/home/xiaohan/.gazebo/models/wood_cube_7_5cm/model.sdf", 'r').read(),
                        robot_namespace = "/object",
                        initial_pose = Pose(position=Point(temp_unloaded[0], temp_unloaded[1],1.4),orientation=temp_orien),
                        reference_frame = "world")


                    if nav_goal != None: 
                        if side == "top":   ###### test2 for an unloading policy change for top side
                            top_unloading = [0,0]
                            top_unloading[0] = unloading_point[0]+ (nav_goal.x-unloading_point[0])/1
                            #top_unloading[0] = unloading_point[0]
                            top_unloading[1] = unloading_point[1]+0.15
                            unloading_point = top_unloading
                        if side == "bot":   ###### test2 for an unloading policy change for bot side
                            bot_unloading = [0,0]
                            bot_unloading[0] = unloading_point[0]+ (nav_goal.x-unloading_point[0])/3
                            #bot_unloading[0] = unloading_point[0]
                            bot_unloading[1] = unloading_point[1]-0.1
                            #bot_unloading[1] = unloading_point[1]
                            unloading_point = bot_unloading
                        
                    print ("#### Attempting to unload at " + str(unloading_point))
                    unload_object(ac, temp_unloaded, unloading_point) 
                    return
                    #check if the object is successfully unloaded to the target position within a specific range
                    time.sleep(1)
                    object_state = model_coordinates("target_object"+str(object_idx),"")
                    object_current_pose = [object_state.pose.position.x, object_state.pose.position.y]

                    is_success = False
                    if dist(object_current_pose, unloading_point) <= success_range:
                        is_success = True
                        print ("What a successful unloading!!!")
                    success_current_task.append(is_success)
                    test_remover(model_name="target_object"+str(object_idx))
                    ac.go_to_init_joints()
                
                #navigation action
                else:
                    nav_loc = a.split('(')[1].split(',')[1]
                    side = nav_loc.split('_')[1]
                    nav_goal = access_loc[nav_loc]
                     
                    start_time = time.time()
                    print ("#### Navigating to " + nav_loc + str(nav_goal))
                    is_there = go_to_unloading(nav, side, nav_goal)

                    end_time = time.time()              
                    time.sleep(1)
                    if is_there:
                        unloading_time.append(end_time-start_time)
                    else:
                        unloading_time.append(60)
    
            print (access_loc)
            location = {}
            for k in access_loc.keys():
                if access_loc[k] != None:
                    location[k] = (access_loc[k].x, access_loc[k].y)
                else:
                    location[k] = None
            task_result['access_loc'] = location
            task_result['chosen_plan'] = optimal_plan
            task_result['utility_func'] = utility
            task_result['cost_func'] = cost_function
            task_result['success'] = success_current_task
            task_result['exec_cost'] = unloading_time
            overall_results.append(task_result)
            with open("test_results.txt", 'w') as outfile:
                json.dump(overall_results, outfile)           
            #print (total_success_trial_num)
        #reinitialize
        cs.delete(chair_num)
    #print (total_success_trial_num)


         

if __name__ == "__main__":
    main()
