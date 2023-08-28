#!/usr/bin/env python
import copy
import datetime
import json
import os
import random
import shutil
import time
from itertools import combinations, permutations, product
from math import acos, atan2, cos, factorial, pi, sin, sqrt, tan
from multiprocessing import Manager, Process

import cma
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as IM
import roslaunch
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import (DeleteModel, GetModelState, SetModelState,
                             SpawnModel)
from geometry_msgs.msg import Point, Pose, Quaternion
from nav_msgs.msg import Odometry
from scipy.stats import multivariate_normal
from sensor_msgs.msg import Image
from std_srvs.srv import Empty

from arm_client import arm_client
from camera_processor import camera_processor
from chair_sampler import chair_sampler
from heatmap import heatmap
from navigator import navigator
from scene_sampler import scene_sampler
from voronoi import voronoi

im_actual_size = [4, 4]
im_size = [64, 64]
table_positions = [[0, 0]]
init_pose = Point(0, -4, 0)
sample_times = 1
object_num = 3
nav_variance = 0.03
max_sample = 10
nav_try = 3
man_try = 10


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


init_orien = euler_to_quat(0, 0, -1.57)
loaded_object_pose = [0.15, -0.3]

trial_num = 20
chair_num = 1

plate_r = 10


def dist(p1, p2):
    return ((((p2[0] - p1[0])**2) + ((p2[1] - p1[1])**2))**0.5)


def point_to_pixel(point, origin, size, im_size):
    #Input a xy coordinate, output the corresponding image pixel point
    x = (point[0] - origin[0] + size[0] / 2) * im_size[0] / size[0]
    y = (-point[1] + origin[1] + size[1] / 2) * im_size[1] / size[1]
    return (int(x), int(y))


def pixel_to_point(pixel, origin, size, im_size):
    #size: [4,4] im_size = [64,64]
    x = float(pixel[0] * size[0]) / float(im_size[0]) + origin[0] - size[0] / 2
    y = (-1) * (float(pixel[1] * size[1]) / float(im_size[1]) - origin[1] -
                size[1] / 2)
    return (x, y)


def quat_to_yaw(x, y, z, w):
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    theta = atan2(siny_cosp, cosy_cosp)
    return (theta)


def go_to_loading(nav, loading_pose):
    loading_orien = euler_to_quat(0, 0, -1.57)
    nav.move_to_goal(loading_pose, loading_orien)


def go_to_unloading(nav, max_side, goal):
    loading_pos = goal
    is_there = False
    if goal != None:
        if max_side == "top":
            # loading_pos = Point(
            #     goal.x, goal.y - 0.06, 0
            # )  ###localization issue, test1 for pushing the robot to the table
            loading_orien = euler_to_quat(0, 0, -1.57)
        elif max_side == "left":

            loading_orien = euler_to_quat(0, 0, 0)

        elif max_side == 'right':

            loading_orien = euler_to_quat(0, 0, 3.14)
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
    robot_orien = rospy.wait_for_message('/odom',
                                         Odometry).pose.pose.orientation

    ac.move_to_ee_pose(robot_pose, robot_orien,
                       Point(object_pose[0], object_pose[1], 1.3))
    ac.open_gripper()
    ac.reach_object()
    ac.close_gripper()
    ac.lift(1.32)
    #ac.go_to_init_joints()
    ac.move_to_ee_pose(Point(0, 0, 0), euler_to_quat(0, 0, 0),
                       Point(goal_pose[0], goal_pose[1], 1.32))
    ac.open_gripper()
    ac.lift(1.4)
    ac.go_to_init_joints()


def unload_object(ac, object_pose, unloading_point, object_name, model_n,
                  test_remover, side):
    height = 1.3
    ac.go_to_init_joints()
    robot_pose = rospy.wait_for_message('/odom', Odometry).pose.pose.position
    robot_orien = rospy.wait_for_message('/odom',
                                         Odometry).pose.pose.orientation

    ac.move_to_ee_pose(robot_pose, robot_orien,
                       Point(object_pose[0], object_pose[1], 1.35))
    ac.open_gripper()

    ac.lift(1.32)
    if object_name in ["teacup mat", "bread plate", "fruit bowl"]:
        ac.close_gripper_loose()
        # height = 1.35
    else:
        ac.close_gripper()
    if object_name in ["bread", "strawberry", "teacup lid"]:
        height = 1.32
    ac.lift(1.35)

    is_load = False
    try_times = 0
    rv = multivariate_normal(unloading_point,
                             [[nav_variance, 0], [0, nav_variance]])

    while try_times < man_try:
        print("Manipulation trying...")
        is_load = ac.move_to_ee_pose(
            robot_pose, robot_orien,
            Point(unloading_point[0], unloading_point[1], height))

        if is_load:
            break
        else:
            try_times += 1
            while True:
                unloading_point = rv.rvs()
                if abs(unloading_point[0]) < 0.225 and abs(
                        unloading_point[1]) < 0.225:
                    break

    if object_name == "dinner fork" or object_name == "dinner knife":
        if side == "top":
            if object_name == "dinner fork":
                is_load = ac.move_to_ee_pose(
                    robot_pose, robot_orien,
                    Point(unloading_point[0] + 0.055, unloading_point[1],
                          height))
            ac.rotate_left()

        if side == "bot":
            if object_name == "dinner fork":
                is_load = ac.move_to_ee_pose(
                    robot_pose, robot_orien,
                    Point(unloading_point[0] + 0.055, unloading_point[1],
                          height))
            ac.rotate_right()
        if side == "right":
            if object_name == "dinner fork":
                is_load = ac.move_to_ee_pose(
                    robot_pose, robot_orien,
                    Point(unloading_point[0] + 0.055, unloading_point[1],
                          height))
            ac.rotate_180()
    ac.open_gripper()
    ac.lift(1.35)
    ac.go_to_init_joints()
    if not is_load:
        test_remover(model_name=model_n)


def predict(filename):
    #split and save the image into two images of both sides
    im = IM.open(filename)
    width, height = im.size
    left = 0
    right = width
    top = height / 2
    bottom = height
    temp_image_path = "log/temp_image/images/"
    temp_annotation_path = "log/temp_image/annotations/"
    temp_result_path = "log/temp_result/"
    result_path = "log/results/"

    orien_bot = filename.split('_')[6][1]
    orien_top = filename.split('_')[6][0]

    #bot image
    os.mkdir("log/temp_image")
    os.mkdir(temp_image_path)
    os.mkdir(temp_result_path)
    os.mkdir(temp_annotation_path)

    im_bot = im.crop((left, top, right, bottom))
    im_bot.save(temp_image_path + str(orien_bot) + '_' +
                filename.split("/")[-1])
    #print(temp_image_path+str(orien_bot)+'_'+filename.split("/")[-1])
    #return
    #duplicate some annotation images due to FCN code restriction
    im_an = IM.open("log/test.png")
    im_an.save(temp_annotation_path + str(orien_bot) + '_' +
               filename.split("/")[-1].split(".")[0] + ".png")
    #predict and save image to temp dir

    #while not os.path.exists(temp_result_path+"pred_0.png"):
    os.system('python predict.py')

    im_res_bot = IM.open(temp_result_path + "pred_0.png")
    im_res_bot.save(result_path + str(orien_bot) + '_' +
                    filename.split("/")[-1])
    shutil.rmtree("log/temp_image")
    shutil.rmtree(temp_result_path)

    #top image
    os.mkdir("log/temp_image")
    os.mkdir(temp_image_path)
    os.mkdir(temp_result_path)
    os.mkdir(temp_annotation_path)

    im_top = im.crop((left, 0, right, top))
    im_top = im_top.rotate(180)
    im_top.save(temp_image_path + str(orien_top) + '_' +
                filename.split("/")[-1])
    #duplicate some annotation images due to FCN code restriction
    im_an = IM.open("log/test.png")
    im_an.save(temp_annotation_path + str(orien_top) + '_' +
               filename.split("/")[-1].split(".")[0] + ".png")
    #im_an.save(temp_annotation_path+str(orien_bot)+'_'+filename.split("/")[-1].split(".")[0] + ".png")
    #predict and save image to temp dir
    while not os.path.exists(temp_result_path + "pred_0.png"):
        os.system('python predict.py')

    #save result to results dir
    #if orien_bot == 'b':
    im_res_top = IM.open(temp_result_path + "pred_0.png")
    im_res_top.save(result_path + str(orien_top) + '_' +
                    filename.split("/")[-1])
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
def in_table(p, table_p):
    width = 6
    if abs(p[0] - table_p[0]) <= width and abs(p[1] - table_p[1]) <= width:
        return True
    return False


# def filter_infeasible_with_table(filename, tables):
#     im = IM.open(filename)
#     pixels = im.load()
#     w, h = im.size
#     for i in range(0, w):
#         for j in range(0, h):
#             for t in tables:
#                 t_p = point_to_pixel(t,(0,0),(10,10), (160,160))
#                 if in_table((i,j), t_p):
#                     pixels[i,j] = 0
#                     break
#     im.save("global_heatmap.png")


def capture_image(old_camera_state, target, filename, rotation):
    new_camera_state = ModelState()
    new_camera_state.model_name = "distorted_camera"
    new_camera_state.pose.orientation = euler_to_quat(0, 1.57, 1.57)
    if rotation:
        new_camera_state.pose.orientation = euler_to_quat(1.57, 1.57, 1.57)
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
    new_camera_state.pose.position = Point(target[0], target[1], 3.46489 * 2.5)
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    cp = camera_processor('/top_down_cam/image_raw')
    set_state(new_camera_state)
    time.sleep(1)
    cp.save_image(env_name)


def capture_config(old_camera_state, env_name):
    new_camera_state = ModelState()
    new_camera_state.model_name = "distorted_camera_clone"
    new_camera_state.pose.orientation = euler_to_quat(0, 1.57, 1.57)
    new_camera_state.pose.position = Point(0, 0, 2.5)
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    cp = camera_processor('/top_down_cam/image_goal')
    set_state(new_camera_state)
    time.sleep(1)
    cp.save_image(env_name)


def sim_point(p):
    x = p[0] + 20
    return (x, p[1])


#action is a list of table id(s)
def visualize_heatmap(feas, state, action):
    im = IM.new('L', (160, 160))
    pixels = im.load()

    #unloading one object at one table
    for k, v in feas[action[0]].items():
        if k in state[action[0]]:
            pixels[k[0], k[1]] = v

    if len(action) > 1:
        for t in action[1:]:
            for i in range(0, 160):
                for j in range(0, 160):
                    if (i, j) in feas[t].keys():
                        #for k, v in feas[t].items():
                        pixels[i,
                               j] = int(pixels[i, j] * feas[t][(i, j)] / 255)
                    else:
                        pixels[i, j] = 0
    #for k, v in d.items():
    #    p = point_to_pixel(k, [0,0], [10,10], [160, 160])
    #    pixels[p[0],p[1]] = v

    #testing purpose: resize
    big_im = IM.new('L', (640, 640))
    big_pixels = big_im.load()
    for i in range(0, 640):
        for j in range(0, 640):
            big_pixels[i, j] = pixels[i / 4, j / 4]

    im.save("global_heatmaps/global_heatmap_" + str(action[0]) + '.png')
    big_im.save("global_heatmaps/resized_global_heatmap_" + str(action[0]) +
                '.png')


def get_side(p1, p2):
    if p2[1] > (p1[1] + 0.25):
        return "top"
    elif p2[1] < (p1[1] - 0.25):
        return "bot"
    elif p2[0] < (p1[0] - 0.25):
        return "left"
    else:
        return "right"


def get_seq_candidates(object_num, abstract_pose):
    one_perm = []
    for i in range(object_num):
        one_perm.append(i)
    perms = list(permutations(one_perm))

    infeas_perms = set()
    for i in range(1, object_num + 1):
        for k, v in abstract_pose.items():
            if "u" + str(i) == k.split('_')[0] and v == "under":
                a_i = int(k.split('_')[1][1])
                for j, p in enumerate(perms):
                    if p.index(i - 1) > p.index(a_i - 1):
                        infeas_perms.add(j)
            if "u" + str(i) == k.split('_')[0] and v == "on top of":
                a_i = int(k.split('_')[1][1])
                for j, p in enumerate(perms):
                    if p.index(i - 1) < p.index(a_i - 1):
                        infeas_perms.add(j)
    res = []
    for i, p in enumerate(perms):
        if i not in infeas_perms:
            res.append(p)
    return res


def get_cost(nav, seq, object_pose):
    res = 0
    while True:
        x = random.uniform(-0.5, 0.5)
        y = random.uniform(-0.5, 0.5)
        if nav.make_plan(init_pose, Point(x, y, 0)):
            break
    table = Point(x, y, 0)
    res += nav.make_plan(
        init_pose, Point(object_pose[seq[0]][0], object_pose[seq[0]][1], 0))
    res += nav.make_plan(
        table, Point(object_pose[seq[0]][0], object_pose[seq[0]][1], 0))
    for o in seq[1:]:
        res += 2 * nav.make_plan(
            table, Point(object_pose[o][0], object_pose[o][1], 0))
    return res


def get_object_model(name):
    if name == "dinner fork":
        return "/home/bwilab/.gazebo/models_add/fork/model.sdf"
    if name == "dinner knife":
        return "/home/bwilab/.gazebo/models_add/knife/model.sdf"
    if name == "dinner plate":
        return "/home/bwilab/.gazebo/models_add/utensil_plate_blue/model.sdf"
    if name == "bread plate":
        return "/home/bwilab/.gazebo/models_add/utensil_plate_blue_small/model.sdf"
    if name == "water cup":
        return "/home/bwilab/.gazebo/models_add/cup_glass/model.sdf"
    if name == "bread":
        return "/home/bwilab/.gazebo/models_add/food_bread/model.sdf"
    if name == "strawberry":
        return "/home/bwilab/.gazebo/models_add/food_strawberry/model.sdf"
    if name == "teacup lid":
        return "/home/bwilab/.gazebo/models_add/cup_lid/model.sdf"
    if name == "teacup mat":
        return "/home/bwilab/.gazebo/models_add/utensil_mat_blue_small/model.sdf"
    if name == "teacup":
        return "/home/bwilab/.gazebo/models_add/cup_yellow/model.sdf"
    if name == "fruit bowl":
        return "/home/bwilab/.gazebo/models_add/bowl/model.sdf"


def main():

    rospy.init_node('run', anonymous=False)

    if not os.path.isdir('log/images'):
        os.mkdir("log/images")
    if not os.path.isdir('log/results'):
        os.mkdir("log/results")
    if os.path.isdir("log/temp_image"):
        shutil.rmtree("log/temp_image")
        shutil.rmtree("log/temp_result")

    table_num = len(table_positions)

    cp = camera_processor('/top_down_cam/image_raw')
    sim_cs = scene_sampler(chair_num, 1)
    hm = heatmap()
    ac = arm_client()
    ac.open_gripper()
    ac.go_to_init_joints()

    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state',
                                           GetModelState)
    old_camera_state = model_coordinates("distorted_camera", "")

    test_spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
    test_remover = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)

    # object_positions = [[[-1.8, -1], [-1.7, -1], [-1.6, -1]]]
    #=========================================
    for task_name in ["task6", "task7", "task8"]:
        for trial_idx in range(0, trial_num):

            nav = navigator(init_pose, init_orien)

            print("Generating scene id: " + str(trial_idx + 1))

            cs = scene_sampler(chair_num, table_num)
            cs.set_object_positions([[0, 0]])
            cs.set_positions_table([[0, 0]])
            cs.spawn_chair()
            chair_positions = cs.get_positions_chair()
            chair_oriens = cs.get_oriens_chair()

            capture_env(old_camera_state, (0, 0),
                        "log/envs/trial_" + str(trial_idx) + '.jpg')

            start_time = time.time()
            os.system("python3.7 " + task_name + '.py')
            """
            with open("gpt3_log/pose_objects.json") as fp:
                object_positions_candidates = json.load(fp)

            """
            with open("gpt3_log/objects.json") as fp:
                objects = json.load(fp)
            """
            with open("gpt3_log/abstract_pose.json") as fp:
                abstract_pose = json.load(fp)
            """

            object_num = len(objects)
            # seq_candidates = get_seq_candidates(object_num, abstract_pose)
            one_perm = []
            for i in range(object_num):
                one_perm.append(i)
            seq_candidates = list(permutations(one_perm))
            """
            for k, v in pose_objects.items():
                if k != 'table':
                    o_x = table_positions[0][0]
                    if v['y'] > 0.0:
                        o_y = float(v['y']) / 100 + float(
                            plate_r) / 100 + table_positions[0][1]
                    elif v['y'] < 0.0:
                        o_y = float(v['y']) / 100 - float(
                            plate_r) / 100 + table_positions[0][1]
                    else:
                        o_y = table_positions[0][1]
                    object_positions[0].append([o_x, o_y])
            print('Object positions: ' + str(object_positions))
            """
            # object_positions = [[[-2.16, -0.8], [-2.16, -1.1], [-2.16, -0.95]]]

            # ****************************************************************************
            initial_object_positions = []
            i = 0
            while i < object_num:
                sample_x = random.uniform(-4.5, 4.5)
                sample_y = random.uniform(-4.5, 4.5)
                if dist((sample_x, sample_y), (0, 0)) > 3 and dist(
                    (sample_x, sample_y), (init_pose.x, init_pose.y)) > 1:
                    initial_object_positions.append((sample_x, sample_y))
                    i += 1

            max_cost = float("inf")
            for seq in seq_candidates:
                seq_cost = get_cost(nav, seq, initial_object_positions)
                if seq_cost < max_cost:
                    selected_seq = seq
                    max_cost = seq_cost

            print("\nTask level results: ")
            print("inital_object_positions: " + str(initial_object_positions))
            print("seq: " + str(selected_seq))
            print("objects: " + str(objects))
            print("\nSelecting best base positions...")
            robot_base_points = []
            temp_robot_base_points = None
            temp_object_positions = None
            min_feas = float("-inf")
            for sample_n in range(sample_times):
                print("Sampling low-level object positions from LLM: " +
                      str(sample_n + 1) + '/' + str(sample_times))
                # **************************************************************
                # random.shuffle(object_positions_candidates)
                object_positions = []
                rv = multivariate_normal(
                    [0, 0], [[nav_variance, 0], [0, nav_variance]])
                for op in range(object_num):
                    while True:
                        oxy = rv.rvs()
                        if abs(oxy[0]) < 0.225 and abs(oxy[1]) < 0.225:

                            object_positions.append([oxy[0], oxy[1]])
                            break

                print("object goal positions: " + str(object_positions))
                time.sleep(3)
                print(
                    "\nLoading GROP model and predicting unloading heatmaps for all the objects..."
                )
                # construct similar env next to the original one for getting top-down observations
                sim_chair_positions = []
                for p in chair_positions:
                    sim_chair_positions.append(sim_point(p))
                sim_table_positions = []
                for p in table_positions:
                    sim_table_positions.append(sim_point(p))
                sim_object_positions = []
                for i in range(table_num):
                    for p in object_positions:
                        sim_object_positions.append(sim_point(p))
                sim_cs.spawn_chair_no_sample(sim_chair_positions, chair_oriens)
                sim_table_state = model_coordinates("table_cube_3", "")
                for t in range(table_num):
                    #vertical
                    new_table_state = ModelState()
                    new_table_state.model_name = "table_cube_3"
                    new_table_state.pose.orientation = euler_to_quat(
                        0, 0, 1.57)
                    new_table_state.pose.position = Point(
                        sim_table_positions[t][0], sim_table_positions[t][1],
                        sim_table_state.pose.position.z)
                    set_state(new_table_state)
                    for obj_idx in range(object_num):
                        image_filename = "log/images/trial_" + str(
                            trial_idx) + "_table_" + str(t) + "_obj_" + str(
                                obj_idx) + "_rl" + '.jpg'
                        capture_image(old_camera_state,
                                      (sim_object_positions[obj_idx][0],
                                       sim_object_positions[obj_idx][1]),
                                      image_filename, True)
                        predict(image_filename)
                    #horizontal
                    new_table_state = ModelState()
                    new_table_state.model_name = "table_cube_3"
                    new_table_state.pose.orientation = euler_to_quat(0, 0, 0)
                    new_table_state.pose.position = Point(
                        sim_table_positions[t][0], sim_table_positions[t][1],
                        sim_table_state.pose.position.z)
                    set_state(new_table_state)
                    for obj_idx in range(object_num):
                        image_filename = "log/images/trial_" + str(
                            trial_idx) + "_table_" + str(t) + "_obj_" + str(
                                obj_idx) + "_tb" + '.jpg'
                        capture_image(old_camera_state,
                                      (sim_object_positions[obj_idx][0],
                                       sim_object_positions[obj_idx][1]),
                                      image_filename, False)
                        predict(image_filename)
                print("Processing predicted heatmaps into feasibility...")
                pixel_feas_dict = {}
                for t in range(table_num):
                    pixel_feas_dict[t] = {}
                    object_side_feas = []
                    for obj_idx in range(object_num):
                        pixel_feas_dict[t][obj_idx] = {}
                        l_image_filename = "log/results/l_trial_" + str(
                            trial_idx) + "_table_" + str(t) + "_obj_" + str(
                                obj_idx) + "_rl" + '.jpg'
                        r_image_filename = "log/results/r_trial_" + str(
                            trial_idx) + "_table_" + str(t) + "_obj_" + str(
                                obj_idx) + "_rl" + '.jpg'
                        t_image_filename = "log/results/t_trial_" + str(
                            trial_idx) + "_table_" + str(t) + "_obj_" + str(
                                obj_idx) + "_tb" + '.jpg'
                        b_image_filename = "log/results/b_trial_" + str(
                            trial_idx) + "_table_" + str(t) + "_obj_" + str(
                                obj_idx) + "_tb" + '.jpg'
                        #bottom image
                        b_im = IM.open(b_image_filename)
                        im_w, im_h = b_im.size
                        pixels = b_im.load()
                        bot_feas = 0
                        for i in range(0, im_w):
                            for j in range(0, im_h):
                                bot_feas += float(pixels[i, j]) / 255

                                ori_p = pixel_to_point((i, j + im_h),
                                                       table_positions[t],
                                                       im_actual_size, im_size)
                                new_p = (ori_p[0], ori_p[1])
                                new_pixel = point_to_pixel(
                                    new_p, (0, 0), (10, 10), (160, 160))
                                if pixels[i, j] > 0:
                                    if new_pixel in pixel_feas_dict[t][
                                            obj_idx].keys():
                                        pixel_feas_dict[t][obj_idx][
                                            new_pixel] = (
                                                pixel_feas_dict[t][obj_idx]
                                                [new_pixel] + pixels[i, j]) / 2
                                    else:
                                        pixel_feas_dict[t][obj_idx][
                                            new_pixel] = pixels[i, j]
                        #top image
                        t_im = IM.open(t_image_filename)
                        im_w, im_h = t_im.size
                        pixels = t_im.load()
                        top_feas = 0
                        for i in range(0, im_w):
                            for j in range(0, im_h):
                                top_feas += float(pixels[i, j]) / 255
                                ori_p = pixel_to_point((i, j + im_h),
                                                       table_positions[t],
                                                       im_actual_size, im_size)
                                new_p = (2 * table_positions[t][0] - ori_p[0],
                                         2 * table_positions[t][1] - ori_p[1])
                                new_pixel = point_to_pixel(
                                    new_p, (0, 0), (10, 10), (160, 160))
                                if pixels[i, j] > 0:
                                    if new_pixel in pixel_feas_dict[t][
                                            obj_idx].keys():
                                        pixel_feas_dict[t][obj_idx][
                                            new_pixel] = (
                                                pixel_feas_dict[t][obj_idx]
                                                [new_pixel] + pixels[i, j]) / 2
                                    else:
                                        pixel_feas_dict[t][obj_idx][
                                            new_pixel] = pixels[i, j]
                        #left image
                        l_im = IM.open(l_image_filename)
                        im_w, im_h = l_im.size
                        pixels = l_im.load()
                        left_feas = 0
                        for i in range(0, im_w):
                            for j in range(0, im_h):

                                left_feas += float(pixels[i, j]) / 255
                                ori_p = pixel_to_point((i, j + im_h),
                                                       table_positions[t],
                                                       im_actual_size, im_size)
                                new_p = (table_positions[t][0] -
                                         table_positions[t][1] + ori_p[1],
                                         table_positions[t][1] +
                                         table_positions[t][0] - ori_p[0])
                                new_pixel = point_to_pixel(
                                    new_p, (0, 0), (10, 10), (160, 160))
                                if pixels[i, j] > 0:
                                    if new_pixel in pixel_feas_dict[t][
                                            obj_idx].keys():
                                        pixel_feas_dict[t][obj_idx][
                                            new_pixel] = (
                                                pixel_feas_dict[t][obj_idx]
                                                [new_pixel] + pixels[i, j]) / 2
                                    else:
                                        pixel_feas_dict[t][obj_idx][
                                            new_pixel] = pixels[i, j]
                        #right image
                        r_im = IM.open(r_image_filename)
                        im_w, im_h = r_im.size
                        pixels = r_im.load()
                        right_feas = 0
                        for i in range(0, im_w):
                            for j in range(0, im_h):

                                right_feas += float(pixels[i, j]) / 255
                                ori_p = pixel_to_point((i, j + im_h),
                                                       table_positions[t],
                                                       im_actual_size, im_size)
                                new_p = (table_positions[t][0] +
                                         table_positions[t][1] - ori_p[1],
                                         table_positions[t][1] -
                                         table_positions[t][0] + ori_p[0])
                                new_pixel = point_to_pixel(
                                    new_p, (0, 0), (10, 10), (160, 160))
                                if pixels[i, j] > 0:
                                    if new_pixel in pixel_feas_dict[t][
                                            obj_idx].keys():
                                        pixel_feas_dict[t][obj_idx][
                                            new_pixel] = (
                                                pixel_feas_dict[t][obj_idx]
                                                [new_pixel] + pixels[i, j]) / 2
                                    else:
                                        pixel_feas_dict[t][obj_idx][
                                            new_pixel] = pixels[i, j]
                        object_side_feas.append(
                            [bot_feas, top_feas, left_feas, right_feas])

                # **************************************************
                print(object_side_feas)
                # return
                feas_sample = 0
                for obj_idx in selected_seq:

                    robot_base_points.append(initial_object_positions[obj_idx])

                    max_feas = 0
                    robot_base_points.append([None, None])

                    this_object_feas = object_side_feas[obj_idx]
                    max_feas = max(this_object_feas)
                    side_index = this_object_feas.index(max_feas)
                    if side_index == 0:
                        robot_base_points[-1] = [0, -0.57]
                    elif side_index == 1:
                        robot_base_points[-1] = [0, 0.57]
                    elif side_index == 2:
                        robot_base_points[-1] = [-0.57, 0]
                    else:
                        robot_base_points[-1] = [0.57, 0]
                    """
                    for pixel, feas in pixel_feas_dict[0][obj_idx].items():
                        # TODO sample 1000 times
                        pixel_point = list(
                            pixel_to_point((pixel[0], pixel[1]), (0, 0), (10, 10),
                                           (160, 160)))
                        average_feas = 0
                        rv = multivariate_normal(
                            pixel_point, [[nav_variance, 0], [0, nav_variance]])
                        for s in range(max_sample):
                            sampled_pixel = point_to_pixel(rv.rvs(), (0, 0),
                                                           (10, 10), (160, 160))
                            # print(str(sampled_pixel) + str(pixel))
                            if sampled_pixel in pixel_feas_dict[0][obj_idx]:
                                average_feas += pixel_feas_dict[0][obj_idx][
                                    sampled_pixel]
                        average_feas = float(average_feas) / max_sample
                        if average_feas > max_feas:
                            max_feas = average_feas
                            robot_base_points[-1] = pixel_point
                    """
                    feas_sample += max_feas
                feas_sample = float(feas_sample) / (object_num)
                print("feasibility: " + str(feas_sample))
                if feas_sample > min_feas:
                    temp_robot_base_points = robot_base_points
                    min_feas = feas_sample
                    temp_object_positions = object_positions
            robot_base_points = temp_robot_base_points
            object_positions = temp_object_positions
            print("Best robot base points: " + str(robot_base_points))
            print("Selected object goal positions: " + str(object_positions))
            # print(max_feas)

            # ****************************************************************************
            # robot_base_points = [[[-1.625, -0.4375], [-2.75, -1.3125],
            #                       [-2.8125, -1.125]]]
            print("Start execution...")
            exe_start = time.time()
            is_loading = True
            for nav_goal_idx in range(len(robot_base_points)):
                nav_goal_list = [
                    robot_base_points[nav_goal_idx][0],
                    robot_base_points[nav_goal_idx][1]
                ]

                nav_goal = Point(nav_goal_list[0], nav_goal_list[1], 0)
                side = get_side(table_positions[0], nav_goal_list)

                is_there = False
                try_times = 0
                rv = multivariate_normal(
                    nav_goal_list, [[nav_variance, 0], [0, nav_variance]])

                while try_times < nav_try:
                    print("Navigation trying...")
                    is_there = go_to_unloading(nav, side, nav_goal)
                    if is_there:
                        break
                    else:
                        try_times += 1
                        nav_goal_list = rv.rvs()
                        nav_goal = Point(nav_goal_list[0], nav_goal_list[1], 0)
                        side = get_side(table_positions[0], nav_goal_list)

                if is_loading:
                    is_loading = False
                    continue
                else:
                    is_loading = True

                time.sleep(1)

                #step1: calculate absolute object (to be generated) pose

                temp_robot_pose = rospy.wait_for_message(
                    '/odom', Odometry).pose.pose.position
                temp_robot_orien = rospy.wait_for_message(
                    '/odom', Odometry).pose.pose.orientation
                rx = temp_robot_pose.x
                ry = temp_robot_pose.y

                theta = quat_to_yaw(temp_robot_orien.x, temp_robot_orien.y,
                                    temp_robot_orien.z, temp_robot_orien.w)

                dx = loaded_object_pose[0]
                dy = loaded_object_pose[1]

                temp_unloaded = [0, 0]
                temp_unloaded[0] = rx + dx * cos(theta) - dy * sin(theta)
                temp_unloaded[1] = ry + dy * cos(theta) + dx * sin(theta)
                temp_orien = temp_robot_orien

                #step2: add the object to the serving plate
                """
                if nav_goal_idx == 0:
                    model_sdf = "/home/bwilab/.gazebo/models_add/fork/model.sdf"
                elif nav_goal_idx == 1:
                    model_sdf = "/home/bwilab/.gazebo/models_add/knife/model.sdf"
                elif nav_goal_idx == 2:
                    model_sdf = "/home/bwilab/.gazebo/models_add/bowl/model.sdf"
                """
                object_name = objects[selected_seq[int(nav_goal_idx / 2)]]
                model_sdf = get_object_model(object_name)
                model_n = 'target_object_' + str(nav_goal_idx)
                test_spawner(model_name=model_n,
                             model_xml=open(model_sdf, 'r').read(),
                             robot_namespace="/object",
                             initial_pose=Pose(position=Point(
                                 temp_unloaded[0], temp_unloaded[1], 1.4),
                                               orientation=temp_orien),
                             reference_frame="world")

                unload_object(
                    ac, temp_unloaded,
                    object_positions[selected_seq[int(nav_goal_idx / 2)]],
                    object_name, model_n, test_remover, side)

            metrics = [time.time() - exe_start]
            trial_result = {}
            trial_result['execution_time'] = metrics
            trial_result['sequence'] = selected_seq
            trial_result['objects'] = objects
            trial_result['base_poses'] = robot_base_points
            trial_result['initial'] = initial_object_positions
            trial_result['goal'] = object_positions
            trial_result['chair_pose'] = chair_positions
            trial_result['chair_orien'] = chair_oriens
            if "grop" not in os.listdir("exp/"):
                os.mkdir("exp/grop/")
            with open("exp/grop/" + task_name + '_' + str(trial_idx) + '.json',
                      'w+') as f:
                json.dump(trial_result, f)
            nav.move_to_goal(init_pose, init_orien)
            old_high_camera_state = model_coordinates("distorted_camera_clone",
                                                      "")
            capture_config(
                old_high_camera_state,
                "exp/grop/" + task_name + '_' + str(trial_idx) + '.png')
            cs.delete_all()
            sim_cs.delete_all()

            for i in range(20):
                test_remover(model_name='target_object_' + str(i))


if __name__ == "__main__":
    main()
