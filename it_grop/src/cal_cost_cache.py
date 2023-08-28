#!/usr/bin/env python
import copy
import datetime
import itertools
import json
import os
import random
import shutil
import time
from collections import defaultdict
from itertools import combinations, product
from math import acos, atan2, cos, factorial, pi, sin, sqrt, tan
from multiprocessing import Manager, Process, Queue

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
from scipy.spatial import Delaunay
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

# parameters
task_num = 20

im_actual_size = [4, 4]
im_size = [64, 64]
chair_num = 6
object_num = 5
remove_prob = 0.2
generations = 20  #CMA-ES
total_sample_times = 200  # the total number of samples for each state
# population_size is equal to total_sample_times/generations
nav_variance = 0.03125  # 2D Gaussian sampling variance. Co-variance is set to 0
sample_times = 5  # Navigation 2D gaussian sampling times for modeling navigation uncertainty
max_cost = 999  # for navigation no plan found or sample to unsatisfied region
large_cost = 20  # for sampling near the objects receiving large cost
max_man = 1  # manipulation cost
max_reward = 100  # reward for task completion
robot_vel = 1  # robot velocity
starting_cost = 20  # constant cost for starting the base
timeout = 99999999999  # timeout for end the planning phase
D = 1  # robot basic reachability area. This is used for speeding up sampling
time_budget = 300
CPU_USE = 12
top_n = 10

# for globally storing table positions of the current scene
CUR_TABLE_POSITIONS = [[-3, 0.2], [-3, 1], [-1.7, 0.2], [-1.7, 1], [1.25, 0],
                       [1.75, 0], [2.25, 0], [2.75, 0], [1.25,
                                                         1.3], [1.75, 1.3],
                       [2, 25, 1.3], [2.75, 1.3], [-1.75, 3], [-1.25, 3],
                       [-0.75, 3], [-0.25, 3], [0.25, 3], [0.75, 3], [1.25, 3],
                       [1.75, 3]]
max_u = float('-inf')  # max utility for dfs baseline
max_p = None  # max plan for dfs baseline
# robot initial position and orientation
init_pose = Point(0, -3, 0)


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


init_orien = euler_to_quat(0, 0, 1.57)


def filter_voronoi_non_adj(points, state_candidates):
    tri = Delaunay(points)
    neiList = defaultdict(set)
    for p in tri.vertices:
        for i, j in itertools.combinations(p, 2):
            neiList[i].add(j)
            neiList[j].add(i)

    # for key in sorted(neiList.iterkeys()):
    # print("%d:%s" % (key, ','.join([str(i) for i in neiList[key]])))
    res = []
    for ss in state_candidates:
        is_valid = True
        for each_combine in ss:
            for each_s in each_combine:
                for temp_s in each_combine:
                    if temp_s != each_s and temp_s not in neiList[each_s]:
                        is_valid = False
        if is_valid:
            res.append(ss)
    return res


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
            loading_pos = Point(
                goal.x, goal.y - 0.06, 0
            )  ###localization issue, test1 for pushing the robot to the table
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


def unload_object(ac, object_pose, unloading_point):
    robot_pose = rospy.wait_for_message('/odom', Odometry).pose.pose.position
    robot_orien = rospy.wait_for_message('/odom',
                                         Odometry).pose.pose.orientation

    ac.move_to_ee_pose(robot_pose, robot_orien,
                       Point(object_pose[0], object_pose[1], 1.35))
    ac.open_gripper()
    ac.lift(1.3)
    ac.close_gripper()
    ac.lift(1.35)
    ac.move_to_ee_pose(robot_pose, robot_orien,
                       Point(unloading_point[0], unloading_point[1], 1.3))
    return
    ac.open_gripper()
    #ac.lift(1.32)
    #ac.go_to_init_joints()


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

    orien_bot = filename.split('_')[4][1]
    orien_top = filename.split('_')[4][0]

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


def combine_state_space(state, com):
    new_state = copy.deepcopy(state)
    for c in com:
        for s in c[1:]:
            for p in state[s]:
                new_state[c[0]].append(p)
        for s in c[1:]:
            new_state[s] = new_state[c[0]]
    return new_state


def dfs(object_states, l, object_positions, state_point, feas_dict, nav, p,
        start_time):
    #TODO search repeat!
    if time.time() - start_time >= timeout:
        return
    global max_u
    if len(p) == len(l) + 1:
        # object_sequence = []
        # for point in p[1:]:
        #     object_sequence.append(pixel_state[point])

        plan_utility, total_nav, m_reward = get_tm_plan_utility(
            p, l, feas_dict, nav)
        # print ("Evaluating plan: " + str(p))
        if plan_utility > max_u:
            global max_p
            max_u = plan_utility
            max_p = p
            print(
                "=========>>>>>>>>> A better plan has been found. The plan utility and cost is "
                + str(max_u) + "," + str(total_nav))
        return
    else:

        # let objects that are in the same state have the same sampled standing position
        if len(object_positions) > 1 and p > 1:
            if object_states[l[len(p) - 1]] == object_states[l[len(p) - 1]]:
                new_p = copy.deepcopy(p)
                new_p.append(p[-1])
                dfs(object_states, l, object_positions, state_point, feas_dict,
                    nav, new_p, start_time)

        for next_point in state_point[l[len(p) - 1]]:
            # to speed up sampling, only sample points that are pretty close to the object
            pixel_check = point_to_pixel(next_point, (0, 0), (10, 10),
                                         (160, 160))
            pixel_key = (pixel_check[0], pixel_check[1])
            action_state = l[len(p) - 1]
            if dist(next_point, object_positions[action_state]) > 1:
                continue
            else:
                new_p = copy.deepcopy(p)
                new_p.append(next_point)
                dfs(object_states, l, object_positions, state_point, feas_dict,
                    nav, new_p, start_time)


def state_addition(combined, state_pixel):
    added_state_pixel = copy.deepcopy(state_pixel)
    search_states = state_pixel.keys()
    for ss in combined:
        added_index = added_state_pixel.keys()[-1] + 1
        search_states.append(added_index)
        added_state_pixel[added_index] = []
        for s in ss:
            search_states.remove(s)
            for p in state_pixel[s]:
                if p not in added_state_pixel[added_index]:
                    added_state_pixel[added_index].append(p)
    return added_state_pixel, search_states


def cmaes_new(combined, seq, object_positions, added_state_pixel, feas_dict,
              nav, robot_initial_position, log_f):
    # object sequences is of the same length as object_positions, and contain non-repeated object indexs
    # object_states may contain repeated index due to state merging
    start_time = time.time()
    object_num = len(object_positions)
    print(">>>>>>>>>>>>>>>>  Search for state space: " + str(combined) +
          " with sequence: " + str(seq))

    bounds = [[], []]

    cmaes_init = []
    # to speed up cames, only consider standing points that are within 1 meter of the object
    for s in seq:

        if s < object_num:  # uncombined states
            bounds[0].append(object_positions[s][0] - D)
            bounds[0].append(object_positions[s][1] - D)

            bounds[1].append(object_positions[s][0] + D)
            bounds[1].append(object_positions[s][1] + D)
            cmaes_init.append(object_positions[s][0])
            cmaes_init.append(object_positions[s][1])
        else:
            objects = combined[s - object_num]
            min_x = float('inf')
            min_y = float('inf')
            max_x = float('-inf')
            max_y = float('-inf')
            for o in objects:
                min_x = min(min_x, object_positions[o][0])
                min_y = min(min_y, object_positions[o][1])
                max_x = max(max_x, object_positions[o][0])
                max_y = max(max_y, object_positions[o][1])

            bounds[0].append(min_x - D)
            bounds[0].append(min_y - D)

            bounds[1].append(max_x + D)
            bounds[1].append(max_y + D)

            cmaes_init.append(object_positions[o][0])
            cmaes_init.append(object_positions[o][1])
    # population_size = optimizer.popsize
    population_size = total_sample_times / generations
    optimizer = cma.CMAEvolutionStrategy(cmaes_init, 0.5, {
        'bounds': bounds,
        'popsize': population_size
    })
    random_combined_dict = {}
    all_points = []
    all_scores = []
    all_costs = []
    all_rates = []
    for g in range(generations):
        # print ("CMA-ES generation: "+str(g))
        pop_point = optimizer.ask()

        scores = []
        for i in range(len(pop_point)):
            point = pop_point[i]
            all_points.append(point)
            standing_positions = []

            is_valid = True
            for i in range(len(seq)):

                standing_positions.append([point[i * 2], point[i * 2 + 1]])
                # early feedback
                one_point = [point[i * 2], point[i * 2 + 1]]
                one_pixel = point_to_pixel(one_point, (0, 0), (10, 10),
                                           (160, 160))
                if one_pixel not in added_state_pixel[seq[i]]:
                    is_valid = False
                for t_p in CUR_TABLE_POSITIONS:
                    if t_p[0] + 0.2 >= one_point[0] >= t_p[0] - 0.2 and t_p[
                            1] + 0.2 >= one_point[1] >= t_p[1] - 0.2:
                        is_valid = False
            if not is_valid:
                score = max_cost
                cost = max_cost
                rate = 0
            else:
                standing_positions.insert(0, robot_initial_position)
                score, cost, rate = cmaes_objective(combined, seq,
                                                    standing_positions,
                                                    feas_dict, nav,
                                                    added_state_pixel)

            scores.append(score)
            all_scores.append(score)
            all_costs.append(cost)
            all_rates.append(rate)

        optimizer.tell(pop_point, scores)

        # visualize cmaes
        im = IM.new('RGB', (160, 160))
        pixels = im.load()
        for p in pop_point:
            for i in range(len(seq)):
                pixel_p = point_to_pixel((p[2 * i], p[2 * i + 1]), (0, 0),
                                         (10, 10), (160, 160))
                pixels[pixel_p[0], pixel_p[1]] = (24, 96, 72)

        max_index_gen = scores.index(min(scores))
        for i in range(len(seq)):
            pixel_p = point_to_pixel((pop_point[max_index_gen][2 * i],
                                      pop_point[max_index_gen][2 * i + 1]),
                                     (0, 0), (10, 10), (160, 160))
            pixels[pixel_p[0], pixel_p[1]] = (48, 255, 96)

        #testing purpose: resize
        big_im = IM.new('RGB', (640, 640))
        big_pixels = big_im.load()
        for i in range(0, 640):
            for j in range(0, 640):
                big_pixels[i, j] = pixels[i / 4, j / 4]
        big_im = big_im.convert("RGBA")

        # blend
        im_b = IM.open("log/envs/trial_0.jpg")
        b_pixels = im_b.load()
        large_b = IM.new('RGB', (640, 640))
        large_pixels = large_b.load()
        for i in range(0, 640):
            for j in range(0, 640):
                large_pixels[i, j] = b_pixels[i / 10, j / 10]
        large_b = large_b.convert("RGBA")
        blended = IM.blend(big_im, large_b, 0.5)

        blended.save(log_f + "gen_" + str(g) + ".png")

    ret_standing_positions = []
    max_value = min(all_scores)
    max_index = all_scores.index(max_value)
    gen_index = (max_index % total_sample_times) / population_size
    for i in range(len(seq)):
        ret_standing_positions.append(
            [all_points[max_index][i * 2], all_points[max_index][i * 2 + 1]])
    ret_standing_positions.insert(0, robot_initial_position)
    res = {}
    res['combined'] = combined
    res['seq'] = seq
    res['log_f'] = log_f
    res['utility'] = (-1) * max_value
    res['success'] = all_rates[max_index]
    res['cost'] = all_costs[max_index]
    res['gen'] = gen_index
    res['positions'] = ret_standing_positions
    with open(log_f + "results.txt", 'a') as f:
        f.write(json.dumps(res))

    print("\nBatch results for log dir: " + log_f)
    print("max utility: " + str((-1) * max_value))
    print("expected completion rate: " + str(all_rates[max_index]))
    print("corresponding cost: " + str(all_costs[max_index]))
    print("best generation: " + str(gen_index) + "\n")
    return ret_standing_positions, all_costs[max_index]


def cmaes_old(combined, object_positions, state_pixel, feas_dict, nav,
              robot_initial_position, start_time, batch, batch_size,
              return_dict, run):
    # object sequences is of the same length as object_positions, and contain non-repeated object indexs
    # object_states may contain repeated index due to state merging

    object_num = len(object_positions)
    added_state_pixel, search_states = state_addition(combined, state_pixel)
    # evaluate cost for all task-level plan and choose the best
    a_sequence = copy.deepcopy(search_states)
    all_seq = []
    for seq in list(product(a_sequence, repeat=len(a_sequence))):
        if len(list(seq)) != len(set(seq)):
            continue
        seq = list(seq)
        all_seq.append(seq)
    all_points = []
    all_scores = []
    all_costs = []
    starting_seq = batch * batch_size
    end_seq = (batch + 1) * batch_size
    all_seq = all_seq[starting_seq:end_seq]
    for seq in all_seq:

        time_escape = time.time() - start_time
        if time_escape >= timeout:
            break
        seq_all = []
        print(">>>>>>>>>>>>>>>>  Search for: " + str(seq))

        bounds = [[], []]

        cmaes_init = []
        # to speed up cames, only consider standing points that are within 1 meter of the object
        for s in seq:

            if s < object_num:  # uncombined states
                bounds[0].append(object_positions[s][0] - D)
                bounds[0].append(object_positions[s][1] - D)

                bounds[1].append(object_positions[s][0] + D)
                bounds[1].append(object_positions[s][1] + D)
                cmaes_init.append(object_positions[s][0])
                cmaes_init.append(object_positions[s][1])
            else:
                objects = combined[s - object_num]
                min_x = float('inf')
                min_y = float('inf')
                max_x = float('-inf')
                max_y = float('-inf')
                for o in objects:
                    min_x = min(min_x, object_positions[o][0])
                    min_y = min(min_y, object_positions[o][1])
                    max_x = max(max_x, object_positions[o][0])
                    max_y = max(max_y, object_positions[o][1])

                bounds[0].append(min_x - D)
                bounds[0].append(min_y - D)

                bounds[1].append(max_x + D)
                bounds[1].append(max_y + D)

                cmaes_init.append(object_positions[o][0])
                cmaes_init.append(object_positions[o][1])
        # population_size = optimizer.popsize
        population_size = total_sample_times / generations
        optimizer = cma.CMAEvolutionStrategy(cmaes_init, 0.5, {
            'bounds': bounds,
            'popsize': population_size
        })
        # points = np.ndarray((generations, population_size, 2*object_num))
        # for g in range(generations):
        # while not optimizer.stop():
        # solutions = []
        # for i in range(optimizer.popsize):
        # pop_point = optimizer.ask(number=population_size)

        random_combined_dict = {}

        for g in range(generations):
            # print ("CMA-ES generation: "+str(g))

            pop_point = optimizer.ask()

            scores = []
            for i in range(len(pop_point)):
                point = pop_point[i]
                all_points.append(point)
                standing_positions = []

                is_valid = True
                for i in range(len(seq)):

                    standing_positions.append([point[i * 2], point[i * 2 + 1]])
                    # early feedback
                    one_point = [point[i * 2], point[i * 2 + 1]]
                    one_pixel = point_to_pixel(one_point, (0, 0), (10, 10),
                                               (160, 160))
                    if one_pixel not in added_state_pixel[seq[i]]:
                        is_valid = False
                    for t_p in CUR_TABLE_POSITIONS:
                        if t_p[0] + 0.2 >= one_point[0] >= t_p[0] - 0.2 and t_p[
                                1] + 0.2 >= one_point[1] >= t_p[1] - 0.2:
                            is_valid = False
                if not is_valid:
                    score = max_cost
                    cost = max_cost
                else:
                    standing_positions.insert(0, robot_initial_position)

                    if not run.is_set():
                        print(
                            "Kill this process because better state space is found"
                        )
                        return

                    if random.uniform(0, 1) <= 1.01**g - 1:
                        removed_index = random.randint(1, len(seq) - 1)
                        random_combined = [[
                            seq[removed_index - 1], seq[removed_index]
                        ]]

                        if str(random_combined
                               ) not in random_combined_dict.keys():
                            random_combined_dict[str(random_combined)] = {}

                            remove_seq = copy.deepcopy(seq)
                            remove_seq.append(remove_seq[-1] + 1)
                            remove_seq.pop(removed_index)
                            remove_seq.pop(removed_index - 1)
                            constructed_state_pixel, constructed_search_states = state_addition(
                                random_combined, added_state_pixel)
                            random_combined_dict[str(
                                random_combined)]["remove_seq"] = remove_seq
                            random_combined_dict[str(random_combined)][
                                "state_pixel"] = constructed_state_pixel
                        else:
                            remove_seq = random_combined_dict[str(
                                random_combined)]["remove_seq"]
                            remove_constructed_state_pixel = random_combined_dict[
                                str(random_combined)]["state_pixel"]

                        remove_standing_positions = copy.deepcopy(
                            standing_positions)
                        remove_standing_positions.pop(removed_index + 1)

                        score, cost = cmaes_objective(
                            random_combined, remove_seq,
                            remove_standing_positions, feas_dict, nav,
                            constructed_state_pixel)
                        if score > max(all_scores):

                            print("Better state space found")
                            run.clear()
                            return_dict[batch] = random_combined
                            return
                    else:
                        score, cost = cmaes_objective(combined, seq,
                                                      standing_positions,
                                                      feas_dict, nav,
                                                      added_state_pixel)

                scores.append(score)
                all_scores.append(score)
                seq_all.append(score)
                all_costs.append(cost)
            # solutions.append((point,score))
            optimizer.tell(pop_point, scores)

            # visualize cmaes
            im = IM.new('RGB', (160, 160))
            pixels = im.load()
            for p in pop_point:
                for i in range(len(seq)):
                    pixel_p = point_to_pixel((p[2 * i], p[2 * i + 1]), (0, 0),
                                             (10, 10), (160, 160))
                    pixels[pixel_p[0], pixel_p[1]] = (24, 96, 72)

            max_index_gen = scores.index(min(scores))
            for i in range(len(seq)):
                pixel_p = point_to_pixel((pop_point[max_index_gen][2 * i],
                                          pop_point[max_index_gen][2 * i + 1]),
                                         (0, 0), (10, 10), (160, 160))
                pixels[pixel_p[0], pixel_p[1]] = (48, 255, 96)

            #testing purpose: resize
            big_im = IM.new('RGB', (640, 640))
            big_pixels = big_im.load()
            for i in range(0, 640):
                for j in range(0, 640):
                    big_pixels[i, j] = pixels[i / 4, j / 4]
            big_im = big_im.convert("RGBA")

            # blend
            im_b = IM.open("log/envs/trial_0.jpg")
            b_pixels = im_b.load()
            large_b = IM.new('RGB', (640, 640))
            large_pixels = large_b.load()
            for i in range(0, 640):
                for j in range(0, 640):
                    large_pixels[i, j] = b_pixels[i / 10, j / 10]
            large_b = large_b.convert("RGBA")
            blended = IM.blend(big_im, large_b, 0.5)

            blended.save("cmaes_log/seq_" +
                         str(all_seq.index(seq) + starting_seq) + "_gen_" +
                         str(g) + ".png")


#             im_b_d = IM.open("cmaes_log/seq_"+str(all_seq.index(seq))+"_gen_"+str(g)+".png")
#             im_v = IM.open("voronoi.png")
#             pixel_v = im_v.load()
#             large_v = IM.new("RGB", (640,640))
#             large_pixels = large_v.load()
#             for i in range(0, 640):
#                 for j in range(0, 640):
#                     large_pixels[i,j] = pixel_v[i/4, j/4]
#             large_v = large_v.convert("RGBA")
#             blended = IM.blend(im_b_d, large_v, 0.2)
#
#             blended.save("cmaes_log/seq_"+str(all_seq.index(seq))+"_gen_"+str(g)+".png")
#

# print("seq best: " + str((-1) * min(seq_all)))
    ret_standing_positions = []
    max_value = min(all_scores)
    max_index = all_scores.index(max_value)
    seq_index = max_index / total_sample_times
    gen_index = (max_index % total_sample_times) / population_size
    for i in range(len(search_states)):
        ret_standing_positions.append(
            [all_points[max_index][i * 2], all_points[max_index][i * 2 + 1]])
    ret_standing_positions.insert(0, robot_initial_position)
    print("\nbatch: " + str(batch))
    print("max_utility: " + str((-1) * max_value))
    print("corresponding_cost: " + str(all_costs[max_index]))
    print("seq: " + str(seq_index + starting_seq) + " | gen: " +
          str(gen_index) + "\n")
    return ret_standing_positions, all_costs[max_index]


def cmaes_objective(combined, seq, standing_positions, feas_dict, nav,
                    state_pixel):
    object_num = len(feas_dict.keys())

    #     for s in range(0, len(standing_positions[1:])):
    #         s_pixel = point_to_pixel(standing_positions[s+1],(0,0),(10,10),(160,160))
    #         o_s = state_pixel[seq[s]]
    #         # import pdb; pdb.set_trace()
    #         if s_pixel not in o_s:
    #             return max_cost, max_cost

    #p_u, p_c = get_tm_plan_utility(standing_positions, object_sequences, feas_dict, nav)
    #return (-1)*p_u, p_c
    object_sequences = []
    for i in seq:
        if i < object_num:
            object_sequences.append(i)
        else:
            for each_o in combined[i - object_num]:
                object_sequences.append(each_o)

    repeated_standing_positions = [standing_positions[0]]
    k = 0
    prev = -1
    for i in range(len(object_sequences)):
        k += 1
        for ss in combined:
            if object_sequences[i] in ss and prev in ss:
                k -= 1
        repeated_standing_positions.append(standing_positions[k])
        prev = object_sequences[i]

    p_u, p_c, m_reward = get_tm_plan_utility(combined,
                                             repeated_standing_positions,
                                             object_sequences, feas_dict, nav,
                                             state_pixel)
    # total_utility.append(p_u)
    # total_cost.append(p_c)
    # m_u = max(total_utility)
    # m_c = total_cost[total_utility.index(m_u)]
    return (-1) * p_u, p_c, float(m_reward) / (float(max_reward) * object_num)


def cost_nav(nav, robot_pose, p_goal):
    # continuous computation
    # import pdb; pdb.set_trace()
    if not nav.make_plan(Point(robot_pose[0], robot_pose[1], 0),
                         Point(p_goal[0], p_goal[1], 0)):
        return max_cost

    rv = multivariate_normal(p_goal, [[nav_variance, 0], [0, nav_variance]])
    total_sum = 0
    for n in range(1, sample_times):
        sampled = pixel_to_point(
            point_to_pixel(rv.rvs(), (0, 0), (10, 10), (160, 160)), (0, 0),
            (10, 10), (160, 160))
        ret = nav.make_plan(Point(robot_pose[0], robot_pose[1], 0),
                            Point(sampled[0], sampled[1], 0))
        # ret = nav.get_cost_from_cache(robot_pose, sampled)
        if ret and ret > 0:  # move distance larger than 1 pixel
            total_sum += starting_cost + ret / robot_vel
        elif ret and ret == 0:
            total_sum += 0
        else:
            total_sum += starting_cost + large_cost / robot_vel
        # elif ret and ret == 0:
        #     total_sum += 0
    return total_sum / (sample_times - 1)


def cost_man():
    return max_man


def reward(combined, robot_standing_points, object_sequence, feas_dict,
           state_pixel_dict):
    # len(robot_standing_points) should be the same as object num,
    # even for standing at the same position to pick up multiple objects
    # discrete computation

    object_state_dict = {}
    object_num = len(robot_standing_points)
    for i in range(object_num):
        object_state_dict[i] = i
        for ss_index in range(len(combined)):
            if i in combined[ss_index]:
                object_state_dict[i] = object_num + ss_index

    #TODO: whether to use production or not
    total_prod = 0
    for i in range(0, len(robot_standing_points)):
        rv = multivariate_normal(robot_standing_points[i],
                                 [[nav_variance, 0], [0, nav_variance]])
        object_idx = object_sequence[i]
        total_sum = 0
        for n in range(1, sample_times):
            # to speed up sampling, only sample on the points that feasibility is larger than 0
            sampled = point_to_pixel(rv.rvs(), (0, 0), (10, 10), (160, 160))
            # if sampled not in feas_dict[i] or feas_dict[i][sampled] < feas_threshold:
            #    n -= 1
            #    continue
            if sampled not in feas_dict[object_idx].keys():
                total_sum += 0
            else:
                total_sum += float(feas_dict[object_idx][sampled]) / float(255)
        e_p = total_sum / (sample_times - 1)
        # print (e_p)
        total_prod = total_prod + e_p
        # print (total_prod)
        # print (total_prod)
    return max_reward * total_prod


def get_tm_plan_utility(combined, robot_base_points, object_sequence,
                        feas_dict, nav, state_pixel):
    robot_standing_points = robot_base_points[1:]
    total_man = 0
    total_nav = 0
    for i in range(1, len(robot_base_points)):
        total_nav += cost_nav(nav, robot_base_points[i - 1],
                              robot_base_points[i])
    for i in range(len(object_sequence)):
        total_man += cost_man()
    m_reward = reward(combined, robot_standing_points, object_sequence,
                      feas_dict, state_pixel)
    utility = -total_man - total_nav + m_reward
    # utility = -total_man-total_nav
    # print ("cost_nav: "+str(total_nav)+" | reward: " + str(m_reward)+" | utility: "+str(utility))
    return utility, total_nav, m_reward


def get_cpu_count(table_num):
    if table_num == 1:
        cpu_count = 1
    elif table_num == 2:
        cpu_count = 2
    elif table_num == 3:
        cpu_count = 6
    else:
        cpu_count = 12
    return cpu_count


def get_combined_states(num, timeout):
    if num == 1:
        return [[]]
    start_time = time.time()
    ret = []
    ret.append([])
    all_state = [[]]
    for i in range(num):
        all_state[0].append(i)
    ret.append(all_state)

    while True:
        if time.time() - start_time > timeout:
            candidates = [[]]
            for ele in ret:
                candidate = []
                for ee in ele:
                    if len(ee) > 1:
                        candidate.append(ee)
                if candidate:
                    candidates.append(candidate)
            return candidates
        combined = []
        random.shuffle(all_state[0])
        remain_state = len(all_state[0])
        prev = 0
        while remain_state > 0:
            sample_num = random.randint(1, num - 1)
            last_remain = remain_state
            remain_state -= sample_num
            if remain_state <= 0:
                sample_num = last_remain
            combined_each = []
            for i in range(prev, prev + sample_num):
                combined_each.append(all_state[0][i])
            combined.append(combined_each)
            prev += sample_num
        for l in combined:
            l.sort()
        combined.sort()
        if combined not in ret:
            ret.append(combined)


def distribute(k, timestamp, state_candidates, state_pixel_dict,
               object_positions, pixel_feas_dict, nav, robot_initial_point):
    print("preparing batch: " + str(k))
    log_f = "cmaes_log/" + timestamp + '/' + 'cpu_' + str(k) + '/'
    os.mkdir(log_f)

    batch_size = time_budget / 30
    batch_processes = []
    idx = 0
    while idx < batch_size:
        # temp_start = time.time()
        # TODO: check adjacency
        while True:
            shuffle_again = False
            random.shuffle(state_candidates)
            for sc in state_candidates[0]:
                if 0 in sc:
                    shuffle_again = True
            if not shuffle_again:
                break
        combined = state_candidates[0]
        added_state_pixel, search_states = state_addition(
            combined, state_pixel_dict)
        a_sequence = copy.deepcopy(search_states)

        all_seq = []
        a_list = list(product(a_sequence, repeat=len(a_sequence)))
        random.shuffle(a_list)
        for seq in a_list:
            if len(list(seq)) != len(set(seq)):
                continue
            seq = list(seq)
            break

        # cpu_count = get_cpu_count(states_num)
        log_ff = log_f + str(idx) + '/'
        p = Process(target=cmaes_new,
                    args=(combined, seq, object_positions, added_state_pixel,
                          pixel_feas_dict, nav, robot_initial_point, log_ff))
        batch_processes.append(p)
        # visited.add(v_couple)
        # p.start()
        idx += 1
    start_time = time.time()
    for p_id in range(len(batch_processes)):
        if time.time() - start_time > time_budget:
            break
        print("batch: " + str(k) + " starts process number: " + str(p_id))
        os.mkdir(log_f + str(p_id))
        batch_processes[p_id].start()
        batch_processes[p_id].join()
    print("batch: " + str(k) + " returns")


def weighted_distribute(k, timestamp, state_candidates, state_pixel_dict,
                        object_positions, pixel_feas_dict, nav,
                        robot_initial_point, weights):
    print("preparing batch: " + str(k))
    log_f = "cmaes_log/" + timestamp + '/' + 'cpu_' + str(k) + '/'
    os.mkdir(log_f)

    batch_size = time_budget / 30
    batch_processes = []
    idx = 0
    prefix_sum = 0
    prefix_sums = []
    for w in weights:
        prefix_sum += w
        prefix_sums.append(prefix_sum)
    total_sum = prefix_sum
    while idx < batch_size:
        # temp_start = time.time()
        target = random.random() * total_sum
        for i, prefix_sum in enumerate(prefix_sums):
            if target < prefix_sum:
                break
        combined = state_candidates[i]
        added_state_pixel, search_states = state_addition(
            combined, state_pixel_dict)
        a_sequence = copy.deepcopy(search_states)

        all_seq = []
        a_list = list(product(a_sequence, repeat=len(a_sequence)))
        random.shuffle(a_list)
        for seq in a_list:
            if len(list(seq)) != len(set(seq)):
                continue
            seq = list(seq)
            break

        # cpu_count = get_cpu_count(states_num)
        log_ff = log_f + str(idx) + '/'
        p = Process(target=cmaes_new,
                    args=(combined, seq, object_positions, added_state_pixel,
                          pixel_feas_dict, nav, robot_initial_point, log_ff))
        batch_processes.append(p)
        # visited.add(v_couple)
        # p.start()
        idx += 1
    start_time = time.time()
    for p_id in range(len(batch_processes)):
        if time.time() - start_time > time_budget:
            break
        print("batch: " + str(k) + " starts process number: " + str(p_id))
        os.mkdir(log_f + str(p_id))
        batch_processes[p_id].start()
        batch_processes[p_id].join()
    print("batch: " + str(k) + " returns")


def main():

    rospy.init_node('run', anonymous=False)

    if not os.path.isdir('log/images'):
        os.mkdir("log/images")
    if not os.path.isdir('log/results'):
        os.mkdir("log/results")
    if os.path.isdir("log/temp_image"):
        shutil.rmtree("log/temp_image")
        shutil.rmtree("log/temp_result")

    #minimal testing
    """
    # test_1_object
    table_positions = [[-2.32, 2.08]]
    chair_positions = []
    chair_oriens = []
    object_positions = [[-2.40, 2.02]]


    # test_3_object
    table_positions = [[-2.32, 2.08], [-0.72, 1.22], [-0.21, 1.89]]
    chair_positions = [[-0.54, 2.58]]
    chair_oriens = [0]
    object_positions = [[-2.40, 2.02], [-0.61, 1.40], [-0.30, 1.72]]

    # test_5_object
    table_positions = [[-2.32, 2.08], [-0.72, 1.22], [-0.21, 1.89],
                       [0.86, 0.29], [1.40, 0.36]]
    chair_positions = [[-0.54, 2.58], [0.24, 0.67]]
    chair_oriens = [0, -0.3]
    object_positions = [[-2.40, 2.02], [-0.61, 1.34], [-0.25, 1.72],
                        [1.02, 0.25], [1.22, 0.45]]

    # test_7_object
    table_positions = [[-2.32, 2.08], [-0.72, 1.22], [-0.21, 1.89],
                       [0.86, 0.29], [1.40, 0.36], [-2.16, -0.95],
                       [-1.06, -0.55]]
    chair_positions = [[-0.54, 2.58], [0.24, 0.67], [-1.59, -1.08]]
    chair_oriens = [0, -0.3, 0.3]
    object_positions = [[-2.40, 2.02], [-0.61, 1.34], [-0.25, 1.72],
                        [1.02, 0.25], [1.22, 0.45], [-2.15, -0.93],
                        [-1.00, -0.65]]
    """
    # table_num = len(table_positions)
    global chair_num
    table_positions = CUR_TABLE_POSITIONS
    cp = camera_processor('/top_down_cam/image_raw')
    cs = scene_sampler(chair_num, len(CUR_TABLE_POSITIONS))
    sim_cs = scene_sampler(chair_num, 1)
    hm = heatmap()
    ac = arm_client()
    ac.open_gripper()
    ac.go_to_init_joints()
    vo = voronoi(160, 160, 10, 10, [0, 0])
    #vo.generate_voronoi(object_positions, [[1, 2], [3, 4]])
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state',
                                           GetModelState)
    old_camera_state = model_coordinates("distorted_camera", "")
    test_spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
    test_remover = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)

    #define some counters
    total_success_trial_num = 0
    overall_results = []
    #=========================================
    for task_idx in range(task_num):

        nav = navigator(init_pose, init_orien)

        navigatable_points = set()
        navigatable_points.add(
            point_to_pixel((0, -3), (0, 0), (10, 10), (160, 160)))

        # single
        for px in range(16, 68):
            for py in range(48, 96):
                add = True
                for tp in table_positions:
                    pxy = pixel_to_point((px, py), (0, 0), (10, 10),
                                         (160, 160))
                    if dist(pxy, tp) <= 0.25:
                        add = False
                if add and (px, py):
                    navigatable_points.add((px, py))

        # double
        for px in range(84, 144):
            for py in range(40, 96):
                add = True
                for tp in table_positions:
                    pxy = pixel_to_point((px, py), (0, 0), (10, 10),
                                         (160, 160))
                    if dist(pxy, tp) <= 0.25:
                        add = False
                if add:
                    navigatable_points.add((px, py))
        # bar
        for px in range(32, 128):
            for py in range(16, 48):
                add = True
                for tp in table_positions:
                    pxy = pixel_to_point((px, py), (0, 0), (10, 10),
                                         (160, 160))
                    if dist(pxy, tp) <= 0.25:
                        add = False
                if add:
                    navigatable_points.add((px, py))
        print(len(navigatable_points))
        navigatable_points = list(navigatable_points)
        cache_cost = {}
        with open('cost_cache.json', 'r') as outfile:
            cache_cost = json.load(outfile)
        for i, nav_1 in enumerate(navigatable_points):
            if str(nav_1) not in cache_cost:
                cache_cost[str(nav_1)] = {}
                for j, nav_2 in enumerate(navigatable_points[i:]):
                    nav_1_p = pixel_to_point(nav_1, (0, 0), (10, 10),
                                             (160, 160))
                    nav_2_p = pixel_to_point(nav_2, (0, 0), (10, 10),
                                             (160, 160))
                    pair_cost = nav.make_plan(Point(nav_1_p[0], nav_1_p[1], 0),
                                              Point(nav_2_p[0], nav_2_p[1], 0))
                    cache_cost[str(nav_1)][str(nav_2)] = pair_cost
                print(i)
                st = time.time()
                with open('cost_cache.txt', 'w+') as outfile:
                    json.dump(cache_cost, outfile)
                print("save json file takes: ", time.time() - st)
        return

        #randomly generate tables and chairs for each scene
        print("Generating scene id: " + str(task_idx + 1))
        # cs.spawn_chair_no_sample(chair_positions, chair_oriens)
        cs.set_positions_table(CUR_TABLE_POSITIONS)
        cs.spawn_chair()
        chair_positions = cs.get_positions_chair()
        chair_oriens = cs.get_oriens_chair()
        # cs.spawn_object_no_sample(object_positions)
        cs.spawn_object(object_num)
        object_positions = cs.get_positions_object()
        capture_env(old_camera_state, (0, 0),
                    "log/envs/task_" + str(task_idx) + '.jpg')

        # construct similar env next to the original one for getting top-down observations
        sim_chair_positions = []
        for p in chair_positions:
            sim_chair_positions.append(sim_point(p))
        sim_table_positions = []
        for p in table_positions:
            sim_table_positions.append(sim_point(p))
        sim_object_positions = []
        for p in object_positions:
            sim_object_positions.append(sim_point(p))
        sim_cs.spawn_chair_no_sample(sim_chair_positions, chair_oriens)
        sim_table_state = model_coordinates("table_cube_3", "")
        object_table_map = {}
        for t in range(object_num):
            # determine which table the object is on
            for t_idx, tp in enumerate(table_positions):
                if dist(tp, object_positions[t]) <= 0.23:
                    object_table_map[t] = t_idx
                    break

            #vertical
            new_table_state = ModelState()
            new_table_state.model_name = "table_cube_3"
            new_table_state.pose.orientation = euler_to_quat(0, 0, 1.57)
            new_table_state.pose.position = Point(
                sim_table_positions[t_idx][0], sim_table_positions[t_idx][1],
                sim_table_state.pose.position.z)
            set_state(new_table_state)
            image_filename = "log/images/task_" + str(
                task_idx) + "_table_" + str(t) + "_rl" + '.jpg'
            capture_image(
                old_camera_state,
                (sim_object_positions[t][0], sim_object_positions[t][1]),
                image_filename, True)
            predict(image_filename)

            #horizontal
            new_table_state.pose.orientation = euler_to_quat(0, 0, 0)
            set_state(new_table_state)
            image_filename = "log/images/task_" + str(
                task_idx) + "_table_" + str(t) + "_tb" + '.jpg'
            capture_image(
                old_camera_state,
                (sim_object_positions[t][0], sim_object_positions[t][1]),
                image_filename, False)
            predict(image_filename)

        #get voronoi graph   return -> key: pixel point -> value: 0-num_of_table
        pixel_state_dict = vo.generate_voronoi(object_positions, None)

        point_state_dict = {}
        for k in pixel_state_dict.keys():
            point_state_dict[pixel_to_point(k, (0, 0), (10, 10),
                                            (160, 160))] = pixel_state_dict[k]

        state_point_dict = {}
        for t in range(object_num):
            state_point_dict[t] = []
        for k in point_state_dict.keys():
            state_point_dict[point_state_dict[k]].append(k)

        state_pixel_dict = {}
        for k, v in state_point_dict.items():
            state_pixel_dict[k] = []
            for p in v:
                state_pixel_dict[k].append(
                    point_to_pixel(p, (0, 0), (10, 10), (160, 160)))

        pixel_feas_dict = {}
        for t in range(object_num):
            pixel_feas_dict[t] = {}
            l_image_filename = "log/results/l_task_" + str(
                task_idx) + "_object_" + str(t) + "_rl" + '.jpg'
            r_image_filename = "log/results/r_task_" + str(
                task_idx) + "_object_" + str(t) + "_rl" + '.jpg'
            t_image_filename = "log/results/t_task_" + str(
                task_idx) + "_object_" + str(t) + "_tb" + '.jpg'
            b_image_filename = "log/results/b_task_" + str(
                task_idx) + "_object_" + str(t) + "_tb" + '.jpg'

            #bottom image
            b_im = IM.open(b_image_filename)
            im_w, im_h = b_im.size
            pixels = b_im.load()
            for i in range(0, im_w):
                for j in range(0, im_h):
                    ori_p = pixel_to_point(
                        (i, j + im_h), table_positions[object_table_map[t]],
                        im_actual_size, im_size)
                    new_p = (ori_p[0], ori_p[1])
                    new_pixel = point_to_pixel(new_p, (0, 0), (10, 10),
                                               (160, 160))
                    if pixels[i, j] > 0:
                        if new_pixel in pixel_feas_dict[t].keys():
                            pixel_feas_dict[t][new_pixel] = (
                                pixel_feas_dict[t][new_pixel] +
                                pixels[i, j]) / 2
                        else:
                            pixel_feas_dict[t][new_pixel] = pixels[i, j]

            #top image
            t_im = IM.open(t_image_filename)
            im_w, im_h = t_im.size
            pixels = t_im.load()
            for i in range(0, im_w):
                for j in range(0, im_h):
                    ori_p = pixel_to_point(
                        (i, j + im_h), table_positions[object_table_map[t]],
                        im_actual_size, im_size)
                    new_p = (2 * table_positions[object_table_map[t]][0] -
                             ori_p[0],
                             2 * table_positions[object_table_map[t]][1] -
                             ori_p[1])
                    new_pixel = point_to_pixel(new_p, (0, 0), (10, 10),
                                               (160, 160))
                    if pixels[i, j] > 0:
                        if new_pixel in pixel_feas_dict[t].keys():
                            pixel_feas_dict[t][new_pixel] = (
                                pixel_feas_dict[t][new_pixel] +
                                pixels[i, j]) / 2
                        else:
                            pixel_feas_dict[t][new_pixel] = pixels[i, j]

            #left image
            l_im = IM.open(l_image_filename)
            im_w, im_h = l_im.size
            pixels = l_im.load()
            for i in range(0, im_w):
                for j in range(0, im_h):
                    ori_p = pixel_to_point(
                        (i, j + im_h), table_positions[object_table_map[t]],
                        im_actual_size, im_size)
                    new_p = (table_positions[object_table_map[t]][0] -
                             table_positions[object_table_map[t]][1] +
                             ori_p[1],
                             table_positions[object_table_map[t]][1] +
                             table_positions[object_table_map[t]][0] -
                             ori_p[0])
                    new_pixel = point_to_pixel(new_p, (0, 0), (10, 10),
                                               (160, 160))
                    if pixels[i, j] > 0:
                        if new_pixel in pixel_feas_dict[t].keys():
                            pixel_feas_dict[t][new_pixel] = (
                                pixel_feas_dict[t][new_pixel] +
                                pixels[i, j]) / 2
                        else:
                            pixel_feas_dict[t][new_pixel] = pixels[i, j]

            #right image
            r_im = IM.open(r_image_filename)
            im_w, im_h = r_im.size
            pixels = r_im.load()
            for i in range(0, im_w):
                for j in range(0, im_h):
                    ori_p = pixel_to_point(
                        (i, j + im_h), table_positions[object_table_map[t]],
                        im_actual_size, im_size)
                    new_p = (table_positions[object_table_map[t]][0] +
                             table_positions[tobject_table_map[t]][1] -
                             ori_p[1],
                             table_positions[object_table_map[t]][1] -
                             table_positions[object_table_map[t]][0] +
                             ori_p[0])
                    new_pixel = point_to_pixel(new_p, (0, 0), (10, 10),
                                               (160, 160))
                    if pixels[i, j] > 0:
                        if new_pixel in pixel_feas_dict[t].keys():
                            pixel_feas_dict[t][new_pixel] = (
                                pixel_feas_dict[t][new_pixel] +
                                pixels[i, j]) / 2
                        else:
                            pixel_feas_dict[t][new_pixel] = pixels[i, j]

        # filter infeasible with table
#        w = 160
#        h = 160
#        for table_n in rangobjecttable_num):
#            for i in range(0, w):
#                for j in range(0, h):
#                    for t in range(0,len(table_positions)):
#                        t_p = point_to_pixel(table_positions[t],(0,0),(10,10), (160,160))
#                        if in_table((i,j), t_p):
#                            pixel_feas_dict[table_n][(i,j)] = 0
#

# action_table = [4]

        vo.generate_voronoi(object_positions, [[1, 2], [3, 4]])
        for global_heatmap_num in range(0, object_num):
            visualize_heatmap(pixel_feas_dict, state_pixel_dict,
                              [global_heatmap_num])

        # for global_heatmap_num in range(0, table_num):
        #     visualize_heatmap(pixel_feas_dict, combined_state, [global_heatmap_num])

        # visualize_heatmap(pixel_feas_dict, combined_state, action_table)
        # filter_infeasible_with_table('global_heatmap.png', table_positions)

        #make prediction on the top down image, return two files of both sides
        #predict(input_image_name)


# make a task-motion plan

        robot_base_points = [(init_pose.x, init_pose.y)]

        # max_u = float('-inf')
        global max_p
        max_p = []
        for i in range(object_num + 1):
            max_p.append((-10, -10))

        # predefiend task plan
        # a_task_plan = []
        # for obj_index in range(table_num):
        #     a_task_plan.append(obj_index)
        # task_plans = list(combinations(a_task_plan, r=3))

        start_time = time.time()
        plan_cost = None

        ###############  ours: GROP for state space sampling + cmaes (time_budget)  #################
        state_candidates = get_combined_states(object_num, 10)
        # state_candidates = [[[1, 2], [3, 4]]]

        state_candidates = filter_voronoi_non_adj(object_positions,
                                                  state_candidates)
        # calculating state space weights
        # voronoi state weights
        state_weights = []
        for k, v in state_pixel_dict.items():
            w = 0
            for p in v:
                if p in pixel_feas_dict[k]:
                    w += pixel_feas_dict[k][p]
            state_weights.append(w)
        weights = []
        for ss in state_candidates:
            w = 0
            added_state_pixel, search_states = state_addition(
                ss, state_pixel_dict)
            for each_state in search_states:
                if each_state < object_num:  # an original voronoi state
                    w += state_weights[each_state]
                else:
                    each_combine = ss[each_state - object_num]
                    for p in added_state_pixel[each_state]:
                        added_feas = 0
                        is_feas = True
                        for s in each_combine:
                            if p not in pixel_feas_dict[s] or pixel_feas_dict[
                                    s][p] == 0:
                                is_feas = False
                                break
                            else:
                                added_feas += pixel_feas_dict[s][p]
                        if not is_feas:
                            added_feas = 0
                        w += added_feas
            weights.append(w)

        # for debugging purpose
        weights_dict = {}
        for i, w in enumerate(weights):
            if w in weights_dict:
                weights_dict[w].append(i)
            else:
                weights_dict[w] = [i]
        sorted_weights = sorted(weights, reverse=True)
        updated_state_candidates = []
        for i in range(top_n):
            for ss in weights_dict[sorted_weights[i]]:
                print(state_candidates[ss])
                updated_state_candidates.append(state_candidates[ss])
                print(float(sorted_weights[i]) / sum(sorted_weights[:top_n]))

        log_fs = []
        distribute_process = []
        print("starting distribute processes....")

        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")

        os.mkdir('cmaes_log/' + timestamp)
        for k in range(CPU_USE):
            log_f = 'cmaes_log/' + timestamp + '/cpu_' + str(k) + '/'
            log_fs.append(log_f)
            p = Process(target=weighted_distribute,
                        args=(k, timestamp, updated_state_candidates,
                              state_pixel_dict, object_positions,
                              pixel_feas_dict, nav, robot_base_points[0],
                              sorted_weights[:top_n]))
            distribute_process.append(p)
            p.start()
            time.sleep(1)
        for p in distribute_process:
            p.join()

        plan_results = []
        for log_f in log_fs:
            for log_ff in os.listdir(log_f):
                filename = log_f + log_ff + "/results.txt"
                if os.path.exists(filename):
                    with open(filename, 'r') as json_file:
                        plan_results.append(json.load(json_file))

        plan_results = sorted(plan_results, key=lambda x: x['utility'])
        print(plan_results[-1])
        # print(return_q)
        return

        ###############  ours: constructed_states + cmaes  ####################
        #         combined_states = []
        #         states_num = len(state_addition(combined_states, state_pixel_dict)[1])
        #         stop_searching = False
        #         while not stop_searching:
        #
        #             cpu_count = get_cpu_count(states_num)
        #
        #             seq_num = factorial(states_num)
        #             batch_size = seq_num / cpu_count
        #             process = []
        #             manager = Manager()
        #             return_dict = manager.dict()
        #             run = manager.Event()
        #             run.set()
        #
        #             for batch in range(cpu_count):
        #                 p = Process(target=cmaes,
        #                             args=(combined_states, object_positions,
        #                                   state_pixel_dict, pixel_feas_dict, nav,
        #                                   robot_base_points[0], start_time, batch,
        #                                   batch_size, return_dict, run))
        #                 process.append(p)
        #                 p.start()
        #
        #             for p in process:
        #                 p.join()
        #             print(return_dict)
        #             stop_searching = True
        #         end_time = time.time()
        #         print("Planning time: " + str(end_time - start_time))

        ###############  Test: (fully or not)combined_states + cmaes  #################

        # for certain combined states testing:

        #         combined_states = [[1, 2]]
        #         global total_sample_times
        #         total_sample_times = total_sample_times * len(combined_states[0])
        #
        #         states_num = len(state_addition(combined_states, state_pixel_dict)[1])
        #         seq_num = factorial(states_num)
        #         batch_size = seq_num / cpu_count
        #         process = []
        #         for batch in range(cpu_count):
        #             p = Process(target=cmaes,
        #                         args=(combined_states, object_positions,
        #                               state_pixel_dict, pixel_feas_dict, nav,
        #                               robot_base_points[0], start_time, batch,
        #                               batch_size))
        #             process.append(p)
        #             p.start()
        #
        #         for p in process:
        #             p.join()
        #

        ###############  baseline: Every state space + cmaes (time_budget)  #################
        state_candidates = get_combined_states(object_num, 10)
        # state_candidates = [[[1, 2], [3, 4]]]
        log_fs = []
        distribute_process = []
        print("starting distribute processes....")

        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")

        os.mkdir('cmaes_log/' + timestamp)
        for k in range(CPU_USE):
            log_f = 'cmaes_log/' + timestamp + '/cpu_' + str(k) + '/'
            log_fs.append(log_f)
            p = Process(target=distribute,
                        args=(k, timestamp, state_candidates, state_pixel_dict,
                              object_positions, pixel_feas_dict, nav,
                              robot_base_points[0]))
            distribute_process.append(p)
            p.start()
            time.sleep(1)
        for p in distribute_process:
            p.join()

        plan_results = []
        for log_f in log_fs:
            for log_ff in os.listdir(log_f):
                filename = log_f + log_ff + "/results.txt"
                if os.path.exists(filename):
                    with open(filename, 'r') as json_file:
                        plan_results.append(json.load(json_file))

        plan_results = sorted(plan_results, key=lambda x: x['utility'])
        print(plan_results[-1])
        # print(return_q)
        return

        ###############  baseline: voronoi_states + cmaes  #################

        # for split states:
        #        combined_states = []
        #        states_num = len(state_addition(combined_states, state_pixel_dict)[1])
        #        seq_num = factorial(states_num)
        #        batch_size = seq_num / cpu_count
        #        process = []
        #        for batch in range(cpu_count):
        #            p = Process(target=cmaes,
        #                        args=(combined_states, object_positions,
        #                              state_pixel_dict, pixel_feas_dict, nav,
        #                              robot_base_points[0], start_time, batch,
        #                              batch_size))
        #            process.append(p)
        #            p.start()
        #
        #        for p in process:
        #            p.join()

        ###############  baseline: voronoi_states + dfs  #################

        # optimal_plan = dfs(object_states, l, object_positions, state_point_dict, pixel_feas_dict, nav, robot_base_points, start_time)

        print('\n<<<<<<<<<<<<<<< Final results: ')
        end_time = time.time()
        print("Planning time: " + str(end_time - start_time))

        cs.delete_all()
        sim_cs.delete_all()
        # print (table_positions)
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
        map_dir = 'grid_maps/' + diff + '/'
        #map_id = 'c15_0'

        #change the chair num accroding to the difficulty of scenes
        if diff == 'normal':
            chair_num = 20
        elif diff == 'easy':
            chair_num = 15
        elif diff == 'hard':
            chair_num = 25
        trial_idx = 9
        map_id = 'c' + str(chair_num) + '_' + str(trial_idx)

        #load pre-saved map:
        package = 'map_server'
        executable = 'map_server'
        arg = '$(find tamp_perception)/src/' + map_dir + map_id + '.yaml'
        node = roslaunch.core.Node(package, executable, args=arg)
        launch = roslaunch.scriptapi.ROSLaunch()
        launch.start()
        process = launch.launch(node)

        #load pre-saved scene:
        scene_file = map_dir + "scenes.txt"
        cs.reconstruct_env(scene_file, map_id)

        for task_idx in range(0, task_num):

            task_result = {}
            task_result['scene_id'] = map_id
            task_result['task_id'] = task_idx
            print("################### TASK: " + str(task_idx) +
                  " ###################")

            nav = navigator(init_pose, init_orien)

            # Planning...

            #test: remove object when navigating
            for pick_idx in range(0, pick_n):
                test_remover(model_name="target_object" + str(pick_idx))

            unloading_points = []

            #planning-step1: motion planner generates a cost function, mainly for navigation costs

            cost_function = {}
            cost_function['load'] = 1
            cost_function['unload'] = 1
            cost_function['navigate_to'] = {}

            access_loc = {
            }  #a dict for storing symbolic navigation goal and 2d space navigation goal
            access_loc['src'] = loading_pose

            feasibility_loc = {}

            #sample 2d space navigation goal using FCN
            for pick_idx in range(0, pick_n):
                #sample a random unloading object goal
                #unloading_point = (random.uniform(unloading_range[0], unloading_range[1]), 6)
                #read unloading object goal from file
                unloading_point = (unloading_dict[map_id][3 * task_idx +
                                                          pick_idx], 6)
                unloading_points.append(unloading_point)
                #print ("Unloading position: " + str(unloading_point))

                #move the camera at unloading point
                new_camera_state = ModelState()
                new_camera_state.model_name = "distorted_camera"
                new_camera_state.pose.orientation = old_camera_state.pose.orientation
                new_camera_state.pose.position = Point(unloading_point[0],
                                                       unloading_point[1],
                                                       3.46489)
                set_state(new_camera_state)
                time.sleep(1)

                #save the top down image
                input_image_name = "log/images" + '/tr' + str(
                    trial_idx) + '_ta' + str(task_idx) + '_p' + str(
                        pick_idx) + '.jpg'
                cp.save_image(input_image_name)

                #make prediction on the top down image, return two files of both sides
                predict(input_image_name)

                #get utility
                predicted_results = [
                    "top_" + input_image_name.split('/')[-1],
                    "bot_" + input_image_name.split('/')[-1]
                ]

                #predicted_results = ["top_"+input_image_name.split('/')[-1], "bot_test.jpg"]
                #testing one side
                if random_sample == True:
                    predicted_results = [
                        "top_random_test.jpg", "bot_random_test.jpg"
                    ]

                for result in predicted_results:
                    unloading_robot_pose = None
                    mirror_unloading_robot_pose = [0, 0]
                    side = result.split('_')[0]
                    feasibility = hm.get_feasibility("log/results/" + result,
                                                     sample_n)
                    if side == "bot":
                        if random_sample == True:
                            unloading_robot_pixel = hm.random_sample_pixel(
                                'log/results/' + result, sample_n)
                        else:
                            unloading_robot_pixel = hm.sample_pixel(
                                "log/results/" + result, sample_n)
                        if unloading_robot_pixel:
                            unloading_robot_pixel = (unloading_robot_pixel[0],
                                                     unloading_robot_pixel[1] +
                                                     im_size[1] / 2)
                            xy_pose = pixel_to_point(unloading_robot_pixel,
                                                     unloading_point,
                                                     im_actual_size, im_size)
                            unloading_robot_pose = Point(
                                xy_pose[0], xy_pose[1], 0)

                        access_loc['l' + str(pick_idx) +
                                   '_bot'] = unloading_robot_pose
                        feasibility_loc['l' + str(pick_idx) +
                                        '_bot'] = feasbility

                    else:
                        if random_sample == True:
                            mirror_unloading_robot_pixel = hm.random_sample_pixel(
                                'log/results/' + result, sample_n)
                        else:

                            #if we use top side of the image, the goal pose have to be mirrored by the center of the camera
                            mirror_unloading_robot_pixel = hm.sample_pixel(
                                "log/results/" + result, sample_n)
                        if mirror_unloading_robot_pixel:
                            mirror_unloading_robot_pixel = (
                                mirror_unloading_robot_pixel[0],
                                mirror_unloading_robot_pixel[1] +
                                im_size[1] / 2)
                            mirror_unloading_robot_pose = pixel_to_point(
                                mirror_unloading_robot_pixel, unloading_point,
                                im_actual_size, im_size)
                            unloading_robot_pose = Point(
                                2 * unloading_point[0] -
                                mirror_unloading_robot_pose[0],
                                2 * unloading_point[1] -
                                mirror_unloading_robot_pose[1], 0)

                        access_loc['l' + str(pick_idx) +
                                   '_top'] = unloading_robot_pose
                        feasibility_loc['l' + str(pick_idx) +
                                        '_top'] = feasibility

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
                        cost_function['navigate_to'][str(k1) + '-' +
                                                     str(k2)] = cost

            for k2 in feasibility_loc.keys():
                v1 = access_loc['src']
                v2 = access_loc[k2]
                if v1 == None or v2 == None:
                    cost = None
                else:
                    cost = nav.make_plan(v1, v2)
                cost_function['navigate_to']['src' + '-' + str(k2)] = cost

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
                    else:  #deal with navigation action
                        ls = a.split('(')[1].split(',')[0]
                        lg = a.split('(')[1].split(',')[1]
                        total_utility += utility[a_name][ls + '-' + lg]
                if total_utility > max_utility:
                    max_utility = total_utility
                    max_line = i
                i += 2
            #print (max_utility)
            #print (lines[max_line])
            optimal_plan = lines[max_line].split(' ')
            print(optimal_plan)

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

            for a_idx in range(
                    4, len(optimal_plan)
            ):  #already loaded object, start from the navigation action
                a = optimal_plan[a_idx]
                print("#### Action being chosen: " + a)

                if a.split('(')[0] == 'unload':
                    object_idx = int(a.split('(')[1].split(',')[0][1])
                    unloading_point = unloading_points[object_idx]
                    #test: add object back when unloading: this walk around is because of the friction between object and serving plate
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

                    test_spawner(
                        model_name='target_object' + str(object_idx),
                        model_xml=open(
                            "/home/xiaohan/.gazebo/models/wood_cube_7_5cm/model.sdf",
                            'r').read(),
                        robot_namespace="/object",
                        initial_pose=Pose(position=Point(
                            temp_unloaded[0], temp_unloaded[1], 1.4),
                                          orientation=temp_orien),
                        reference_frame="world")

                    if nav_goal != None:
                        if side == "top":  ###### test2 for an unloading policy change for top side
                            top_unloading = [0, 0]
                            top_unloading[0] = unloading_point[0] + (
                                nav_goal.x - unloading_point[0]) / 1
                            #top_unloading[0] = unloading_point[0]
                            top_unloading[1] = unloading_point[1] + 0.15
                            unloading_point = top_unloading
                        if side == "bot":  ###### test2 for an unloading policy change for bot side
                            bot_unloading = [0, 0]
                            bot_unloading[0] = unloading_point[0] + (
                                nav_goal.x - unloading_point[0]) / 3
                            #bot_unloading[0] = unloading_point[0]
                            bot_unloading[1] = unloading_point[1] - 0.1
                            #bot_unloading[1] = unloading_point[1]
                            unloading_point = bot_unloading

                    print("#### Attempting to unload at " +
                          str(unloading_point))
                    unload_object(ac, temp_unloaded, unloading_point)
                    return
                    #check if the object is successfully unloaded to the target position within a specific range
                    time.sleep(1)
                    object_state = model_coordinates(
                        "target_object" + str(object_idx), "")
                    object_current_pose = [
                        object_state.pose.position.x,
                        object_state.pose.position.y
                    ]

                    is_success = False
                    if dist(object_current_pose,
                            unloading_point) <= success_range:
                        is_success = True
                        print("What a successful unloading!!!")
                    success_current_task.append(is_success)
                    test_remover(model_name="target_object" + str(object_idx))
                    ac.go_to_init_joints()

                #navigation action
                else:
                    nav_loc = a.split('(')[1].split(',')[1]
                    side = nav_loc.split('_')[1]
                    nav_goal = access_loc[nav_loc]

                    start_time = time.time()
                    print("#### Navigating to " + nav_loc + str(nav_goal))
                    is_there = go_to_unloading(nav, side, nav_goal)

                    end_time = time.time()
                    time.sleep(1)
                    if is_there:
                        unloading_time.append(end_time - start_time)
                    else:
                        unloading_time.append(60)

            print(access_loc)
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
