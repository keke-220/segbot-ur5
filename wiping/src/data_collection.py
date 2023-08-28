#!/usr/bin/env python
import copy
import datetime
import json
import os
import random
import shutil
import time
from itertools import combinations, product
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


def go_to_wiping(nav, max_side, goal):
    loading_pos = goal
    is_there = False
    if goal != None:
        if max_side == "top":
            loading_orien = euler_to_quat(0, 0, -1.57)
        elif max_side == "bot":
            loading_orien = euler_to_quat(0, 0, 1.57)
        elif max_side == "left":
            loading_orien = euler_to_quat(0, 0, 0)
        else:
            loading_orien = euler_to_quat(0, 0, 3.14)
        loading_pos = Point(
            goal[0], goal[1],
            0)  ###localization issue, test1 for pushing the robot to the table
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


def wipe(ac):
    robot_pose = rospy.wait_for_message('/odom', Odometry).pose.pose.position
    robot_orien = rospy.wait_for_message('/odom',
                                         Odometry).pose.pose.orientation

    y_range = [0.56, 0.4, 0.24, 0.08, -0.08, -0.24, -0.4, -0.56]
    for y_base in y_range:
        for i in range(1, 5):
            x_base = sqrt(0.7 - y_base**2)
            x = x_base * (1 - float(i) / float(8))
            y = y_base * (1 - float(i) / float(8))
            ac.move_relative_pose(Point(x, y, 1.178))
    ac.go_to_init_joints()
    # ac.move_relative_pose(Point(0.5475, 0.3, 1.25))
    # ac.move_relative_pose(Point(0.365, 0.2, 1.25))


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

        plan_utility, total_nav = get_tm_plan_utility(p, l, feas_dict, nav)
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


def cmaes(combined, object_positions, state_pixel, feas_dict, nav,
          robot_initial_position, start_time, batch, batch_size, return_dict,
          run):
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

    p_u, p_c = get_tm_plan_utility(combined, repeated_standing_positions,
                                   object_sequences, feas_dict, nav,
                                   state_pixel)
    # total_utility.append(p_u)
    # total_cost.append(p_c)
    # m_u = max(total_utility)
    # m_c = total_cost[total_utility.index(m_u)]
    return (-1) * p_u, p_c


# parameters
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

# global variable

CUR_TABLE_POSITIONS = [
]  # for globally storing table positions of the current scene
max_u = float('-inf')  # max utility for dfs baseline
max_p = None  # max plan for dfs baseline


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
    return utility, total_nav


def get_cpu_count(table_num):
    if table_num == 1:
        cpu_count = 1
    elif table_num <= 3:
        cpu_count = 2
    else:
        cpu_count = 12
    return cpu_count


def main():
    im_actual_size = [4, 4]
    im_size = [64, 64]
    rospy.init_node('run', anonymous=False)

    # robot initial position and orientation
    init_pose = Point(0, -3, 0)
    init_orien = euler_to_quat(0, 0, 1.57)
    nav = navigator(init_pose, init_orien)
    ac = arm_client()

    # nav_points = [[-0.1, -0.9], [-1.3, 0.15]]
    # nav_sides = ["bot", "left"]
    num_chair = 4
    # chair_positions = [[1.5, 0.2], [0.24, 0.97], [-0.59, -1.08]]
    # chair_oriens = [0, -0.3, 0.3]

    model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state',
                                           GetModelState)
    old_camera_state = model_coordinates("distorted_camera", "")

    num_data = 1000
    for data_idx in range(num_data):
        # robot wiping starting pose
        x = 0
        y = 0
        while dist((x, y), (0, 0)) < 3:
            x = random.uniform(-3.5, 3.5)
            y = random.uniform(-3.5, 3.5)
        yaw = random.uniform(0, 6.28)
        nav.move_to_goal(Point(x, y, 0), euler_to_quat(0, 0, yaw))

        #random chair
        cs = scene_sampler(num_chair, 0)
        cs.spawn_chair()

        timestamp = datetime.datetime.now().strftime("%m%d%Y-%H%M%S")
        image_filename = "wiping_data/image/" + timestamp + '.jpg'
        capture_image(old_camera_state, (0, 0), image_filename, False)

        while True:
            # approach pose
            sides = ['top', 'bot', 'left', 'right']
            random.shuffle(sides)
            nav_side = sides[0]
            if nav_side == 'top':
                nav_x = random.uniform(-1, 1)
                nav_y = random.uniform(0.5, 1.1)
                nav_yaw = random.uniform(3.93, 5.50)
            elif nav_side == 'bot':
                nav_x = random.uniform(-1, 1)
                nav_y = random.uniform(-1.1, -0.5)
                nav_yaw = random.uniform(0.79, 2.36)
            elif nav_side == 'left':
                nav_x = random.uniform(-1.6, -1)
                nav_y = random.uniform(-0.5, 0.5)
                nav_yaw = random.uniform(5.50, 7.07)
            else:
                nav_x = random.uniform(1, 1.6)
                nav_y = random.uniform(-0.5, 0.5)
                nav_yaw = random.uniform(2.36, 3.93)

            wiped_points = []
            if nav.move_to_goal(Point(nav_x, nav_y, 0),
                                euler_to_quat(0, 0, nav_yaw)):
                wipe(ac)
                nav_filename = "wiping_data/nav/" + timestamp + '.txt'
                with open(nav_filename, 'w') as nav_f:
                    nav_f.write(
                        str(nav_x) + ',' + str(nav_y) + ',' + str(nav_yaw))

                if os.path.exists("/home/bwilab/.ros/temp_contact.txt"):
                    with open("/home/bwilab/.ros/temp_contact.txt") as wf:
                        lines = wf.readlines()
                        for l in lines:
                            x = float(l.split(" ")[0])
                            y = float(l.split(" ")[1])
                            wiped_points.append((x, y))
                    os.remove("/home/bwilab/.ros/temp_contact.txt")

                # save wiping result as label
                wi = IM.open(image_filename)
                im_w, im_h = wi.size
                pixels = wi.load()
                label_filename = "wiping_data/label/" + timestamp + '.jpg'
                for pi in range(im_w):
                    for pj in range(im_h):
                        pixels[pi, pj] = (0, 0, 0)
                for wiped_point in wiped_points:
                    wiped_pixel = point_to_pixel(wiped_point, (0, 0), (4, 4),
                                                 (64, 64))
                    if pixels[wiped_pixel[0], wiped_pixel[1]] == (0, 0, 0):
                        pixels[wiped_pixel[0],
                               wiped_pixel[1]] = (255, 255, 255)
                wi.save(label_filename)
                break
        #delete chair
        cs.delete_all()
    return
    # save training image


"""
# mark robot base point
wi = IM.open(image_filename)
im_w, im_h = wi.size
pixels = wi.load()

nav_pixel = point_to_pixel(nav_points[nav_idx], (0, 0), (4, 4),
                           (64, 64))
for offset in [(0, 0), (0, 1), (0, -1), (1, 0), (1, 1), (1, -1),
               (-1, 0), (-1, -1), (-1, 1)]:
    x = nav_pixel[0] + offset[0]
    y = nav_pixel[1] + offset[1]
    pixels[x, y] = (0, 0, 255)
wi.save(image_filename)

wiped_points = []
if go_to_wiping(nav, nav_sides[nav_idx], nav_points[nav_idx]):
    wipe(ac)
with open("/home/bwilab/.ros/temp_contact.txt") as wf:
    lines = wf.readlines()
    for l in lines:
        x = float(l.split(" ")[0])
        y = float(l.split(" ")[1])
        wiped_points.append((x, y))
os.remove("/home/bwilab/.ros/temp_contact.txt")
# save wiping result as label
label_filename = "wiping_data/label/" + timestamp + '.jpg'
for pi in range(im_w):
    for pj in range(im_h):
        pixels[pi, pj] = (0, 0, 0)
for wiped_point in wiped_points:
    wiped_pixel = point_to_pixel(wiped_point, (0, 0), (4, 4), (64, 64))
    if pixels[wiped_pixel[0], wiped_pixel[1]] == (0, 0, 0):
        pixels[wiped_pixel[0], wiped_pixel[1]] = (255, 255, 255)
wi.save(label_filename)

return
"""
if __name__ == "__main__":
    main()
