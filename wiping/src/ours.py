#!/usr/bin/env python
import pdb
from pdb import set_trace
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
from skimage import data, filters, io, transform
from std_srvs.srv import Empty

from arm_client import arm_client
from camera_processor import camera_processor
from chair_sampler import chair_sampler
from heatmap import heatmap
from navigator import navigator
from scene_sampler import scene_sampler

# parameters
CMAES_DEBUG_FREQ = 5
# pop size of 100 is petty good for num_approach=1
POP_SIZE = 25
MAX_SCORE = 1
TIME_BUDGET = 600000000000
D = 0.6  # robot basic reachability area. This is used for speeding up sampling
AWAY_FROM_TABLE = 0.15
IM_ACTUAL_SIZE = [4, 4]
IM_SIZE = [64, 64]

# TABLE_SIZE = [2.0, 1.0]
# TABLE_POSE = [-2.0, 3.0]

# TABLE_SIZE = [1.5, 1.0]
# TABLE_POSE = [2.0, 2.0]

TABLE_SIZE = [1.0, 1.0]
TABLE_POSE = [2.0, -1.0]


# TABLE_POSE = []
NUM_CHAIR = 2

STD_TABLE_SIZE = [2.0, 1.0]

NUM_APPROACH = 2
# robot initial position and orientation
INIT_POSE = Point(-2, -3, 0)

WHOLE_TABLE_MASK_COUNT = 0


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


INIT_ORIEN = euler_to_quat(0, 0, 1.57)

NUM_TRIAL = 10


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


def draw_robot_position(ori_file, absolute_robot_pose, yaw, is_aug, is_small_mark=False):
    robot_pose = []
    robot_pose.append(absolute_robot_pose[0] - TABLE_POSE[0])
    robot_pose.append(absolute_robot_pose[1] - TABLE_POSE[1])

    image = io.imread(ori_file)

    robot = np.zeros([64, 64, 3], dtype=np.uint8)
    robot.fill(0)
    x = robot_pose[0]
    y = robot_pose[1]
    pixel_y = int((float(x) + 2) * 64 / 4)
    pixel_x = int((-float(y) + 2) * 64 / 4)

    if not is_small_mark:
        for a, b in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1),
                     (-1, 1), (-1, -1), (-2, 0), (2, 0), (2, -1), (2, 1), (0, -2),
                     (0, 2), (1, -2), (1, 2), (2, -2), (2, 2), (2, -3), (2, 3),
                     (1, -3), (1, 3), (2, -4), (2, 4)]:
            robot[31 + a, 31 + b] = [0, 255, 0]
    else:
         for a, b in [(0, 0)]:
            robot[31 + a, 31 + b] = [0, 255, 0]       

    # rotate
    robot = transform.rotate(robot, yaw * 180 / 3.14 - 90)
    # overlay
    if not is_aug:
        for px in range(64):
            for py in range(64):
                if not np.array_equal(robot[px, py], np.array([0, 0, 0])):
                    image[pixel_x + px - 31, pixel_y + py - 31] = robot[px, py]
        return [image]
    else:
        res = []
        for aug in [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1),
                    (-1, 1), (-1, -1)]:
            temp_image = np.copy(image)
            pixel_x += aug[0]
            pixel_y += aug[1]
            for px in range(64):
                for py in range(64):
                    if not np.array_equal(robot[px, py], np.array([0, 0, 0])):
                        temp_image[pixel_x + px - 31,
                                   pixel_y + py - 31] = robot[px, py]
            res.append(temp_image)
        return res


def cmaes(table_pose, table_size, num_approach, image_filename, dirty_mask):

    print(">>>>>>>>>>>>>>>>  Search for: " + str(num_approach) +
          " approach points")

    bounds = [[], []]

    cmaes_init = []
    # to speed up cames, only consider standing points that are within 1 meter of the object
    for s in range(num_approach):
        bounds[0].append(table_pose[0] - table_size[0] / 2 - D)
        bounds[0].append(table_pose[1] - table_size[1] / 2 - D)
        bounds[0].append(0.79)
        bounds[1].append(table_pose[0] + table_size[0] / 2 + D)
        bounds[1].append(table_pose[1] + table_size[1] / 2 + D)
        bounds[1].append(7.07)
        cmaes_init.append(table_pose[0])
        cmaes_init.append(table_pose[1])
        cmaes_init.append(3.93)
    start_time = time.time()
    g = 0
    optimizer = cma.CMAEvolutionStrategy(cmaes_init, 0.5, {
        'bounds': bounds,
        'popsize': POP_SIZE
    })
    max_score = float("inf")
    ret_standing_positions = None
    while True:
        if time.time() - start_time > TIME_BUDGET:
            break
        print("cmaes generation: " + str(g + 1))
        pop_point = optimizer.ask()
        scores = {}
        img_dict = {}
        is_valid_dict = {}
        standing_positions = {}

        # create pop_point visulization for debug
        if g%CMAES_DEBUG_FREQ == 0:
            print ("cmaes logs will save for this iteration")
            pop_image_name = "wiping_log/cmaes/" + str(g+1) + '.png'
            image = io.imread(image_filename)
            io.imsave(pop_image_name, image)
        

        for i in range(len(pop_point)):
            point = pop_point[i]
            standing_positions[i] = []
            
            is_valid_list = []
            for j in range(num_approach):
                is_valid = True
                standing_positions[i].append(
                    [point[j * 3], point[j * 3 + 1], point[j * 3 + 2]])
                # early feedback
                one_point = [point[j * 3], point[j * 3 + 1]]
                one_orien = point[j * 3 + 2]
                one_pixel = point_to_pixel(one_point, table_pose,
                                           IM_ACTUAL_SIZE, IM_SIZE)
                # in collision with table
                # if one_point[
                #         0] > table_pose[0] - table_size[0] / 2 and one_point[
                #             0] < table_pose[0] + table_size[0] / 2:
                #     if one_point[1] > table_pose[
                #             1] - table_size[1] / 2 and one_point[
                #                 1] < table_pose[1] + table_size[1] / 2:
                #         is_valid = False
                side = get_side(one_point[0], one_point[1], TABLE_POSE,
                                TABLE_SIZE)
                if side == "invalid":
                    is_valid = False

                # if orientation is off from what we assume in data collection
                else:
                    min_yaw, max_yaw = orien_range(side)
                    if one_orien > max_yaw or one_orien < min_yaw:
                        is_valid = False
                    # else:
                    #     print(side)
                is_valid_list.append(is_valid)
                # for debug purpose, visualizing the pop_point for each iteration

            if g%CMAES_DEBUG_FREQ == 0:
                for j, standing_position in enumerate(standing_positions[i]):
                    image = draw_robot_position(pop_image_name,
                                                standing_position[:2],
                                                standing_position[2], False, is_small_mark=True)[0]
                    io.imsave(pop_image_name, image)         


            if True not in is_valid_list:
                scores[i] = MAX_SCORE
            else:
                img_lists = []
                for j, standing_position in enumerate(standing_positions[i]):
                    # if is_valid_list[j] == False:
                    #     continue
                    image = draw_robot_position(image_filename,
                                                standing_position[:2],
                                                standing_position[2], False)[0]
                    image_robot_name = 'wiping_log/image_robot/' + image_filename.split(
                        '/')[-1].split('.')[0] + '_' + str(g) + '_' + str(
                            i) + '_' + str(j) + '.jpg'
                    img_lists.append(image_robot_name)
                    io.imsave(image_robot_name, image)
                is_valid_dict[i] = is_valid_list
                img_dict[i] = img_lists
        if img_dict != {}:
            computed_scores = cmaes_objective(img_dict, dirty_mask,
                                              num_approach, is_valid_dict)
        scores_list = []
        for i in range(len(pop_point)):
            if i in scores:
                score_to_add = scores[i]
            else:
                score_to_add = computed_scores[i]

            scores_list.append(score_to_add)

            if score_to_add < max_score:
                max_score = score_to_add
                ret_standing_positions = standing_positions[i]
                print("found better")
                print(max_score)
                print(ret_standing_positions)

        optimizer.tell(pop_point, scores_list)
        g += 1
    return ret_standing_positions
    """
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
        im_b = IM.open("log/envs/diff_" + str(diff) + "_task_" +
                       str(task_idx) + ".jpg")
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
    res['task_idx'] = task_idx
    res['diff'] = diff
    res['reward'] = res['success'] * max_reward * object_num
    res['approach'] = len(res['seq'])

    with open(log_f + "results.txt", 'a') as f:
        f.write(json.dumps(res))

    print("\nBatch results for log dir: " + log_f)
    print("max utility: " + str((-1) * max_value))
    print("expected completion rate: " + str(all_rates[max_index]))
    print("corresponding cost: " + str(all_costs[max_index]))
    print("best generation: " + str(gen_index) + "\n")
    return ret_standing_positions, all_costs[max_index]
    """


def cmaes_objective(img_dict, dirty_mask, num_approach, is_valid_dict):

    computed_scores = {}

    prediction_cmd = "python3.6 ../Pytorch-UNet/predict.py -m ../Pytorch-UNet/cpt/checkpoint_epoch100.pth -i"
    outfiles = []
    for k, v in img_dict.items():
        for im in v:
            prediction_cmd += " " + im
            outfiles.append(im.split('.')[0] + '_OUT' + '.jpg')
    prediction_cmd += " -o"
    for im in outfiles:
        prediction_cmd += " " + im
    os.system(prediction_cmd)

    for k, v in img_dict.items():

        wiped_mask = []
        for i in range(num_approach):

            if is_valid_dict[k][i] == False:
                print ("don't need to run prediction for this image!!!!!!!!!!!!!!!!")
                continue
                

            pred_im = IM.open(v[i].split('.')[0] + '_OUT' + '.jpg')
            pixels = pred_im.load()

            invalid_pred = False
            black_count = 0
            for px in range(IM_SIZE[0]):
                for py in range(IM_SIZE[1]):
                    if pixels[px, py] == 0:
                        black_count += 1

            if black_count > WHOLE_TABLE_MASK_COUNT:
                invalid_pred = True

            if invalid_pred:
                # wiped_mask = []
                # break
                continue

            for px in range(IM_SIZE[0]):
                for py in range(IM_SIZE[1]):
                    if pixels[px, py] < 255 and [px, py] in dirty_mask and [
                            px, py
                    ] not in wiped_mask:
                        wiped_mask.append([px, py])
        ret = 1 - float(len(wiped_mask)) / float(len(dirty_mask))
        # print(ret)
        if ret < 0.5:
            print("Warning: something might go wrong")
            print(v[i].split('.')[0] + '_OUT' + '.jpg')
        computed_scores[k] = ret
    return computed_scores


def orien_range(nav_side):

    if nav_side == 'top':
        nav_yaw = [3.93, 5.50]
    elif nav_side == 'bot':
        nav_yaw = [0.79, 2.36]
    elif nav_side == 'left':
        nav_yaw = [5.50, 7.07]
    elif nav_side == 'right':
        nav_yaw = [2.36, 3.93]
    else:
        return "invalid"
    return nav_yaw


def get_side(x, y, tp, ts):
    if x > tp[0] - ts[0] / 2 and x < tp[0] + ts[0] / 2 and y > tp[
            1] + ts[1] / 2 + AWAY_FROM_TABLE and y < tp[1] + ts[1] / 2 + D:
        return "top"
    elif x > tp[0] - ts[0] / 2 and x < tp[0] + ts[0] / 2 and y < tp[
            1] - ts[1] / 2 - AWAY_FROM_TABLE and y > tp[1] - ts[1] / 2 - D:
        return "bot"
    elif y > tp[1] - ts[1] / 2 and y < tp[1] + ts[1] / 2 and x < tp[
            0] - ts[0] / 2 - AWAY_FROM_TABLE and x > tp[0] - ts[0] / 2 - D:
        return "left"
    elif y > tp[1] - ts[1] / 2 and y < tp[1] + ts[1] / 2 and x > tp[
            0] + ts[0] / 2 + AWAY_FROM_TABLE and x < tp[0] + ts[0] / 2 + D:
        return "right"
    else:
        return "invalid"


def main():

    rospy.init_node('run', anonymous=False)

    nav = navigator(INIT_POSE, INIT_ORIEN)
    ac = arm_client()

    # nav_points = [[-0.1, -0.9], [-1.3, 0.15]]
    # nav_sides = ["bot", "left"]
    # chair_positions = [[1.5, 0.2], [0.24, 0.97], [-0.59, -1.08]]
    # chair_oriens = [0, -0.3, 0.3]

    model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state',
                                           GetModelState)
    old_camera_state = model_coordinates("distorted_camera", "")

    # print(get_side(-3.3, 2.6, TABLE_POSE, TABLE_SIZE))
    # return
    shutil.rmtree("wiping_log/image_robot", ignore_errors=True)
    shutil.rmtree("wiping_log/cmaes", ignore_errors=True)
    os.mkdir("wiping_log/image_robot")
    os.mkdir("wiping_log/cmaes")

    for trial_idx in range(NUM_TRIAL):
        """
        #random chair
        cs = scene_sampler(NUM_CHAIR, 0)
        cs.set_table(TABLE_POSE)
        cs.spawn_chair(TABLE_POSE, TABLE_SIZE)
        chair_positions = cs.get_positions_chair()
        chair_oriens = cs.get_oriens_chair()

        std_cs = scene_sampler(NUM_CHAIR, 0)

        timestamp = datetime.datetime.now().strftime("%m%d%Y-%H%M%S")
        image_filename = "wiping_log/image/" + timestamp + '.jpg'
        capture_image(old_camera_state, TABLE_POSE, image_filename, False)
        with open(image_filename.split('.')[0]+'.json', 'w') as f:
            json.dump({'table_pose': TABLE_POSE,
                       'table_size': TABLE_SIZE,
                       'chair_num': NUM_CHAIR,
                       'chair_poses': chair_positions,
                       'chair_oriens': chair_oriens,
                       }, f, indent=4)
            
        cs.delete_all()
        continue
        """ 
        cs = scene_sampler(NUM_CHAIR, 0)
        std_cs = scene_sampler(NUM_CHAIR, 0)

        # loading a pre-generated scene
        
        # transform from pre-generated scenes to std tables
        for scene_config in os.listdir("scenes_full_table"):
            if scene_config.split('.')[-1] != 'json':
                continue
            with open("scenes_full_table/"+scene_config) as f:
                scene_config_content = json.load(f)

            table_size = scene_config_content['table_size']

            if table_size == STD_TABLE_SIZE:
                continue
               
            table_pose = scene_config_content['table_pose']
            std_table_pose = [table_pose[0] + 20, table_pose[1]]
            std_cs.spawn_table_no_sample([std_table_pose])

            chair_positions = scene_config_content['chair_poses']
            chair_oriens = scene_config_content['chair_oriens']
            # construct std scene for planning -- might use it later when there are more sizes of tables            
            std_chair_positions = []
            std_chair_oriens = []
            offsite_x = (STD_TABLE_SIZE[0]-table_size[0])/2
            offsite_y = (STD_TABLE_SIZE[1]-table_size[1])/2
            for chair_idx, cp in enumerate(chair_positions):

                if cp[0] < table_pose[0]: #left general
                    if cp[1] <= table_pose[1]+table_size[1]/2 and cp[1] >= table_pose[1]-table_size[1]/2: #left normal
                        std_chair_positions.append([cp[0] + 20-offsite_x, cp[1]])
                        std_chair_oriens.append(chair_oriens[chair_idx])
                    elif cp[1] > table_pose[1]:
                        if cp[0] > table_pose[0] - table_size[0]/2: #top normal
                            std_chair_positions.append([cp[0] + 20, cp[1]+offsite_y])
                            std_chair_oriens.append(chair_oriens[chair_idx])
                        else: # top corner
                            std_chair_positions.append([cp[0] + 20, cp[1]+offsite_y])
                            std_chair_positions.append([cp[0] + 20-offsite_x, cp[1]])
                            std_chair_oriens.append(chair_oriens[chair_idx])
                            std_chair_oriens.append(chair_oriens[chair_idx])
                    else:
                        if cp[0] > table_pose[0] - table_size[0]/2: #bot normal
                            std_chair_positions.append([cp[0] + 20, cp[1]-offsite_y])
                            std_chair_oriens.append(chair_oriens[chair_idx])
                        else: # bot corner
                            std_chair_positions.append([cp[0] + 20, cp[1]-offsite_y])
                            std_chair_oriens.append(chair_oriens[chair_idx])
                            std_chair_oriens.append(chair_oriens[chair_idx])
                            std_chair_positions.append([cp[0] + 20-offsite_x, cp[1]])
                else: #right general
                    if cp[1] <= table_pose[1]+table_size[1]/2 and cp[1] >= table_pose[1]-table_size[1]/2: #right normal
                        std_chair_positions.append([cp[0] + 20+offsite_x, cp[1]])
                        std_chair_oriens.append(chair_oriens[chair_idx])
                    elif cp[1] > table_pose[1]:
                        if cp[0] > table_pose[0] - table_size[0]/2: #top normal
                            std_chair_positions.append([cp[0] + 20, cp[1]+offsite_y])
                            std_chair_oriens.append(chair_oriens[chair_idx])
                        else: # top corner
                            std_chair_positions.append([cp[0] + 20, cp[1]+offsite_y])
                            std_chair_oriens.append(chair_oriens[chair_idx])
                            std_chair_oriens.append(chair_oriens[chair_idx])
                            std_chair_positions.append([cp[0] + 20+offsite_x, cp[1]])
                    else:
                        if cp[0] > table_pose[0] - table_size[0]/2: #bot normal
                            std_chair_positions.append([cp[0] + 20, cp[1]-offsite_y])
                            std_chair_oriens.append(chair_oriens[chair_idx])
                        else: # bot corner
                            std_chair_positions.append([cp[0] + 20, cp[1]-offsite_y])
                            std_chair_oriens.append(chair_oriens[chair_idx])
                            std_chair_oriens.append(chair_oriens[chair_idx])
                            std_chair_positions.append([cp[0] + 20+offsite_x, cp[1]])


            std_cs.spawn_chair_no_sample(std_chair_positions, std_chair_oriens)
            fname = "scenes_full_table/transforms/"+scene_config.split('.')[0]+'.jpg'
            config_fname = "scenes_full_table/transforms/"+scene_config.split('.')[0]+'.json'
            new_scene_config = scene_config_content
            new_scene_config['chair_oriens'] = std_chair_oriens
            new_chair_positions = []
            for cp in std_chair_positions:
                new_chair_positions.append([cp[0]-20, cp[1]])
            new_scene_config['chair_poses'] = new_chair_positions
            with open(config_fname, 'w') as f:
                json.dump(new_scene_config, f, indent=4)
            capture_image(old_camera_state, std_table_pose, fname, False)
            std_cs.delete_all()
            cs.delete_all()
        

            
            

        whole_table_mask = []
        (min_px, min_py) = point_to_pixel([
            TABLE_POSE[0] - TABLE_SIZE[0] / 2,
            TABLE_POSE[1] + TABLE_SIZE[1] / 2
        ], TABLE_POSE, IM_ACTUAL_SIZE, IM_SIZE)
        (max_px, max_py) = point_to_pixel([
            TABLE_POSE[0] + TABLE_SIZE[0] / 2,
            TABLE_POSE[1] - TABLE_SIZE[1] / 2
        ], TABLE_POSE, IM_ACTUAL_SIZE, IM_SIZE)

        print(min_px, max_px, min_py, max_py)

        # full table as dirty mask
        for px in range(IM_SIZE[0]):
            for py in range(IM_SIZE[1]):
                if px > min_px and px < max_px and py > min_py and py < max_py:
                    whole_table_mask.append([px, py])

        global WHOLE_TABLE_MASK_COUNT
        WHOLE_TABLE_MASK_COUNT = len(whole_table_mask)
        dirty_mask = whole_table_mask

        # test
        # img_lists = []
        # image = draw_robot_position(image_filename, [-2.89, 3.96], 4.39,
        #                             False)[0]
        # image_robot_name = 'wiping_log/image_robot/' + "test.jpg"
        # img_lists.append(image_robot_name)
        # io.imsave(image_robot_name, image)
        # computed_scores = cmaes_objective({1: [img_lists[0]]}, dirty_mask,
        #                                   NUM_APPROACH)
        # print(computed_scores)

        # return
        approach_points = cmaes(TABLE_POSE, TABLE_SIZE, NUM_APPROACH,
                                image_filename, dirty_mask)
        print(approach_points)

        for approach in approach_points:
            # approach pose
            """
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
            """
            nav_x = approach[0]
            nav_y = approach[1]
            nav_yaw = approach[2]

            wiped_points = []
            if nav.move_to_goal(Point(nav_x, nav_y, 0),
                                euler_to_quat(0, 0, nav_yaw)):
                wipe(ac)
                """
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
                """
        #delete chair
        cs.delete_all()
    return
    # save training image


if __name__ == "__main__":
    main()
