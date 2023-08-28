#!/usr/bin/env python
import pdb
import copy
import datetime
import json
import os
import random
import shutil
import time
from itertools import combinations, product, permutations
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
NUM_APPROACH = None

CMAES_DEBUG_FREQ = 1000000
POP_SIZE = 20
MAX_SCORE = 1
TIME_BUDGET = None # 1200s for 3 num_approach, 600s for 2, 180s for 1
D = 0.6  # robot basic reachability area. This is used for speeding up sampling
AWAY_FROM_TABLE = 0.15
IM_ACTUAL_SIZE = [4.0, 4.0]
IM_SIZE = [64, 64]

TABLE_SIZE = None
TABLE_POSE = None

NUM_CHAIR = 0

STD_TABLE_SIZE = [2.0, 1.0]

# robot initial position and orientation
INIT_POSE = Point(0, -3, 0)

WHOLE_TABLE_MASK_COUNT = 0
WHOLE_TABLE_MASK = []
MAX_WIPING_THRESHOLD = 150

SIGMA = 0.5
LOG = {}

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

NUM_TRIAL = 2


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


def random_init_pose(table_pose, table_size):
    poses = [[table_pose[0], table_pose[1]-table_size[1]/2-0.3, 1.58], #bot
             [table_pose[0], table_pose[1]+table_size[1]/2+0.3, 4.71], #top
             [table_pose[0]-table_size[0]/2-0.3, table_pose[1], 6.29], #left
             [table_pose[0]+table_size[0]/2+0.3, table_pose[1], 3.15]  #right
             ]
    return random.sample(poses, 1)[0]


def cmaes_single_optimizer(table_pose, table_size, num_approach, image_filename, dirty_mask, chair_positions, log_dir_root):
    
    result_log = {}


    original_image_filename = image_filename
    transform_objective = False
    if table_size != STD_TABLE_SIZE:
        image_filename = image_filename.split('/')[0]+'/transforms/'+image_filename.split('/')[1]
        image_config = image_filename.split('.')[0]+'.json'
        with open(image_config) as f:
            chair_positions = json.load(f)['chair_poses']
        transform_objective = True
        offsite_x = (STD_TABLE_SIZE[0]-table_size[0])/2
        offsite_y = (STD_TABLE_SIZE[1]-table_size[1])/2
        offsite_x_pixel = int(offsite_x * IM_SIZE[0] / IM_ACTUAL_SIZE[0])
        offsite_y_pixel = int(offsite_y * IM_SIZE[0] / IM_ACTUAL_SIZE[0])
        restrict_table_size = table_size
        table_size = STD_TABLE_SIZE


    print(">>>>>>>>>>>>>>>>  Search for: " + str(num_approach) +
          " approach points")

    bounds = [[], []]

    cmaes_init = []

    bound_orien_min = 0.79
    bound_orien_max = 7.07
    
    x_scale = float(IM_ACTUAL_SIZE[0])
    y_scale = float(IM_ACTUAL_SIZE[1])
    orien_scale = float(bound_orien_max-bound_orien_min)

    x_zero = table_pose[0]-float(IM_ACTUAL_SIZE[0])/2
    y_zero = table_pose[1]-float(IM_ACTUAL_SIZE[1])/2
    orien_zero = bound_orien_min 

    # to speed up cames, only consider standing points that are within 1 meter of the object
    for s in range(num_approach):
        bounds[0].append(((table_pose[0]-table_size[0]/2-D)-x_zero)/x_scale)
        bounds[0].append(((table_pose[1]-table_size[1]/2-D)-y_zero)/y_scale)
        bounds[0].append(0)
        bounds[1].append(((table_pose[0]+table_size[0]/2+D)-x_zero)/x_scale)
        bounds[1].append(((table_pose[1]+table_size[1]/2+D)-y_zero)/y_scale)
        bounds[1].append(1)
        
        init_wiping_pose = random_init_pose(table_pose, table_size)
        cmaes_init.append((init_wiping_pose[0]-x_zero)/x_scale)
        cmaes_init.append((init_wiping_pose[1]-y_zero)/y_scale)
        cmaes_init.append((init_wiping_pose[2]-orien_zero)/orien_scale)
    start_time = time.time()
    g = 0
    optimizer = cma.CMAEvolutionStrategy(cmaes_init, SIGMA, {
        'bounds': bounds,
        'popsize': POP_SIZE
    })
    max_score = float("inf")
    ret_standing_positions = None
    while True:
        if time.time() - start_time > TIME_BUDGET:
            break
        result_log[g] = {}
        print("cmaes generation: " + str(g))
        pop_point = optimizer.ask()
        scores = {}
        img_dict = {}
        is_valid_dict = {}
        standing_positions = {}
        side_dict = {}

        # create pop_point visulization for debug
        if g%CMAES_DEBUG_FREQ == 0:
            print ("cmaes logs will save for this iteration")
            pop_image_name = log_dir_root+"wiping_log/cmaes/" + str(g) + '.png'
            image = io.imread(image_filename)
            io.imsave(pop_image_name, image)
        

        for i in range(len(pop_point)):
            
            point = pop_point[i]

            #debug
            # temp = []
            # for s in range(num_approach):
            #     init_wiping_pose = random_init_pose(table_pose, table_size)
            #     temp.append((init_wiping_pose[0]-x_zero)/x_scale)
            #     temp.append((init_wiping_pose[1]-y_zero)/y_scale)
            #     temp.append((init_wiping_pose[2]-orien_zero)/orien_scale)

            # point = temp
            # debug end
            standing_positions[i] = []
            
            is_valid_list = []
            side_list = []

            for j in range(num_approach):
                is_valid = True
                sample_xytheta = [point[j * 3]*x_scale+x_zero, point[j * 3 + 1]*y_scale+y_zero, point[j * 3 + 2]*orien_scale+orien_zero]
                standing_positions[i].append(sample_xytheta)
                one_point = sample_xytheta[:2]
                one_orien = sample_xytheta[2]

                one_pixel = point_to_pixel(one_point, table_pose,
                                           IM_ACTUAL_SIZE, IM_SIZE)
                # early feedback
                # in collision with table
                if one_point[
                        0] > table_pose[0] - table_size[0] / 2 and one_point[
                            0] < table_pose[0] + table_size[0] / 2:
                    if one_point[1] > table_pose[
                            1] - table_size[1] / 2 and one_point[
                                1] < table_pose[1] + table_size[1] / 2:
                        is_valid = False

                # hacky: in collision with chairs; NEED TO BE REMOVED FOR BASELINES!!!!!!
                for cp in chair_positions:
                    if dist(cp, one_point) < 0.3:
                        is_valid = False 

                
                side = get_side(one_point[0], one_point[1], table_pose,
                                table_size)
                side_list.append(side)
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
                    image = draw_robot_position(image_filename,
                                                standing_position[:2],
                                                standing_position[2], False)[0]
                    image_robot_name = log_dir_root+'wiping_log/image_robot/' + image_filename.split(
                        '/')[-1].split('.')[0] + '_' + str(g) + '_' + str(
                            i) + '_' + str(j) + '.jpg'
                    img_lists.append(image_robot_name)
                    io.imsave(image_robot_name, image)
                is_valid_dict[i] = is_valid_list
                side_dict[i] = side_list
                img_dict[i] = img_lists
        if img_dict != {}:
            if transform_objective:
                computed_scores, wiped_mask_dict = cmaes_objective(img_dict, dirty_mask,
                                                num_approach, is_valid_dict, transform=True, side_dict=side_dict, offsite_x=offsite_x_pixel, offsite_y=offsite_y_pixel)
            else:
                computed_scores, wiped_mask_dict = cmaes_objective(img_dict, dirty_mask,
                                                num_approach, is_valid_dict)
        else:
            wiped_mask_dict = {}
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
                if transform_objective and i in side_dict:
                    for j, standing_position in enumerate(ret_standing_positions):
                        side = side_dict[i][j]
                        if side == 'top':
                            standing_position[1] -= offsite_y
                        elif side == 'left':
                            standing_position[0] += offsite_x
                        elif side == 'right':
                            standing_position[0] -= offsite_x
                        elif side == 'bot':
                            standing_position[1] += offsite_y
                print("found better!!!")
                print(max_score)
                print(ret_standing_positions)
                so_far_the_best_fname = log_dir_root + "wiping_log/cmaes/"+'sofarthebest_'+str(g)+'_'+str(i)+'.png'
                image = io.imread(original_image_filename)

                for px, py in dirty_mask:
                    image[py, px] = np.array([255,0,0])
                if i in wiped_mask_dict and wiped_mask_dict[i] != []:
                    for px, py in wiped_mask_dict[i]:
                        image[py, px] = np.array([0,0,255])
                io.imsave(so_far_the_best_fname, image)

                for j, standing_position in enumerate(ret_standing_positions):
                    image = draw_robot_position(so_far_the_best_fname,
                                                standing_position[:2],
                                                standing_position[2], False)[0]
                    io.imsave(so_far_the_best_fname, image)         
        result_log[g]["pop_point"] = np.asarray(pop_point).tolist()
        result_log[g]['scores'] = scores_list
        optimizer.tell(pop_point, scores_list)
        g += 1
    result_log['ret'] = ret_standing_positions
    with open(log_dir_root+'result_log.json', 'w') as f:
        json.dump(result_log, f, indent=4)
    return ret_standing_positions

def get_score(wiped_masks, dirty_area):
    non_overlap = []
    for wiped_mask in wiped_masks:
        for each_mask in wiped_mask:
            if each_mask not in non_overlap and each_mask in dirty_area:
                non_overlap.append(each_mask)
    return 1 - float(len(non_overlap)) / float(len(dirty_mask))


def cmaes_objective(img_dict, dirty_mask, num_approach, is_valid_dict, transform=False, side_dict=None, offsite_x=None, offsite_y=None):

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
    wiped_mask_dict = {}
    for k, v in img_dict.items():

        wiped_mask = []
        for i in range(num_approach):

            if is_valid_dict[k][i] == False:
                # wiped_mask = []
                # break
                continue
                
            if transform:
                side = side_dict[k][i]

            pred_im = IM.open(v[i].split('.')[0] + '_OUT' + '.jpg')
            pixels = pred_im.load()

            invalid_pred = False
            black_count = 0
            for px in range(IM_SIZE[0]):
                for py in range(IM_SIZE[1]):
                    if pixels[px, py] == 0:
                        black_count += 1
                    # if pixels[px, py] == 0 and [px, py] not in WHOLE_TABLE_MASK:
                    #     invalid_pred = True

            if black_count > MAX_WIPING_THRESHOLD:
                invalid_pred = True

            if invalid_pred:
                # wiped_mask = []
                # break
                continue

            for px in range(IM_SIZE[0]):
                for py in range(IM_SIZE[1]):
                    if transform:
                        if pixels[px, py] < 128:
                            transformed_px = px
                            transformed_py = py
                            if side == 'top':
                                transformed_py -= offsite_y
                            elif side == 'left':
                                transformed_px += offsite_x
                            elif side == 'right':
                                transformed_px -= offsite_x
                            elif side == 'bot':
                                transformed_py += offsite_y
                            else:
                                print ("invalid sides!!!!!something definetely goes wrong!!!!!!")
                            
                            if [transformed_px, transformed_py] in dirty_mask and [transformed_px, transformed_py] not in wiped_mask:
                                wiped_mask.append([transformed_px, transformed_py])
                    else:
                        if pixels[px, py] < 128 and [px, py] in dirty_mask and [
                                px, py
                        ] not in wiped_mask:
                            wiped_mask.append([px, py])
        ret = 1 - float(len(wiped_mask)) / float(len(dirty_mask))
        # print(ret)
        # if ret < 0.5:
        #     print("Warning: something might go wrong")
        #     print(v[i].split('.')[0] + '_OUT' + '.jpg')
        computed_scores[k] = ret
        wiped_mask_dict[k] = wiped_mask
    return computed_scores, wiped_mask_dict


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
    
    timestamp = datetime.datetime.now().strftime("%m%d%Y-%H%M%S")
    log_dir = 'eval/' + timestamp + '/'
    os.mkdir(log_dir)

    # shutil.rmtree("wiping_log/image_robot", ignore_errors=True)
    # shutil.rmtree("wiping_log/cmaes", ignore_errors=True)

    # os.mkdir(log_dir+"wiping_log")
    # os.mkdir(log_dir+"wiping_log/image_robot")
    # os.mkdir(log_dir+"wiping_log/cmaes")

    global NUM_APPROACH

    for NUM_APPROACH in [1, 2, 3]:

        global TIME_BUDGET

        if NUM_APPROACH == 1:
            TIME_BUDGET = 180
        elif NUM_APPROACH == 2:
            TIME_BUDGET = 600
        elif NUM_APPROACH == 3:
            TIME_BUDGET == 1200
        
        
        for trial_idx in range(NUM_TRIAL):

            for image_filename in os.listdir("scenes_full_table"):          
                
                if image_filename == 'transforms' or image_filename.split('.')[-1] == 'json':
                    continue
                scene_config_f =  "scenes_full_table/" + image_filename.split('.')[0] + '.json'
                with open(scene_config_f) as f:
                    scene_config = json.load(f)

                if scene_config['table_size'] != [2.0, 1.0]: #### run the exp in parallel
                    continue

                log_dir_root =  log_dir + 'approach_'+str(NUM_APPROACH)+'_scene_'+image_filename.split('.')[0]+'_run_'+str(trial_idx) + '/'
                os.mkdir(log_dir_root)
                os.mkdir(log_dir_root+'wiping_log')
                os.mkdir(log_dir_root+"wiping_log/image_robot")
                os.mkdir(log_dir_root+"wiping_log/cmaes")



                image_filename = "scenes_full_table/" + image_filename
                chair_positions = scene_config['chair_poses']

                global TABLE_POSE
                global TABLE_SIZE
                TABLE_POSE = scene_config['table_pose']
                TABLE_SIZE = scene_config['table_size']

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
                        if px >= min_px and px <= max_px and py >= min_py and py <= max_py:
                            whole_table_mask.append([px, py])

                global WHOLE_TABLE_MASK_COUNT
                global WHOLE_TABLE_MASK
                WHOLE_TABLE_MASK_COUNT = len(whole_table_mask)
                WHOLE_TABLE_MASK = whole_table_mask
                dirty_mask = whole_table_mask


                approach_points = cmaes_single_optimizer(TABLE_POSE, TABLE_SIZE, NUM_APPROACH,
                                        image_filename, dirty_mask, chair_positions, log_dir_root)
                print(approach_points)
    # save training image


if __name__ == "__main__":
    main()
