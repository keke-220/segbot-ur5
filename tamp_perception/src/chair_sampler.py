#!/usr/bin/env python
import rospy
import time
import math
import random
import json
from math import cos, sin
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose, Point, Quaternion
from camera_processor import camera_processor

class chair_sampler(object):
    """sample different chair locations and spawn them in the gazebo env
    """
    def __init__(self, num_chair):
        rospy.wait_for_service("/gazebo/spawn_urdf_model")
        self._num_chair = num_chair
        #store sampled positions for all the chairs
        self._positions = []
        self._oriens = []
        #define sample region
        self._x_range = [-7, 7]
        self._y_range = [[4.35, 5.35], [6.65, 7.65]]
        #the minimum distance between two chairs
        self._collision_dist = 0.8

    def reconstruct_env(self, scene_file, scene_id):
        with open(scene_file, 'r') as f:
            chair_locations = json.load(f)
        for scene in chair_locations:
            if scene["image_id"] == scene_id:
                self._positions = scene["chair_pose"]
                self._oriens = scene["chair_orien"]
        try:
            spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            for i in range(0, len(self._positions)):
                spawner(model_name = 'chair_'+str(i+1), 
                        model_xml = open("/home/xiaohan/.gazebo/models/chair_2/model.sdf", 'r').read(), 
                        robot_namespace = "/chair", 
                        initial_pose = Pose(position=Point(self._positions[i][0],self._positions[i][1],0),orientation=self.euler_to_quat(0, 0, self._oriens[i])), 
                        reference_frame = "world")
            print("Chairs added.")
        except rospy.ServiceException as e:
            print("Spawner fails: ", e)

    def get_positions(self):
        return self._positions

    def get_oriens(self):
        return self._oriens

    def distance(self, p1, p2):
        return math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))

    def euler_to_quat(self, roll, pitch, yaw):
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

    def sample_pose_both_sides(self):
        num_sampled = 0
        pose = []
        while num_sampled < self._num_chair/2:
            #sample a random number within the x range
            x = random.uniform(self._x_range[0], self._x_range[1])
            #randomly choose to sample on one side of the table
            #y_range = random.choice(self._y_range)
            y_range = self._y_range[0]
            #sample y value
            y = random.uniform(y_range[0], y_range[1])
            #check if sampled point is in collision with other existing points
            is_collision = False
            for p in pose:
                if self.distance(p, [x, y]) < self._collision_dist:
                    is_collision = True
            if is_collision == False:
                num_sampled += 1
                pose.append([x, y])
        #self._positions = pose
        while num_sampled < self._num_chair:
            #sample a random number within the x range
            x = random.uniform(self._x_range[0], self._x_range[1])
            #randomly choose to sample on one side of the table
            #y_range = random.choice(self._y_range)
            y_range = self._y_range[1]
            #sample y value
            y = random.uniform(y_range[0], y_range[1])
            #check if sampled point is in collision with other existing points
            is_collision = False
            for p in pose:
                if self.distance(p, [x, y]) < self._collision_dist:
                    is_collision = True
            if is_collision == False:
                num_sampled += 1
                pose.append([x, y])
        self._positions = pose       
        #sample orientations
        for i in range(0, self._num_chair):
            self._oriens.append(random.uniform((-1)*math.pi, math.pi))


    def sample_pose(self):
        num_sampled = 0
        pose = []
        while num_sampled < self._num_chair:
            #sample a random number within the x range
            x = random.uniform(self._x_range[0], self._x_range[1])
            #randomly choose to sample on one side of the table
            #y_range = random.choice(self._y_range)
            y_range = self._y_range[0]
            #sample y value
            y = random.uniform(y_range[0], y_range[1])
            #check if sampled point is in collision with other existing points
            is_collision = False
            for p in pose:
                if self.distance(p, [x, y]) < self._collision_dist:
                    is_collision = True
            if is_collision == False:
                num_sampled += 1
                pose.append([x, y])
        self._positions = pose
        
        #sample orientations
        for i in range(0, self._num_chair):
            self._oriens.append(random.uniform((-1)*math.pi, math.pi))

    def spawn(self):
        for i in range(0, self._num_chair):
            self.sample_pose()
        try:
            spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            for i in range(0, len(self._positions)):
                spawner(model_name = 'chair_'+str(i+1), 
                        model_xml = open("/home/xiaohan/.gazebo/models/chair_2/model.sdf", 'r').read(), 
                        robot_namespace = "/chair", 
                        initial_pose = Pose(position=Point(self._positions[i][0],self._positions[i][1],0),orientation=self.euler_to_quat(0, 0, self._oriens[i])), 
                        reference_frame = "world")
            print("Chairs added.")
        except rospy.ServiceException as e:
            print("Spawner fails: ", e)

    def spawn_both_sides(self):
        for i in range(0, self._num_chair):
            self.sample_pose_both_sides()
        try:
            spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            for i in range(0, len(self._positions)):
                spawner(model_name = 'chair_'+str(i+1), 
                        model_xml = open("/home/xiaohan/.gazebo/models/chair_2/model.sdf", 'r').read(), 
                        robot_namespace = "/chair", 
                        initial_pose = Pose(position=Point(self._positions[i][0],self._positions[i][1],0),orientation=self.euler_to_quat(0, 0, self._oriens[i])), 
                        reference_frame = "world")
            print("Chairs added.")
        except rospy.ServiceException as e:
            print("Spawner fails: ", e)

    def delete_all(self):

        try:
            remover = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
            for i in range(0, self._num_chair):
                remover(model_name = 'chair_'+str(i+1))
            remover(model_name = 'test_stick')
            print("Chairs removed.")
        except rospy.ServiceException as e:
            print("Spawner fails: ", e)

def main():
    test = chair_sampler(10)
    #test.spawn()
    #time.sleep(3)
    test.delete_all()
    #test.sample_pose()
    test.reconstruct_env("data_100/scenes.txt", 78)
if __name__ == '__main__':
    main()
