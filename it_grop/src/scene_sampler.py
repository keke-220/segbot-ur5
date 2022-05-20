#!/usr/bin/env python
import rospy
import time
import math
import random
import json
from math import cos, sin
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose, Point, Quaternion

class scene_sampler(object):
    """sample different chair locations and spawn them in the gazebo env
    """
    def __init__(self, num_chair, num_table):
        rospy.wait_for_service("/gazebo/spawn_urdf_model")
        self._num_chair = num_chair
        self._num_table = num_table        
        #store sampled positions for all the chairs
        self._positions_chair = []
        self._oriens_chair = []

        self._positions_table = []

        r = 0.3 #r of the chair model
        
        #the minimum distance between two chairs (or tables for now)
        self._collision_dist = 2*r
        
        #the range for sampling all around the map
        self._range_all = [-3, 3]


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

    def get_positions_chair(self):
        return self._positions_chair

    def get_oriens_chair(self):
        return self._oriens_chair

    def get_positions_table(self):
        return self._positions_table

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

    def check_collision_table(self, p):
        for ep in self._positions_table:
            if self.distance(ep, p) < self._collision_dist:
                return True
        return False
    
    def check_collision_chair(self, p):
        for ep in self._positions_chair:
            if self.distance(ep, p) < self._collision_dist:
                return True
        return False   
    
    def sample_pose_chair(self):
        num_sampled = 0
        while num_sampled < self._num_chair:
            #sample a random number within the x range
            x = random.uniform(self._range_all[0], self._range_all[1])
            #sample y value
            y = random.uniform(self._range_all[0], self._range_all[1])
            #check if sampled point is in collision with other existing points
            if self.check_collision_chair([x,y]) == False and self.check_collision_table([x,y])==False:
                num_sampled += 1
                self._positions_chair.append([x, y])
        #sample orientations
        for i in range(0, self._num_chair):
            self._oriens_chair.append(random.uniform((-1)*math.pi, math.pi))
    
    def sample_pose_table(self):
        num_sampled = 0
        while num_sampled < self._num_table:
            #sample a random number within the x range
            x = random.uniform(self._range_all[0], self._range_all[1])
            #sample y value
            y = random.uniform(self._range_all[0], self._range_all[1])
            #check if sampled point is in collision with other existing points
            if self.check_collision_chair([x,y]) == False and self.check_collision_table([x,y])==False:
                num_sampled += 1
                self._positions_table.append([x, y])


    def spawn_table(self):
        self.sample_pose_table()
        #self._positions_table = [[0,0], [0,0.75]]
        try:
            spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            for i in range(0, len(self._positions_table)):
                spawner(model_name = 'table_'+str(i+1), 
                        model_xml = open("/home/xiaohan/.gazebo/models/table_cube_square/model.sdf", 'r').read(), 
                        robot_namespace = "/table", 
                        initial_pose = Pose(position=Point(self._positions_table[i][0],self._positions_table[i][1],0),orientation=self.euler_to_quat(0, 0, 0)), 
                        reference_frame = "world")
            print("Table added.")
        except rospy.ServiceException as e:
            print("Spawner fails: ", e)
    
    def spawn_chair(self):
        self.sample_pose_chair()
        try:
            spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            for i in range(0, len(self._positions_chair)):
                spawner(model_name = 'chair_'+str(i+1), 
                        model_xml = open("/home/xiaohan/.gazebo/models/chair_2/model.sdf", 'r').read(), 
                        robot_namespace = "/chair", 
                        initial_pose = Pose(position=Point(self._positions_chair[i][0],self._positions_chair[i][1],0),orientation=self.euler_to_quat(0, 0, self._oriens_chair[i])), 
                        reference_frame = "world")
            print("Chairs added.")
        except rospy.ServiceException as e:
            print("Spawner fails: ", e)

    def spawn_table_no_sample(self, p):
        try:
            spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            for i in range(0, len(p)):
                spawner(model_name = 'table_test_'+str(i+1), 
                        model_xml = open("/home/xiaohan/.gazebo/models/table_cube_square/model.sdf", 'r').read(), 
                        robot_namespace = "/table", 
                        initial_pose = Pose(position=Point(p[i][0],p[i][1],0),orientation=self.euler_to_quat(0, 0, 0)), 
                        reference_frame = "world")
            print("Table added.")
        except rospy.ServiceException as e:
            print("Spawner fails: ", e)
    
    def spawn_chair_no_sample(self, p, o):
        try:
            spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            for i in range(0, len(p)):
                spawner(model_name = 'chair_test_'+str(i+1), 
                        model_xml = open("/home/xiaohan/.gazebo/models/chair_2/model.sdf", 'r').read(), 
                        robot_namespace = "/chair", 
                        initial_pose = Pose(position=Point(p[i][0],p[i][1],0),orientation=self.euler_to_quat(0, 0, o[i])), 
                        reference_frame = "world")
            print("Chairs added.")
        except rospy.ServiceException as e:
            print("Spawner fails: ", e)



    def delete(self, chair_num):

        try:
            remover = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
            for i in range(0, chair_num):
                remover(model_name = 'chair_'+str(i+1))
            print("Chairs removed.")
        except rospy.ServiceException as e:
            print("Spawner fails: ", e)

    def delete_all(self):

        try:
            remover = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
            for i in range(0, self._num_chair):
                remover(model_name = 'chair_'+str(i+1))
                remover(model_name = 'chair_test_'+str(i+1))
            print("Chairs removed.")
            for i in range(0, self._num_table):
                remover(model_name = 'table_'+str(i+1))
            print("Tables removed.")
        except rospy.ServiceException as e:
            print("Spawner fails: ", e)

def main():
    test = scene_sampler(10, 1)
    test.spawn_table()
    test.spawn_chair()
    time.sleep(3)
    test.delete_all()
    #test.sample_pose()
    #test.reconstruct_env("data_100/scenes.txt", 78)
if __name__ == '__main__':
    main()
