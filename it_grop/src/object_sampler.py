#!/usr/bin/env python
import os
import rospy
import time
import math
import random
import json
from math import cos, sin
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose, Point, Quaternion

class object_sampler(object):
    """sample different object (boxes) locations and spawn them in the gazebo env
    """
    def __init__(self, num_t, num_o, table_pos, table_length, table_width, min_r, max_r):
        rospy.wait_for_service("/gazebo/spawn_urdf_model")
        self.num_t = num_t #target object number
        self.num_o = num_o #obstacle object number
        self.object_file = "~/.gazebo/models/random_cube/model.sdf"#for random obstacle objects
        self.target_file = "~/.gazebo/models/wood_cube_7_5cm/model.sdf"#for random obstacle objects
        self.min_r = min_r
        self.max_r = max_r
        table_length = float(table_length)
        table_width = float(table_width)
        self._x_range = [table_pos[0]-table_length/2+1.5*max_r, table_pos[0]+table_length/2-1.5*max_r]
        self._y_range = [table_pos[1]-table_width/2+1.5*max_r, table_pos[1]+table_width/2-1.5*max_r]
        self._collision_dist = 3*max_r
        self._positions = []
        self.table_pos = table_pos


    def random_size(self):
        x = 2*random.uniform(self.min_r, self.max_r)   
        y = 2*random.uniform(self.min_r, self.max_r)   
        z = 2*random.uniform(self.min_r, self.max_r) 
        return [x,y,z]

    def create_object_model(self):
        f = open(os.path.expanduser(self.object_file), 'r')
        replacement = ''
        changes = ''
        s = self.random_size()
        for line in f:
            if '<size>' in line:
                changes="<size>"+str(s[0])+' '+str(s[1])+' '+str(s[2])+'</size>'+'\n'
            elif '<pose>' in line:
                changes="<pose>0 0 "+str(float(s[2])/2)+' 0 0 0</pose>'+'\n'
            else:
                changes = line
            replacement = replacement+changes
            changes=''
        f.close()
        fout = open(os.path.expanduser(self.object_file), 'w')
        fout.write(replacement)
        fout.close()

    def random_pose(self, num):
        num_sampled = 0
        pose = []
        while num_sampled < num:
            #sample a random number within the x range
            x = random.uniform(self._x_range[0], self._x_range[1])
            #sample y value
            y = random.uniform(self._y_range[0], self._y_range[1])
            #check if sampled point is in collision with other existing points
            is_collision = False
            for p in pose:
                if self.distance(p, [x, y]) < self._collision_dist:
                    is_collision = True
            if is_collision == False:
                num_sampled += 1
                pose.append([x, y])
        for p in pose:
            self._positions.append(p)
        return pose

    def distance(self, p1, p2):
        return math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))

    def random_orien(self, num):
        oriens = []
        for i in range(0, num):
            oriens.append(random.uniform((-1)*math.pi, math.pi))
        return oriens
        
    def delete_tar(self):

        try:
            remover = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
            for i in range(0, self.num_o):
                remover(model_name = 'tar_'+str(i+1))
            print("Target objects removed.")
        except rospy.ServiceException as e:
            print("Spawner fails: ", e)


    def delete_obs(self):

        try:
            remover = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
            for i in range(0, self.num_o):
                remover(model_name = 'obs_'+str(i+1))
            print("Obstacle objects removed.")
        except rospy.ServiceException as e:
            print("Spawner fails: ", e)

    def spawn_obs(self):
        #for i in range(0, self._num_chair):
        pose = self.random_pose(self.num_o)
        oriens = self.random_orien(self.num_o)
        try:
            spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            for i in range(0, len(pose)):
                self.create_object_model()
                spawner(model_name = 'obs_'+str(i+1), 
                        model_xml = open(os.path.expanduser(self.object_file), 'r').read(), 
                        robot_namespace = "/obs", 
                        initial_pose = Pose(position=Point(pose[i][0],pose[i][1],1.2),orientation=self.euler_to_quat(0, 0, oriens[i])), 
                        reference_frame = "world")
            print("Obstacle objects added.")
        except rospy.ServiceException as e:
            print("Spawner fails: ", e)

    def spawn_tar(self):
        #for i in range(0, self._num_chair):
        pose = self.random_pose(self.num_t)
        try:
            spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            for i in range(0, len(pose)):
                self.create_object_model()
                spawner(model_name = 'tar_'+str(i+1), 
                        model_xml = open(os.path.expanduser(self.target_file), 'r').read(), 
                        robot_namespace = "/tar", 
                        initial_pose = Pose(position=Point(pose[i][0],self.table_pos[1],1.2),orientation=self.euler_to_quat(0, 0, 0)), 
                        reference_frame = "world")
            print("Target objects added.")
        except rospy.ServiceException as e:
            print("Spawner fails: ", e)

   

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
def main():
    max_r = 0.1 #the maximum object size for random obstacles
    min_r = 0.01
    
    test = object_sampler(10, 20, [0,0], 16, 1, min_r, max_r)
    #test.create_object_model()
    test.delete_obs()
    test.delete_tar()
    test.spawn_tar()
    test.spawn_obs()
    print (len(test._positions))
if __name__ == '__main__':
    main()
