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
        self._positions_object = []
        r = 0.3  #r of the chair model

        #the minimum distance between two chairs (or tables for now)
        self._collision_dist = 2 * r
        self._object_dist = 0.2

        #the range for sampling all around the map
        self._range_all = [-3, 3]
        self._range_x = [-4, 4]
        self._range_y = [-1, 4]
        self.added_model_names = []

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
                spawner(model_name='chair_' + str(i + 1),
                        model_xml=open(
                            "/home/bwilab/.gazebo/models/chair_2/model.sdf",
                            'r').read(),
                        robot_namespace="/chair",
                        initial_pose=Pose(position=Point(
                            self._positions[i][0], self._positions[i][1], 0),
                                          orientation=self.euler_to_quat(
                                              0, 0, self._oriens[i])),
                        reference_frame="world")
            print("Chairs added.")
        except rospy.ServiceException as e:
            print("Spawner fails: ", e)

    def get_positions_chair(self):
        return self._positions_chair

    def get_oriens_chair(self):
        return self._oriens_chair

    def get_positions_table(self):
        return self._positions_table

    def get_positions_object(self):
        return self._positions_object

    def set_positions_table(self, l):
        self._positions_table = l

    def set_object_positions(self, l):
        self._positions_object = l

    def distance(self, p1, p2):
        return math.sqrt(((p1[0] - p2[0])**2) + ((p1[1] - p2[1])**2))

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

    def check_collision_object(self, p):
        ret = False

        # check if on the table

        at_least_one = False
        for ep in self._positions_table:
            if self.distance(ep, p) <= 0.23:
                at_least_one = True
        if not at_least_one:
            ret = True

        # check collision with other objects
        for ep in self._positions_object:
            if self.distance(ep, p) <= self._object_dist:
                ret = True

        return ret

    def sample_pose_chair(self):
        num_sampled = 0
        while num_sampled < self._num_chair:
            #sample a random number within the x range
            x = random.uniform(self._range_x[0], self._range_x[1])
            #sample y value
            y = random.uniform(self._range_y[0], self._range_y[1])
            #check if sampled point is in collision with other existing points
            if self.check_collision_chair([
                    x, y
            ]) == False and self.check_collision_table([x, y]) == False:
                num_sampled += 1
                self._positions_chair.append([x, y])
        #sample orientations
        for i in range(0, self._num_chair):
            self._oriens_chair.append(random.uniform((-1) * math.pi, math.pi))

    def sample_pose_chair_around_object(self):
        num_sampled = 0
        while num_sampled < self._num_chair:
            op = self._positions_object[num_sampled]
            while True:
                #sample a random number within the x range
                x = random.uniform(self._range_x[0], self._range_x[1])
                #sample y value
                y = random.uniform(self._range_y[0], self._range_y[1])
                if self.distance(op, (x, y)) <= 1:
                    break

            #check if sampled point is in collision with other existing points
            if self.check_collision_chair([
                    x, y
            ]) == False and self.check_collision_table([x, y]) == False:
                num_sampled += 1
                self._positions_chair.append([x, y])
        #sample orientations
        for i in range(0, self._num_chair):
            self._oriens_chair.append(random.uniform((-1) * math.pi, math.pi))

    def sample_pose_table(self):
        num_sampled = 0
        while num_sampled < self._num_table:
            #sample a random number within the x range
            x = random.uniform(self._range_all[0], self._range_all[1])
            #sample y value
            y = random.uniform(self._range_all[0], self._range_all[1])
            #check if sampled point is in collision with other existing points
            if self.check_collision_chair([
                    x, y
            ]) == False and self.check_collision_table([x, y]) == False:
                num_sampled += 1
                self._positions_table.append([x, y])

    def sample_pose_object(self, object_num):
        num_sampled = 0
        while num_sampled < object_num:
            #sample a random number within the x range
            x = random.uniform(self._range_x[0], self._range_x[1])

            #sample y value
            y = random.uniform(self._range_y[0], self._range_y[1])
            #check if sampled point is in collision with other existing points
            if self.check_collision_object([x, y]) == False:
                num_sampled += 1
                self._positions_object.append([x, y])

    def spawn_table(self):
        self.sample_pose_table()
        #self._positions_table = [[0,0], [0,0.75]]
        try:
            spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            for i in range(0, len(self._positions_table)):
                spawner(
                    model_name='table_' + str(i + 1),
                    model_xml=open(
                        "/home/bwilab/.gazebo/models/table_cube_square/model.sdf",
                        'r').read(),
                    robot_namespace="/table",
                    initial_pose=Pose(position=Point(
                        self._positions_table[i][0],
                        self._positions_table[i][1], 0),
                                      orientation=self.euler_to_quat(0, 0, 0)),
                    reference_frame="world")
                self.added_model_names.append('table_' + str(i + 1))
            print("Table added.")
        except rospy.ServiceException as e:
            print("Spawner fails: ", e)

    def spawn_chair(self):
        self.sample_pose_chair_around_object()
        try:
            spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            for i in range(0, len(self._positions_chair)):
                spawner(model_name='chair_' + str(i + 1),
                        model_xml=open(
                            "/home/bwilab/.gazebo/models/chair_2/model.sdf",
                            'r').read(),
                        robot_namespace="/chair",
                        initial_pose=Pose(position=Point(
                            self._positions_chair[i][0],
                            self._positions_chair[i][1], 0),
                                          orientation=self.euler_to_quat(
                                              0, 0, self._oriens_chair[i])),
                        reference_frame="world")
                self.added_model_names.append('chair_' + str(i + 1))
            print("Chairs added.")
        except rospy.ServiceException as e:
            print("Spawner fails: ", e)

    def spawn_table_no_sample(self, p):
        try:
            spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            for i in range(0, len(p)):
                spawner(
                    model_name='table_' + str(i + 1),
                    model_xml=open(
                        "/home/bwilab/.gazebo/models/table_cube_square/model.sdf",
                        'r').read(),
                    robot_namespace="/table",
                    initial_pose=Pose(position=Point(p[i][0], p[i][1], 0),
                                      orientation=self.euler_to_quat(0, 0, 0)),
                    reference_frame="world")
                self.added_model_names.append('table_' + str(i + 1))
            print("Table added.")
        except rospy.ServiceException as e:
            print("Spawner fails: ", e)

    def spawn_chair_no_sample(self, p, o):
        try:
            spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

            for i in range(0, len(p)):
                model_name = 'chair_' + str(int(time.time() * 1000))
                spawner(model_name=model_name,
                        model_xml=open(
                            "/home/bwilab/.gazebo/models/chair_2/model.sdf",
                            'r').read(),
                        robot_namespace="/chair",
                        initial_pose=Pose(position=Point(p[i][0], p[i][1], 0),
                                          orientation=self.euler_to_quat(
                                              0, 0, o[i])),
                        reference_frame="world")
                self.added_model_names.append(model_name)
            print("Chairs added.")
        except rospy.ServiceException as e:
            print("Spawner fails: ", e)

    def spawn_object(self, object_num):
        self.sample_pose_object(object_num)
        p = self._positions_object
        try:
            spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            for i in range(0, len(p)):
                spawner(
                    model_name='object_' + str(i + 1),
                    model_xml=open(
                        "/home/bwilab/.gazebo/models_add/wood_cube_7_5cm/model.sdf",
                        'r').read(),
                    robot_namespace="/object",
                    initial_pose=Pose(position=Point(p[i][0], p[i][1], 1.0375),
                                      orientation=self.euler_to_quat(0, 0, 0)),
                    reference_frame="world")
                self.added_model_names.append('object_' + str(i + 1))
            print("Object added.")
        except rospy.ServiceException as e:
            print("Spawner fails: ", e)

    def spawn_object_no_sample(self, p):
        try:
            spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            for i in range(0, len(p)):
                spawner(
                    model_name='object_' + str(i + 1),
                    model_xml=open(
                        "/home/bwilab/.gazebo/models_add/wood_cube_7_5cm/model.sdf",
                        'r').read(),
                    robot_namespace="/object",
                    initial_pose=Pose(position=Point(p[i][0], p[i][1], 1.0375),
                                      orientation=self.euler_to_quat(0, 0, 0)),
                    reference_frame="world")
                self.added_model_names.append('object_' + str(i + 1))
            print("Object added.")
        except rospy.ServiceException as e:
            print("Spawner fails: ", e)

    def delete(self, chair_num):

        try:
            remover = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
            for i in range(0, chair_num):
                remover(model_name='chair_' + str(i + 1))
            print("Chairs removed.")
        except rospy.ServiceException as e:
            print("Spawner fails: ", e)

    def delete_all(self):

        try:
            remover = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
            for n in range(0, len(self.added_model_names)):
                remover(model_name=str(self.added_model_names[n]))
        except rospy.ServiceException as e:
            print("Spawner fails: ", e)


def main():
    test = scene_sampler(10, 1)
    CUR_TABLE_POSITIONS = [[-3, 0.2], [-3, 1], [-1.7, 0.2], [-1.7, 1],
                           [1.25, 0], [1.75, 0], [2.25, 0], [2.75, 0],
                           [1.25, 1.3], [1.75, 1.3], [2, 25, 1.3], [2.75, 1.3],
                           [-1.75, 3], [-1.25, 3], [-0.75, 3], [-0.25, 3],
                           [0.25, 3], [0.75, 3], [1.25, 3], [1.75, 3]]
    test.set_positions_table(CUR_TABLE_POSITIONS)
    test.sample_pose_object(1)
    print(test.check_collision_object([-3, 1]))
    # test.spawn_table()
    # test.spawn_chair()
    # time.sleep(3)
    # test.delete_all()
    #test.sample_pose()
    #test.reconstruct_env("data_100/scenes.txt", 78)


if __name__ == '__main__':
    main()
