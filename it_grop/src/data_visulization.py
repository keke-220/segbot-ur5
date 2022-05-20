#!/usr/bin/env python

import json
import os
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""
This file is used for visualizing the collected dataset.
If and unloading behavior is successful, it will be marked with green circle. Red cross for unsuccessful case.
"""


class data_visualization(object):

    def __init__(self, image_path, annotation_file):
        self._path = image_path
        self._ant_file = annotation_file

        #read annotation file (json) into an object
        self.annotations = {}
        self.h = 256
        self.w = 256
        with open (self._ant_file, 'r') as f:   
            self.annotations = json.load(f)

    def mark_labels(self):
        #mark the annotations on the image for every input data
        for f in os.listdir(self._path):
            image = mpimg.imread(self._path+'/'+f)
            image_id = int(f.split('.')[0])
            plt.imshow(image)
            #create a list to store unloading points. Notice, the y-coord needs conversion
            for instance in self.annotations:
                if instance["image_id"] == image_id:
                    
                    point = [instance['robot_pose'][0], instance['robot_pose'][1]-self.h/2]
                    if instance['unloading_result'] == True:
                        plt.plot(point[0], point[1], 'og', markersize=5)
                    else:
                        plt.plot(point[0], point[1], marker='x', color='red', markersize=5)
            #plt.show()
            plt.savefig('data/visualization/'+f)
            plt.close()
            #break

if __name__ == '__main__':
    image_path = 'data/images'
    annotation_file = 'data/annotations.txt'

    test = data_visualization(image_path, annotation_file)
    test.mark_labels()
