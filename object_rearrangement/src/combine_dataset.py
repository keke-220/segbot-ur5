#!/usr/bin/env python
import os
import shutil
import json

datasets = ['data_15', 'data_31']
path = "combined_dataset"
os.mkdir(path)
os.mkdir(path + '/images')

annotations = []
scenes = []
i = 1

for d in datasets:
    #load annotations
    with open(d+'/annotations.txt', 'r') as f:
        cur_annotation = json.load(f)
    #load scene
    #with open(d+'/scenes.txt', 'r') as f:
    #    cur_scene = json.load(f)
    #copy images
    for im in os.listdir(d+'/images'):
        shutil.copyfile(d+'/images/'+im, path+'/images/'+ str(i)+'.jpg')
        image_id = im.split('.')[0]
        #print (image_id)
        for item in cur_annotation:
            if item['image_id'] == int(image_id):

                new_item = {'image_id': i,
                            'robot_pose': item['robot_pose'],
                            'unloading_result': item['unloading_result']}               
                annotations.append(new_item)
        #for item in cur_scene:
        #    if item['image_id'] == int(image_id):
        #        new_item = {'image_id': i,
        #                'chair_pose': item['chair_pose'],
        #                'chair_orien': item['chair_orien']}
        #        scenes.append(new_item)
        i += 1
print (i)

print (len(annotations))
#print (len(scenes))

with open(path+'/annotations.txt', 'w') as out:
    json.dump(annotations, out)

with open(path+'/scenes.txt', 'w') as out:
    json.dump(scenes, out)



   
    
