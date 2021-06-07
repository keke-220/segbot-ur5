import os
import json
from PIL import Image, ImageColor

im_path = "data/images/"
annotation_file = 'data/annotations.txt'


colors = [(0,0,255),(0,255,0),(0,128,0),(255,255,0),(255,150,0),(255,0,0)]
annotations = {}
#self.h = 256
#self.w = 256

whole_image = True

with open (annotation_file, 'r') as f:
    annotations = json.load(f)

print (annotations)

for f in os.listdir(im_path):
    im = Image.open(im_path+f)
    im = im.convert("RGBA")
    pixelMap = im.load()
    image_id = int(f.split('.')[0])
    #im.show()
    img = Image.new( 'RGBA', im.size)
    #img.show()
    pixelsNew = img.load()
    levels = {}
    if not whole_image:
        for instance in annotations:
            if instance["image_id"] == image_id:
                key = (instance["robot_pose"][0], instance["robot_pose"][1])
                if key not in levels.keys():
                    levels[key] = 0
        for instance in annotations:
            if instance["image_id"] == image_id:
                if instance["unloading_result"] == True:
                    levels[(int(instance["robot_pose"][0]),int(instance["robot_pose"][1]))] += 1
        for key in levels.keys():
            pixelsNew[key[0], key[1]] = colors[levels[key]]
    else:
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                levels[(i,j)] = 0
        for instance in annotations:
            if instance["image_id"] == image_id:
                if instance["unloading_result"] == True:
                    levels[(int(instance["robot_pose"][0]),int(instance["robot_pose"][1]))] += 1
       
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                pixelsNew[i,j] = colors[levels[(i,j)]]
        

   
    #img.show()
    new_img = Image.blend(im, img, 0.5)
    #new_img.show()
    #new_img.save("heatmaps/"+f.split('.')[0]+".png")
    img.save("heatmaps/"+f.split('.')[0]+".png")
