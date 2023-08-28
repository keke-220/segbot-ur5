#!/usr/bin/env python
import os
from PIL import Image

path = 'test_images'

os.mkdir(path + '/train_images')

for image_name in os.listdir(path+'/images'):
    im = Image.open(path+'/images/'+image_name)
    width, height = im.size
    left = 0
    right = width
    top = height/2
    bottom = height

    im1 = im.crop((left, top, right, bottom))
    im1.save(path+'/train_images/'+image_name)
