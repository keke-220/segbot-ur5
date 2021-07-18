import os
from PIL import Image

im_path = "demo_images"
size = (576, 288)
for filename in os.listdir(im_path):
    large_im = Image.new('RGBA', size)
    pixel_large = large_im.load()
    ori_im = Image.open(os.path.join(im_path, filename))
    ori_im = ori_im.convert('RGBA')
    pixel_ori = ori_im.load()
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            li = int(i/9)
            lj = int(j/9)
            pixel_large[i, j] = pixel_ori[li, lj]
    large_im.save(im_path+'/large_'+filename.split(".")[0]+'.png') 
            
            
