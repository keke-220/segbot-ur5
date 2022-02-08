import random
import os
import json
import cv2
from PIL import Image, ImageColor


class heatmap():

    def __init__(self):
        self.im_path = "data_100/images/"
        self.annotation_file = 'data_100/annotations.txt'
        self.gray_colors = [0, 50, 100, 150, 200, 250]
        self.colors = [(0,0,255),(0,255,0),(0,128,0),(255,255,0),(255,150,0),(255,0,0)]
        self.whole_image = False
        
    def similar_color(self, color1, color2):
        tolerance = 50
        if (abs(color1[0]-color2[0])+abs(color1[1]-color2[1])+abs(color1[2]-color2[2])) <= tolerance:
            return True
        else:
            return False

    def gray_to_color(self, filename):

        #manipulate the color map to use only half of the color in the map
        im = Image.open(filename)
        pixels = im.load()
        w = im.size[0]
        h = im.size[1]
        temp_im = Image.new('L', im.size)
        temp_pixels = temp_im.load()
        #pixel_set = []
        
        for i in range(0, w):
            for j in range(0, h):
                temp_pixels[i, j] = 255-(pixels[i, j] + int((255-pixels[i, j])/2.5)) #for gt image
                #temp_pixels[i, j] = 255-(pixels[i, j] + int((255-pixels[i, j])/1.8))
                #if temp_pixels[i, j] not in pixel_set:
                    #pixel_set.append(temp_pixels[i,j])
        #print (pixel_set)
        
        temp_im.save("temp"+filename)
        
        image = cv2.imread("temp"+filename)

        #image = cv2.imread(filename)
        #color_im = cv2.applyColorMap(image, cv2.COLORMAP_PLASMA)
        #color_im = cv2.applyColorMap(image, cv2.COLORMAP_HOT)
        color_im = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        #color_im = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        cv2.imwrite('color.png', color_im)
        """        
        test_im = Image.open('color.png')
        test_pixels = test_im.load()
        for i in range(0, w):
            print (test_pixels[i, 14])
        """
    def generate_heatmaps(self):
        #generate grayscale groundtruth heatmaps from annotations, and this is a discrete heatmap
        annotations = {}
        with open (self.annotation_file, 'r') as f:
            annotations = json.load(f)

        for f in os.listdir(self.im_path):
            im = Image.open(self.im_path+f)
            im = im.convert("RGBA")
            pixelMap = im.load()
            image_id = int(f.split('.')[0])
            #im.show()
            #img = Image.new( 'RGBA', im.size)
            #Grayscale image
            img = Image.new('L', im.size)
            #img.show()
            pixelsNew = img.load()
            levels = {}
            if not self.whole_image:
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
                    #pixelsNew[key[0], key[1]] = self.colors[levels[key]]
                    pixelsNew[key[0], key[1]] = self.gray_colors[levels[key]]
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
                        #pixelsNew[i,j] = self.colors[levels[(i,j)]]
                        pixelsNew[i,j] = self.gray_colors[levels[(i,j)]]
            #new_img = Image.blend(im, img, 0.5)
            #crop to half
            width, height = img.size
            left = 0
            right = width
            top = height/2
            bottom = height

            img = img.crop((left, top, right, bottom))

            img.save("heatmaps/"+f.split('.')[0]+".png")
   
    def blend_images(self, filename1, filename2):
        im1 = Image.open(filename1)
        im1 = im1.convert("RGBA")
        im2 = Image.open(filename2)
        im2 = im2.convert("RGBA")

        pixels = im2.load()
        pixels_ori = im1.load()
        #print pixels[0,0]
        
        for i in range(0, 512):
            for j in range(0, 256):
                """
                if i <=10 and i>=0:
                    pixels_ori[i, j] = pixels[i,j]
                """
                if not self.similar_color(pixels[i,j], pixels[0,0]):
                    pixels_ori[i, j] = pixels[i,j]
        
        #new_img = Image.blend(im1, im2, 0.75)

        #new_img.save('predict_'+filename1.split('.')[0]+'.png')
        im1.save('predict_'+filename1.split('.')[0]+'.png')



    def sample_pixel(self, filename, sample_n):
        #sample a feasible pixel using the same method we calculated feasibility. This function will be used in testing phase.

        img = Image.open(filename)
        #print (filename)
        pixel_set = []
        pixels = img.load()

        #generate a set containing non-black pixels
        for px in range(0, img.size[0]):
            for py in range(0, img.size[1]):
                pixel_value = pixels[px, py]
                if pixel_value > 0:
                    pixel_set.append((px, py))
                    #print (pixel_value)

        #augment the pixel set according to the value of each pixel
        augmented_pixel_set = []
        for p in pixel_set:
            for i in range(0, pixels[p[0], p[1]]):
                augmented_pixel_set.append(p)
        if len(augmented_pixel_set) > 0: 
            max_pixel = (0, 0)
            for n in range(0, sample_n):
                ran = random.randint(0, len(augmented_pixel_set)-1)
                if pixels[augmented_pixel_set[ran][0], augmented_pixel_set[ran][1]] >= pixels[max_pixel[0], max_pixel[1]]:
                    max_pixel = augmented_pixel_set[ran]

            return max_pixel
        return None
    
    def random_sample_pixel(self, filename, sample_n):
        #sample a feasible pixel randomly from all feasible pixel

        img = Image.open(filename)
        #print (filename)
        pixel_set = []
        pixels = img.load()

        #generate a set containing non-black pixels
        for px in range(0, img.size[0]):
            for py in range(0, img.size[1]):
                pixel_value = pixels[px, py]
                if pixel_value > 0:
                    pixel_set.append((px, py))
                    #print (pixel_value)

        #augment the pixel set according to the value of each pixel
        #augmented_pixel_set = []
        #for p in pixel_set:
        #    for i in range(0, pixels[p[0], p[1]]):
        #        augmented_pixel_set.append(p)
        if len(pixel_set) > 0: 
            max_pixel = (0, 0)
            for n in range(0, sample_n):
                ran = random.randint(0, len(pixel_set)-1)
                if pixels[pixel_set[ran][0], pixel_set[ran][1]] >= pixels[max_pixel[0], max_pixel[1]]:
                    max_pixel = pixel_set[ran]

            return max_pixel
        return None
    def enlarge_image(self, filename):
        #size = (576, 288)
        size = (512, 256)
        large_im = Image.new('RGBA', size)
        pixel_large = large_im.load()
        ori_im = Image.open(filename)
        ori_im = ori_im.convert('RGBA')
        pixel_ori = ori_im.load()
        for i in range(0, size[0]):
            for j in range(0, size[1]):
                li = int(i/8)
                lj = int(j/8)
                pixel_large[i, j] = pixel_ori[li, lj]
        large_im.save('large_'+filename.split(".")[0]+'.png') 



    def get_feasibility(self, filename, sample_n):
        average_n = 10000
        img = Image.open(filename)
        #print (filename)
        pixel_set = []
        pixels = img.load()
       
        #generate a set containing non-black pixels
        for px in range(0, img.size[0]):
            for py in range(0, img.size[1]):
                pixel_value = pixels[px, py]
                if pixel_value > 0:
                    pixel_set.append((px, py))
                    #print (pixel_value)

        #augment the pixel set according to the value of each pixel
        augmented_pixel_set = []
        for p in pixel_set:
            for i in range(0, pixels[p[0], p[1]]):
                augmented_pixel_set.append(p)
        if len(augmented_pixel_set) > 0: 
            #average the feasibility results
            total_feasibility = 0
            for i in range(0, average_n):

                max_pixel = (0, 0)
                for n in range(0, sample_n):
                    ran = random.randint(0, len(augmented_pixel_set)-1)
                    if pixels[augmented_pixel_set[ran][0], augmented_pixel_set[ran][1]] >= pixels[max_pixel[0], max_pixel[1]]:
                        max_pixel = augmented_pixel_set[ran]

                total_feasibility += float(pixels[max_pixel[0], max_pixel[1]])/float(255)
            
            return total_feasibility/float(average_n)
        return 0


if __name__ == '__main__':
    test = heatmap()
    filename = "h1_t.jpg"
    #sample_n = 1
    #test.generate_heatmaps()
    #print (test.get_feasibility(filename, sample_n))
    test.gray_to_color(filename)
    test.enlarge_image("color.png")
    test.blend_images("t1_t.jpg", "large_color.png")

















