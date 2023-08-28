import random
import os
import json
import cv2
import math
from PIL import Image, ImageColor


class voronoi():

    def __init__(self, w, h, dw, dh, ori):
        self.w = w
        self.h = h
        self.domain_w = dw
        self.domain_h = dh
        self.ori = ori
        self.colors = [(255, 0, 0), (255, 128, 0), (255, 255, 0), (0, 255, 0),
                       (0, 255, 255), (0, 0, 255), (128, 0, 255)]

    def point_to_pixel(self, p):
        x = (p[0] - (self.ori[0] - self.domain_w / 2)) * self.w / self.domain_w
        y = ((self.domain_h / 2 - self.ori[1]) - p[1]) * self.h / self.domain_h
        return (x, y)

    def distance(self, p1, p2):
        return math.sqrt(((p1[0] - p2[0])**2) + ((p1[1] - p2[1])**2))

    def generate_voronoi(self, points, combined):
        pixel_points = []
        for p in points:
            pp = self.point_to_pixel(p)
            pixel_points.append(pp)
        img = Image.new('RGB', (self.w, self.h))
        pixels = img.load()
        color_dict = {}

        #compute distances
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                nearest_idx = -1
                nearest_dist = float("inf")
                for p in range(len(pixel_points)):
                    d = self.distance((i, j), pixel_points[p])
                    if nearest_dist > d:
                        nearest_dist = d
                        nearest_idx = p
                color_dict[(i, j)] = nearest_idx
                #print (color_dict)
                #draw voronoi graphs
                pixels[i, j] = self.colors[color_dict[(i, j)]]
        #for p in pixel_points:
        #    pixels[p[0], p[1]] = (255,255,255)

        # if combined parameter is passed, combine the colors as indicated
        if combined:
            for i in range(img.size[0]):
                for j in range(img.size[1]):
                    for l in combined:
                        for l_s in l[1:]:
                            if color_dict[(i, j)] == l_s:
                                pixels[i, j] = self.colors[l[0]]

        img.save("voronois/voronoi_" + str(combined) + '.png')
        return color_dict


if __name__ == '__main__':
    test = voronoi(160, 160, 10, 10, [0, 0])
    points = []
    for i in range(6):
        points.append((random.randint(-5, 5), random.randint(-5, 5)))
    print(points)
    test.generate_voronoi(points)
