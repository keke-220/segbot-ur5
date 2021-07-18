#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class camera_processor(object):
    def __init__(self, image_topic):
        self.topic = image_topic
        self.bridge = CvBridge()
    def save_image(self, path):
        '''transfer msg data to cv2 image
        '''
        image_msg = rospy.wait_for_message(self.topic, Image)
        # Convert your ROS Image message to OpenCV2
        cv2_img = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        
        cv2.imwrite(path, cv2_img)
    def save_image_both_sides(self, path):
        print ("") 

if __name__ == '__main__':
    rospy.init_node('camera_processor', anonymous=True)
    test = camera_processor('/top_down_cam/image_raw')
    #test.get_image()
    test.save_image('78.jpg')
