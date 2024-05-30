#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import numpy as np


class LoadPicture(object):
    """
    capture images from camera and save them 
    """
    def __init__(self):
    
        self.sub = rospy.Subscriber("/camera/rgb/image_raw",Image,self.camera_callback)
        self.bridge = CvBridge()

    def camera_callback(self,data):
        try:
            cv_image = self.bridge.imgmsgadded_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
        
        cv.imwrite('/home/user/catkin_ws/src/robot-perception/images/image.jpg',cv_image)
        cv.imshow('image',cv_image)
        cv.waitKey(1)


if __name__ == '__main__':
    load_picture_object = LoadPicture()
    rospy.init_node('load_face_node', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv.destroyAllWindows()