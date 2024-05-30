#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import numpy as np
from geometry_msgs.msg import Twist 


class LoadPeople(object):
    """
    detect people in images captured by camera, 
    using the Histogram of Oriented Gradients (HOG) descriptor
    """
    def __init__(self):
    
        self.sub = rospy.Subscriber("/camera/rgb/image_raw",Image,self.camera_callback)              
        self.bridge = CvBridge()        

    def camera_callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)                      

        # initialize HOG descriptor
        hog = cv.HOGDescriptor()
        hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
        # process the images
        img_2 = cv.resize(cv_image,(700,500))
        gray_2 = cv.cvtColor(img_2, cv.COLOR_RGB2GRAY)
        # detect people using HOG
        boxes_2, weights_2 = hog.detectMultiScale(gray_2, winStride=(8,6) )
        boxes_2 = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes_2])

        # iterate over detected bounding boxes, calculate the center of each bounding box 
        # and draw a circle at the center, draw rectangles around detected people
        for (xA, yA, xB, yB) in boxes_2:
            
            medX = xB - xA 
            xC = int(xA+(medX/2)) 
            medY = yB - yA 
            yC = int(yA+(medY/2)) 

            cv.circle(img_2,(xC,yC), 1, (0,255,255), -1)
            cv.rectangle(img_2, (xA, yA), (xB, yB),(255, 255, 0), 2)    
  
            
        cv.imshow('image',img_2)                  
        cv.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('load_people_node', anonymous=True)
    load_people_object = LoadPeople()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv.destroyAllWindows()