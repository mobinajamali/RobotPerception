#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import numpy as np
from geometry_msgs.msg import Twist 




class LoadPeople(object):

    def __init__(self):
    
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw",Image,self.camera_callback)              
        self.bridge_object = CvBridge()        
        self.a = 0.0
        self.b = 0.0

    def sonar_callback(self, msg):
        self.a = msg.range 

    def camera_callback(self,data):
        try:
            # We select bgr8 because its the OpenCV encoding by default
            cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)                      

        hog = cv.HOGDescriptor()
        hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

        
        #Size for the image 
        imX = 700
        imY = 500

        img_2 = cv.resize(cv_image,(imX,imY))
        gray_2 = cv.cvtColor(img_2, cv.COLOR_RGB2GRAY)
        boxes_2, weights_2 = hog.detectMultiScale(gray_2, winStride=(8,6) )
        boxes_2 = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes_2])


        for (xA, yA, xB, yB) in boxes_2:
            
            #Center in X 
            medX = xB - xA 
            xC = int(xA+(medX/2)) 

            #Center in Y
            medY = yB - yA 
            yC = int(yA+(medY/2)) 

            #Draw a circle in the center of the box 
            cv.circle(img_2,(xC,yC), 1, (0,255,255), -1)

            # display the detected boxes in the colour picture
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