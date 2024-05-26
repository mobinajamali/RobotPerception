#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import numpy as np


class LoadFeature(object):

    def __init__(self):
    
        self.sub = rospy.Subscriber("/camera/rgb/image_raw",Image,self.camera_callback)
        self.bridge = CvBridge()

    def camera_callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
        

        image_1 = cv.imread('/home/user/catkin_ws/src/robot-perception/images/image.jpg',1)
        image_2 = cv_image
        hog = cv.HOGDescriptor()
        hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
        
        img_2 = cv.resize(cv_image,(700,500))
        image_2 = img_2

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

        gray_1 = cv.cvtColor(image_1, cv.COLOR_RGB2GRAY)
        gray_2 = cv.cvtColor(image_2, cv.COLOR_RGB2GRAY)

        #Initialize the ORB Feature detector 
        orb = cv.ORB_create(nfeatures = 1000)

        #Make a copy of the original image to display the keypoints found by ORB
        #This is just a representative
        preview_1 = np.copy(image_1)
        preview_2 = np.copy(image_2)

        #Create another copy to display points only
        dots = np.copy(image_1)

        #Extract the keypoints from both images
        train_keypoints, train_descriptor = orb.detectAndCompute(gray_1, None)
        test_keypoints, test_descriptor = orb.detectAndCompute(gray_2, None)

        #Draw the found Keypoints of the main image
        cv.drawKeypoints(image_1, train_keypoints, preview_1, flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.drawKeypoints(image_1, train_keypoints, dots, flags=2)

        ################## MATCHER ##################

        #Initialize the BruteForce Matcher
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)

        #Match the feature points from both images
        matches = bf.match(train_descriptor, test_descriptor)

        #The matches with shorter distance are the ones we want.
        matches = sorted(matches, key = lambda x : x.distance)

        #Catch some of the matching points to draw           
        good_matches = matches[:313] 
        
        #Parse the feature points
        train_points = np.float32([train_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        test_points = np.float32([test_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        #Create a mask to catch the matching points 
        M, mask = cv.findHomography(train_points, test_points, cv.RANSAC,5.0)

        #Catch the width and height from the main image
        h,w = gray_1.shape[:2]

        #Create a floating matrix for the new perspective
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

        #Create the perspective in the result 
        dst = cv.perspectiveTransform(pts,M)


        # Draw the points of the new perspective in the result image (This is considered the bounding box)
        result = cv.polylines(image_2, [np.int32(dst)], True, (50,0,255),3, cv.LINE_AA)


        cv.imshow('Points',preview_1)        
        cv.imshow('Result',result)      
        cv.waitKey(1)
    
            
    
if __name__ == '__main__':
    load_feature_object = LoadFeature()
    rospy.init_node('load_feature_node', anonymous=True)
    try:
        rospy.spin()
        
    except KeyboardInterrupt:
        print("Shutting down")
    cv.destroyAllWindows()