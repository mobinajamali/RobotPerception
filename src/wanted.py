#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import numpy as np


class LoadFeature(object):
    """
    detect wanted person using the Histogram of Oriented Gradients (HOG) descriptor 
    and match features between camera livefeed and stored image 
    """
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
        
        # process images
        img_2 = cv.resize(cv_image,(700,500))
        image_2 = img_2
        gray_2 = cv.cvtColor(img_2, cv.COLOR_RGB2GRAY)

        # people detection using HOG descriptor
        hog = cv.HOGDescriptor()
        hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
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

        gray_1 = cv.cvtColor(image_1, cv.COLOR_RGB2GRAY)
        gray_2 = cv.cvtColor(image_2, cv.COLOR_RGB2GRAY)

        # ORB feature detection and matching initialization
        orb = cv.ORB_create(nfeatures = 1000)

        # make copies of the original images for displaying keypoints
        preview_1 = np.copy(image_1)
        preview_2 = np.copy(image_2)
        dots = np.copy(image_1)

        # detect ORB keypoints and compute descriptors for the stored image and camera output
        train_keypoints, train_descriptor = orb.detectAndCompute(gray_1, None)
        test_keypoints, test_descriptor = orb.detectAndCompute(gray_2, None)

        # draw Keypoints of the images
        cv.drawKeypoints(image_1, train_keypoints, preview_1, flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.drawKeypoints(image_1, train_keypoints, dots, flags=2)


        # initialize the BruteForce matcher and matches feature points between the two images
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
        matches = bf.match(train_descriptor, test_descriptor)

        # select the best matches based on distance
        matches = sorted(matches, key = lambda x : x.distance)        
        good_matches = matches[:313] 
        
        # extract the matching keypoints coordinates
        train_points = np.float32([train_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        test_points = np.float32([test_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        # compute the homography matrix  and define the corners of the image 
        # and transform to the prospective of camera
        M, mask = cv.findHomography(train_points, test_points, cv.RANSAC,5.0)
        h,w = gray_1.shape[:2]
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)

        # draw the points of the new perspective in the result image 
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
