#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import numpy as np
from cv2 import aruco


class ARTag(object):
    def __init__(self):
        self.sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.camera_callback)
        self.bridge = CvBridge()

    def camera_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        def order_coordinates(pts, var):
            #Initialize an empty array to save to next values 
            coordinates = np.zeros((4, 2), dtype="int")
            if var:
                # Parameters sort model 1
                s = pts.sum(axis=1)
                coordinates[0] = pts[np.argmin(s)]
                coordinates[3] = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                coordinates[1] = pts[np.argmin(diff)]
                coordinates[2] = pts[np.argmax(diff)]
            else:
                # Parameters sort model 2
                s = pts.sum(axis=1)
                coordinates[0] = pts[np.argmin(s)]
                coordinates[2] = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                coordinates[1] = pts[np.argmin(diff)]
                coordinates[3] = pts[np.argmax(diff)]
            return coordinates

        image = cv_image
        h, w = image.shape[:2]

        image = cv.resize(image, (int(w * 0.7), int(h * 0.7)))
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Initialize the aruco Dictionary and its parameters
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()

        # Detect the corners and ids in the image
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # Initialize an empty list for the coordinates
        params = []
        try:
            for i in range(len(ids)):
                # Catch the corners of each tag
                c = corners[i][0]
                # Draw a circle in the center of each detection
                cv.circle(image, (int(c[:, 0].mean()), int(c[:, 1].mean())), 3, (255, 255, 0), -1)
                # Save the coordinates of the center of each tag
                params.append((int(c[:, 0].mean()), int(c[:, 1].mean())))

            # Transform the coordinates list to an array
            params = np.array(params)
            if len(params) >= 4:
                # Sort model 1
                params = order_coordinates(params, False)
                # Sort Model 2
                params_2 = order_coordinates(params, True)


                paint = cv.imread('/home/user/catkin_ws/src/robot-perception/images/wanted.png')
                height, width = paint.shape[:2]

                # Extract the coordinates of this new image which are basically the full-sized image
                coordinates = np.array([[0, 0], [width, 0], [0, height], [width, height]])

                # Find a perspective transformation between the planes
                hom, status = cv.findHomography(coordinates, params_2)

                # Save the warped image in a dark space with the same size as the main image
                warped_image = cv.warpPerspective(paint, hom, (int(w * 0.7), int(h * 0.7)))

                # Create a black mask to do the image operations
                mask = np.zeros([int(h * 0.7), int(w * 0.7), 3], dtype=np.uint8)

                # Replace the area described by the AR tags with white on the black mask
                cv.fillConvexPoly(mask, np.int32([params]), (255, 255, 255), cv.LINE_AA)

                # Perform image subtraction and addition
                substraction = cv.subtract(image, mask)
                addition = cv.add(warped_image, substraction)

                cv.imshow('detection', addition)

        except Exception as e:
            print(e)

        cv.imshow('image', cv_image)
        cv.waitKey(1)

if __name__ == '__main__':
    AR_object = ARTag()
    rospy.init_node('artag_node', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv.destroyAllWindows()
