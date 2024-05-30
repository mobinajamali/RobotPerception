#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import numpy as np
from cv2 import aruco


class ARTag(object):
    """
    detect ArUco markers in images and overlaying 
    a new image on top of the detected markers
    """
    def __init__(self):
        self.sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.camera_callback)
        self.bridge = CvBridge()

    def camera_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        def order_coordinates(pts, var):
            """
            order the corners of the detected markers either in a 
            clockwise or counter-clockwise based on var
            """
            coordinates = np.zeros((4, 2), dtype="int")
            if var:
                s = pts.sum(axis=1)
                coordinates[0] = pts[np.argmin(s)]
                coordinates[3] = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                coordinates[1] = pts[np.argmin(diff)]
                coordinates[2] = pts[np.argmax(diff)]
            else:
                s = pts.sum(axis=1)
                coordinates[0] = pts[np.argmin(s)]
                coordinates[2] = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                coordinates[1] = pts[np.argmin(diff)]
                coordinates[3] = pts[np.argmax(diff)]
            return coordinates

        # process the image
        image = cv_image
        h, w = image.shape[:2]
        image = cv.resize(image, (int(w * 0.7), int(h * 0.7)))
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # detect ArUco Markers
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # calculate the mean position and draw a circle for each detected marker
        params = []
        try:
            for i in range(len(ids)):
                # Catch the corners of each tag
                c = corners[i][0]
                cv.circle(image, (int(c[:, 0].mean()), int(c[:, 1].mean())), 3, (255, 255, 0), -1)
                # Save the coordinates of the center of each tag
                params.append((int(c[:, 0].mean()), int(c[:, 1].mean())))

            # compute the homography to warp the person image onto the detected markers
            params = np.array(params)
            if len(params) >= 4:
                params = order_coordinates(params, False)
                params_2 = order_coordinates(params, True)

                paint = cv.imread('/home/user/catkin_ws/src/robot-perception/images/wanted.png')
                height, width = paint.shape[:2]

                # Extract the coordinates of this new image which are basically the full-sized image
                coordinates = np.array([[0, 0], [width, 0], [0, height], [width, height]])
                hom, status = cv.findHomography(coordinates, params_2)
                warped_image = cv.warpPerspective(paint, hom, (int(w * 0.7), int(h * 0.7)))

                # Create a mask, subtract the original image from the mask, and add the warped image to the result
                mask = np.zeros([int(h * 0.7), int(w * 0.7), 3], dtype=np.uint8)
                cv.fillConvexPoly(mask, np.int32([params]), (255, 255, 255), cv.LINE_AA)
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
