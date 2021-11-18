#!/usr/bin/env python
from __future__ import print_function
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import cv2
import rospy
import sys
from re import sub

import numpy as np
import roslib
roslib.load_manifest('visnet')
# from std_msgs.msg import String


class GzbObjTracker:

    def __init__(self, tracker_name=None, blob_params=None, subtractor_name=None):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "camera_1/image_raw", Image, self.callback)
        if tracker_name is not None:
            # tracker_names = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
            tracker = None
            if tracker_name == 'BOOSTING':
                tracker = cv2.TrackerBoosting_create()
            if tracker_name == 'MIL':
                tracker = cv2.TrackerMIL_create()
            if tracker_name == 'KCF':
                tracker = cv2.TrackerKCF_create()
            if tracker_name == 'TLD':
                tracker = cv2.TrackerTLD_create()
            if tracker_name == 'MEDIANFLOW':
                tracker = cv2.TrackerMedianFlow_create()
            if tracker_name == 'GOTURN':
                tracker = cv2.TrackerGOTURN_create()
            if tracker_name == 'MOSSE':
                tracker = cv2.TrackerMOSSE_create()
            if tracker_name == "CSRT":
                tracker = cv2.TrackerCSRT_create()
            if tracker is None:
                print('Tracker Type not supported')
            else:
                self.tracker = tracker
                self.tracker_name = tracker_name

        if blob_params is not None:
            self.blob_detector = cv2.SimpleBlobDetector_create(blob_params)

        if subtractor_name is not None:
            if subtractor_name == 'MOG2':
                self.back_subtractor = cv2.createBackgroundSubtractorMOG2()
            else:
                self.back_subtractor = cv2.createBackgroundSubtractorKNN()
            self.subtractor_name = subtractor_name

        self.timer = None
        self.bboxes = None

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # self.obj_track(cv_image)
            # self.blob_detection(cv_image)
            self.obj_detect(cv_image)

        except CvBridgeError as e:
            print(e)

        # try:
        #   self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        # except CvBridgeError as e:
        #   print(e)

    def obj_track(self, frame):
        # Define an initial bounding box
        bbox = (1508, 608, 63, 57)
        # Uncomment the line below to select a different bounding box
        # bbox = cv2.selectROI(frame, False)
        tracker = self.tracker
        OK = False
        if not OK:
            # Initialize tracker with first frame and bounding box
            OK = tracker.init(frame, bbox)
            # Start timer
            self.timer = cv2.getTickCount()

        # Update tracker
        OK, bbox = tracker.update(frame)
        print(bbox)
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - self.timer)

        # Draw bounding box
        if OK:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker name on frame
        cv2.putText(frame, self.tracker_name + " Tracker", (100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(3) & 0xff
        # if k == 27 : break

    def obj_detect(self, frame):
        """
        Detects moving objects in the frame, and returns the bounding boxes
        """
        self.timer = cv2.getTickCount()

        orig_frame = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
        fg_mask = self.back_subtractor.apply(frame)

        th, fg_binary = cv2.threshold(fg_mask, 100, 255, cv2.THRESH_BINARY)
        contours, hierachy = cv2.findContours(
            fg_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # for i, cnt in enumerate(contours):
        #   cv2.drawContours(frame, contours, i, (0, 0, 255), 3)

        bboxes = []
        for contour in contours:
            # print(contour.shape)
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            bboxes.append((x, y, w, h))
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - self.timer)

        cv2.putText(orig_frame, self.subtractor_name + " Subtractor",
                    (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        cv2.putText(orig_frame, "FPS : " + str(int(fps)), (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.imshow('Motion Detection', orig_frame)
        k = cv2.waitKey(3) & 0xff

        self.bboxes = bboxes
        return bboxes

    def blob_detection(self, frame):
        keypoints = self.blob_detector.detect(frame)
        frame = cv2.drawKeypoints(frame, keypoints, np.array(
            []), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('Blob Detection', frame)

        k = cv2.waitKey(3) & 0xff


def main(args):
    # Set up tracker.
    # Instead of MIL, you can also use

    blob_params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    blob_params.minThreshold = 10
    blob_params.maxThreshold = 200

    # Filter by Area.
    blob_params.filterByArea = False
    blob_params.minArea = 1500

    # Filter by Circularity
    blob_params.filterByCircularity = True
    blob_params.minCircularity = 0.1

    # Filter by Convexity
    blob_params.filterByConvexity = True
    blob_params.minConvexity = 0.87

    # Filter by Inertia
    blob_params.filterByInertia = True
    blob_params.minInertiaRatio = 0.01

    gzb_tracker = GzbObjTracker(subtractor_name='MOG2')

    rospy.init_node('image_converter', anonymous=False)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
