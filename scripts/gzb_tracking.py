#!/usr/bin/env python3
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import vectorize
import roslib
roslib.load_manifest('visnet')
import sys
import rospy
import cv2
from std_msgs.msg import Int64MultiArray
from visnet.msg import CamMsmt
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

class Detector:
    def __init__(self, subtractor_name=None, tracker_name=None, view=False):
        if subtractor_name is not None:
            if subtractor_name == 'MOG2':
                self.back_subtractor = cv2.createBackgroundSubtractorMOG2()
            else:
                self.back_subtractor = cv2.createBackgroundSubtractorKNN()
            self.subtractor_name = subtractor_name

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

        self.timer = None
        self.bboxes = None
        self.view = view

    def detect(self, frame):
        """
        Detects moving objects in the frame, and returns the bounding boxes
        """
        self.timer = cv2.getTickCount()

        orig_frame = frame.copy()

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_lower = np.array([136, 87, 111], np.uint8)
        red_upper = np.array([180, 255, 255], np.uint8)
        red_mask = cv2.inRange(hsv_frame, red_lower, red_upper)

        green_lower = np.array([40, 40, 40], np.uint8)
        green_upper = np.array([70, 255, 255], np.uint8)
        green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)

        kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        # kernal = np.ones((5, 5), "uint8") # squre kernal
        
        # For red color (bird)
        red_mask = cv2.dilate(red_mask, kernal, iterations=2)
        
        # For green color (drone)
        green_mask = cv2.dilate(green_mask, kernal, iterations=2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (7, 7), 0)
        fg_mask = self.back_subtractor.apply(frame)

        th, fg_binary = cv2.threshold(fg_mask, 80, 255, cv2.THRESH_BINARY)
        fg_binary = cv2.dilate(fg_binary, kernal, iterations=1)

        red_motion_mask = cv2.bitwise_and(red_mask, fg_binary)
        green_motion_mask = cv2.bitwise_and(green_mask, fg_binary)

        red_contours, _ = cv2.findContours(red_motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_contours, _ = cv2.findContours(green_motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blob_bboxes = []
        red_bboxes = []
        blob_thresh = 400
        for contour in red_contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if w * h > blob_thresh:
                cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(orig_frame, "bird", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))  
                red_bboxes.append([x, y, w, h])
            else:
                cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 0, 0), 2)
                cv2.putText(orig_frame, "BLOB", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))  
                blob_bboxes.append([x, y, w, h])

        green_bboxes = []
        for contour in green_contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if w * h > blob_thresh:
                cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(orig_frame, "drone", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))  
                green_bboxes.append([x, y, w, h])
            else:
                cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 0, 0), 2)
                cv2.putText(orig_frame, "BLOB", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))  
                blob_bboxes.append([x, y, w, h])

        bboxes = {'bird':red_bboxes, 'drone':green_bboxes, 'blob':blob_bboxes}
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - self.timer)
        self.bboxes = bboxes
        if self.view:
            # cv2.putText(orig_frame, self.subtractor_name + " Subtractor",
            #             (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Display FPS on frame
            cv2.putText(orig_frame, "FPS : " + str(int(fps)), (100, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            cv2.imshow('Motion Detection', orig_frame)
            cv2.imshow("Motion Mask Red", red_motion_mask)
            # cv2.imshow("Motion Mask Green", green_motion_mask)
            cv2.imshow("Motion Mask", fg_binary)
            cv2.imshow("FG Mask", fg_mask)
            k = cv2.waitKey(3) & 0xff
            # print(k)
            return bboxes, orig_frame      
        return bboxes, None

    def track_2d(self, frame):
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

class GzbVisTracker:
    def __init__(self, n_cam=2, view=False):
        self.bridge = CvBridge()
        
        self.detectors = []
        self.msmt_pubs = []

        if view: 
            self.img_pubs = []
        
        self.img_subs = []
        self.cam_info_subs = []

        for i in range(n_cam):
            camera_name = "camera_"+str(i)
            detector = Detector(subtractor_name="MOG2", view=view)
            msmt_pub = rospy.Publisher(camera_name+"_msmt", CamMsmt, queue_size=2)
            
            self.detectors.append(detector)
            self.msmt_pubs.append(msmt_pub)

            cb_args = [camera_name, detector, msmt_pub]
            if view:
                img_pub = rospy.Publisher("tracked_view_"+str(i), Image,queue_size=5)
                self.img_subs.append(img_pub)

                cb_args.append(img_pub)

            img_sub = rospy.Subscriber(camera_name+"/image_raw", Image, self.img_sub_callback, cb_args)
            self.img_subs.append(img_sub)
            

    def img_sub_callback(self, data, cb_args):
        cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        topic_name = cb_args[0]
        detector = cb_args[1]
        msmt_pub = cb_args[2]

        bboxes, frame = detector.detect(cv_img)
        bird_bboxes = np.array(bboxes['bird'], dtype=np.int16).ravel()
        drone_bboxes = np.array(bboxes['drone'], dtype=np.int16).ravel()
        msmts = np.concatenate([drone_bboxes, bird_bboxes])

        bird_labels = np.zeros(int(bird_bboxes.shape[0]/4))
        drone_labels = np.ones(int(drone_bboxes.shape[0]/4))
        labels = np.concatenate([drone_labels, bird_labels])
        labels = labels.astype(np.int16)

        # print("labels", labels.shape)
        # print("msmts: ", msmts.shape)
        data = CamMsmt()
    
        data.labels = labels.tolist()
        data.msmts = msmts.tolist()
        msmt_pub.publish(data)
        
        # print(topic_name, bboxes)
        if frame is not None:
            try:
                img_pub = cb_args[3]
                frame = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                img_pub.publish(frame)
            except CvBridgeError as e:
                print(e)

def main(args):
    rospy.init_node('drone_tracker')
    gzb_vis_tracker = GzbVisTracker(n_cam=1, view=True)  

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shuttin down...")
    cv2.destroyAllWindows()

if __name__=="__main__":
    main(sys.argv)