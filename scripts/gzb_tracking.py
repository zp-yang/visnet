import numpy as np
from numpy.lib.function_base import vectorize
import roslib
roslib.load_manifest('visnet')
import sys
import rospy
import cv2
from std_msgs.msg import Int16MultiArray
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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
        fg_mask = self.back_subtractor.apply(frame)

        th, fg_binary = cv2.threshold(fg_mask, 120, 255, cv2.THRESH_BINARY)
        contours, hierachy = cv2.findContours(fg_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []
        for contour in contours:
            # print(contour.shape)
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            bboxes.append([x, y, w, h])
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - self.timer)
        self.bboxes = bboxes

        if self.view:
            cv2.putText(orig_frame, self.subtractor_name + " Subtractor",
                        (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Display FPS on frame
            cv2.putText(orig_frame, "FPS : " + str(int(fps)), (100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # cv2.imshow('Motion Detection', orig_frame)
            # k = cv2.waitKey(3) & 0xff
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

        """
        # self.detector1 = Detector(subtractor_name="MOG2", view=view)
        # self.img_pub1 = rospy.Publisher("tracked_view1", Image, queue_size=5)
        # self.msmt_pub1 = rospy.Publisher("bbox1", Int16MultiArray)
        # self.img_sub1 = rospy.Subscriber("camera_1/image_raw", Image, self.callback, ['cam1', self.detector1, self.msmt_pub1, self.img_pub1])
        """
        for i in range(n_cam):
            camera_name = "camera_"+str(i+1)
            detector = Detector(subtractor_name="MOG2", view=view)
            msmt_pub = rospy.Publisher(camera_name+"_msmt", Int16MultiArray, queue_size=2)
            
            self.detectors.append(detector)
            self.msmt_pubs.append(msmt_pub)

            cb_args = [camera_name, detector, msmt_pub]
            if view:
                img_pub = rospy.Publisher("tracked_view_"+str(i+1), Image,queue_size=5)
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
        bboxes = np.array(bboxes)
        msmt = Int16MultiArray()
        msmt.data = bboxes
        msmt_pub.publish(msmt)
        
        print(topic_name, bboxes)
        if frame is not None:
            try:
                img_pub = cb_args[3]
                frame = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                img_pub.publish(frame)
            except CvBridgeError as e:
                print(e)

def main(args):
    gzb_vis_tracker = GzbVisTracker(n_cam=2, view=False)

    rospy.init_node('drone_tracker')

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shuttin down...")
    cv2.destroyAllWindows()

if __name__=="__main__":
    main(sys.argv)