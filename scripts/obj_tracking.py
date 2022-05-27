import numpy as np
import cv2
import util
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

class MotionDetector:
    def __init__(self, subtractor_name=None, tracker_name=None, view=False):
        if subtractor_name is not None:
            if subtractor_name == 'MOG2':
                self.back_subtractor = cv2.createBackgroundSubtractorMOG2()
            else:
                self.back_subtractor = cv2.createBackgroundSubtractorKNN()
            self.subtractor_name = subtractor_name

            self.bridge = CvBridge()

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
        self.last_mask = None

    def detect(self, frame):
        """
        Detects moving objects in the frame, and returns the bounding boxes
        """
        self.timer = cv2.getTickCount()

        orig_frame = frame.copy()

        kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
        # kernal = np.ones((5, 5), "uint8") # squre kernal
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        fg_mask = self.back_subtractor.apply(frame)

        th, fg_binary = cv2.threshold(fg_mask, 80, 255, cv2.THRESH_BINARY)
        fg_binary = cv2.dilate(fg_binary, kernal, iterations=2)

        if self.last_mask is not None:
            motion_mask = cv2.bitwise_and(fg_binary, self.last_mask)
        else:
            motion_mask = fg_binary
        
        self.last_mask = fg_binary
        blob_thresh = 400
        
        bboxes = None
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - self.timer)
        self.bboxes = bboxes
        if self.view:
            # cv2.putText(orig_frame, self.subtractor_name + " Subtractor",
            #             (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Display FPS on frame
            cv2.putText(fg_binary, "FPS : " + str(int(fps)), (100, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            cv2.imshow("Binary Mask", fg_binary)
            cv2.imshow("Motion Mask", motion_mask)
            # cv2.imshow("FG Mask", fg_mask)
            k = cv2.waitKey(3) & 0xff
            # print(k)
            return bboxes, orig_frame      
        return bboxes, None

    def img_cb(self, data, args):
        cv_img = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        self.detect(cv_img)

def main():
    rospy.init_node("motion_detection")
    cam_names = [
    "camera_0",
    "camera_1",
    # "camera_2",
    "camera_3",
    ]

    motion_det = MotionDetector("MOG2", view=True)
    i = 0
    img_cb_args = []
    img_sub_ = rospy.Subscriber(f"{cam_names[i]}/image_raw/compressed/compressed", CompressedImage, motion_det.img_cb, img_cb_args)

    rospy.spin()

if __name__=="__main__":
    try:
        main()

    except Exception as e:
        print(e)