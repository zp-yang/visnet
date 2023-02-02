#!/usr/bin/env python3
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node

cam_names = [
    "camera_0",
    "camera_1",
    "camera_2",
    "camera_3",
    ]

class MotionDetector(Node):
    def __init__(self, cam_name) -> None:
        super().__init__(f"{cam_name}_motion_detector")
        self.get_logger().info(f"Motion detection node for {cam_name} is started...")
        self.back_subtractor = cv2.createBackgroundSubtractorMOG2()
        # self.back_subtractor = cv2.createBackgroundSubtractorKNN()

        self.cam_name = cam_name
        self.timer = None
        self.bboxes = None
        self.last_mask = None

        self.tracker_timeout = 0.5 # kill tracker after failing to track
        self.trackers = []
        self.tracker_timers = []
        self.not_ok_time_list = []

        self.bridge = CvBridge()

        self.img_sub = self.create_subscription(
            Image,
            f'{self.cam_name}/image_raw',
            lambda msg: self.img_sub_cb(msg, []),
            5,
        )

        self.mot_mask_pub = self.create_publisher(
            Image,
            f'/{self.cam_name}/mot_mask',
            10,
        )

        # self.timer = self.create_timer(1/30, self.timer_cb)
    
    def img_sub_cb(self, msg, args):

        if isinstance(msg, CompressedImage):
            cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        elif isinstance(msg, Image):
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        else:
            cv_img = None
            print("Not a valid image msg type")
        if cv_img is not None:
            mot_mask = self.detect(cv_img)
            mask_msg = self.bridge.cv2_to_imgmsg(mot_mask)

            self.mot_mask_pub.publish(mask_msg)

        
    def detect(self, frame):
        """
        Detects moving objects in the frame
        """
        self.timer = cv2.getTickCount()

        orig_frame = frame.copy()

        kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
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
        
        bboxes = []

        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if w * h > blob_thresh:
                bboxes.append([x,y,w,h])
                cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - self.timer)

        cv2.putText(motion_mask, "FPS : " + str(int(fps)), (100, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        return motion_mask

        # if self.view:
        #     # cv2.putText(orig_frame, self.subtractor_name + " Subtractor",
        #     #             (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        #     # Display FPS on frame
        #     cv2.putText(orig_frame, "FPS : " + str(int(fps)), (100, 20),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        #     # cv2.imshow("original", orig_frame)

        #     # cv2.imshow("Binary Mask", fg_binary)
        #     # cv2.imshow("Motion Mask", motion_mask)
        #     # cv2.imshow("FG Mask", fg_mask)
        #     # print(k)
        #     return bboxes, orig_frame      
        # return bboxes, None


def main(args=None):
    import threading
    rclpy.init(args=args)

    """spin single node"""
    mot_det_node = MotionDetector(cam_names[0])
    rclpy.spin(mot_det_node)

    """ spin multiple nodes """
    # mot_det_nodes = [MotionDetector(cam_name) for cam_name in cam_names]

    # executor = rclpy.executors.MultiThreadedExecutor()
    # for node in mot_det_nodes:
    #     executor.add_node(node)
    
    # executor.spin()
    # mot_det_node.destroy_node()
    rclpy.shutdown()
    

if __name__=="__main__":
    main()