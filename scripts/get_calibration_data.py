#!/usr/bin/env python3
import os

import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PoseStamped

# 1. needs to get camera pose
# 2. needs to select other camera in the frame ROI
# 3. calibrate this camera's orientation using other camera's position and ROI position
# 4. save the new projectin matrix to somewhere ?.yaml?
# Note: qualisys node should be running

data_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), os.pardir)) + "/data/" 
print(data_dir)
cam_names = [
    "camera_0",
    "camera_1",
    # "camera_2",
    "camera_3",
    ]

def get_cam_world_pos():
    cam_poses = {}
    # get the first 10 pose msgs from qualisys and average them to store as camera's position
    for cam in cam_names:
        pos_list = []
        print(f"getting position of {cam}...")
        for i in range(10):
            msg_: PoseStamped = rospy.wait_for_message(f"/qualisys/{cam}/pose", PoseStamped)
            pos_list.append([msg_.pose.position.x, msg_.pose.position.y, msg_.pose.position.z])
        
        cam_poses[cam] = np.hstack([np.mean(pos_list, axis=0), np.zeros(3)]).tolist()
    print(cam_poses)

    return cam_poses

def get_cam_pixel_pos():
    cam_rois = {}

    for cam in cam_names:
        img_msg = rospy.wait_for_message(f"{cam}/image_raw/compressed/compressed", CompressedImage)
        bridge = CvBridge()
        cv_img = bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")
        window_name = f"{cam}_calibration"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, cv_img)
        roi = cv2.selectROIs(window_name, cv_img, showCrosshair=True)
        cam_rois[cam] = roi.tolist()
        print(roi)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            print("q is pressed")
            cv2.destroyAllWindows()
    print(cam_rois)
    return cam_rois

def save_cam_view():
    for cam in cam_names:
        img_msg = rospy.wait_for_message(f"{cam}/image_raw/compressed/compressed", CompressedImage)
        bridge = CvBridge()
        cv_img = bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")
        cv2.imwrite(f"{data_dir}/calibration/{cam}_view.jpg", cv_img)

def main():
    import json
    fcalib_data = f"{data_dir}/calibration/calib_data.json"
    cam_poses = get_cam_world_pos()
    cam_rois = get_cam_pixel_pos()
    with open(fcalib_data, "w") as fs:
        data = {"pose": cam_poses, "roi": cam_rois}
        json.dump(data, fs)

if __name__=="__main__":
    rospy.init_node("extrinsic_calibration")
    main()
    # save_cam_view()
