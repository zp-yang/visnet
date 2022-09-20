#!/usr/bin/env python3
import os

import rclpy
from rclpy.node import Node
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

def wait_for_message(node : Node, msg_type, topic, time_out=10):
    import time
    class _wfm(object):
        def __init__(self) -> None:
            self.time_out = time_out
            self.msg = None
        
        def cb(self, msg):
            self.msg = msg
    elapsed = 0
    wfm = _wfm()
    subscription = node.create_subscription(msg_type, topic, wfm.cb, 1)
    # rate = node.create_rate(10)
    while rclpy.ok():
        if wfm.msg != None : return wfm.msg
        node.get_logger().info(f'waiting for {topic} ...')
        rclpy.spin_once(node)
        time.sleep(0.1)
        elapsed += 0.1
        if elapsed >= wfm.time_out:
            node.get_logger().warn(f'time out waiting for {topic}...')
            return None
    subscription.destroy()

def get_cam_world_pos(node: Node):
    cam_poses = {}
    for cam in cam_names:
        print(f"getting position of {cam}...")
        msg_: PoseStamped = wait_for_message(node, PoseStamped, f"/qualisys/{cam}/pose")
        cam_poses[cam] = [msg_.pose.position.x, msg_.pose.position.y, msg_.pose.position.z, 0, 0, 0]

    print(cam_poses)
    return cam_poses

def get_cam_pixel_pos(node: Node):
    cam_rois = {}

    for cam in cam_names:
        img_msg = wait_for_message(node, CompressedImage, f"{cam}/camera/image_raw/compressed")
        bridge = CvBridge()
        cv_img = bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")
        window_name = f"{cam}_calibration"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, cv_img)
        roi = cv2.selectROIs(window_name, cv_img, showCrosshair=True)
        cam_rois[cam] = roi.tolist()
        print(roi)
        print("Press Q to select next camera ...")
        if cv2.waitKey(0) & 0xFF == ord('q'):
            print("q is pressed")
            cv2.destroyAllWindows()
    print(cam_rois)
    return cam_rois

def save_cam_view(node: Node, save_dir):
    for cam in cam_names:
        img_msg = wait_for_message(node, CompressedImage, f"{cam}/camera/image_raw/compressed")
        bridge = CvBridge()
        cv_img = bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")
        cv2.imwrite(f"{save_dir}/{cam}_view.jpg", cv_img)

def main(args=None):
    import json
    from datetime import date, datetime
    import os, errno

    rclpy.init(args=args)
    node = Node('test')
    
    cam_poses = get_cam_world_pos(node)
    cam_rois = get_cam_pixel_pos(node)

    try:
        date_str = f"{date.today().strftime('%Y-%m-%d')}-{datetime.now().time().strftime('%I%M%S')}"
        calib_dir = f"{data_dir}/calibration-{date_str}/"
        os.makedirs(calib_dir)
        fcalib_data = f"{data_dir}/calibration-{date_str}/calib_data.json"
    
        with open(fcalib_data, "w") as fs:
            data = {"pose": cam_poses, "roi": cam_rois}
            json.dump(data, fs, indent=2)
        
        # if args.save_img:
        save_cam_view(node, calib_dir)
    except OSError as e:
        print(e)
        if e.errno != errno.EEXIST:
            raise
        else:
            print("could not create directory, it already exists")
    rclpy.shutdown()
    

if __name__=="__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # # parser.add_argument("save_img", type=bool)
    # args = parser.parse_args()

    main()
