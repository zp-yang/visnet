import rclpy
from rclpy.node import Node

import numpy as np
import os
data_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), os.pardir)) + "/data/" 
info_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), os.pardir)) + "/camera_info/"
import cv2
import json
import yaml
from camera import Camera
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge

from visnet.msg import CamMsmt

cam_names = [
    "camera_0",
    "camera_1",
    # "camera_2",
    "camera_3",
    ]

cams = []
cams_dist = []
cams_pinhole_K = []

for i, cam_name in enumerate(cam_names):
    cam_name = cam_names[i]

    calib_data = {}
    with open(f"{data_dir}/calibration/calib_data.json", "r") as fs:
        calib_data = json.load(fs)
    cam_poses = calib_data["pose"]

    cam_info = {}
    with open(f"{info_dir}/{cam_name}.yaml", 'r') as fs: 
        cam_info = yaml.safe_load(fs)

    cam_pinhole_K = np.array(cam_info["camera_matrix"]["data"])
    cam_pos = np.array(cam_poses[cam_name][0:3])
    cam_att = np.array(cam_poses[cam_name][3:6])
    cam_dist = np.array(cam_info["distortion_coefficients"]["data"])
    cam_K, dist_valid_roi = cv2.getOptimalNewCameraMatrix(np.array(cam_pinhole_K).reshape(3,3), np.array(cam_dist), (1600,1200), 1, (1600, 1200))
    
    cams_pinhole_K.append(cam_pinhole_K)
    cams_dist.append(cam_dist)
    cams.append(Camera(cam_K.reshape(-1), cam_pos, cam_att))

class Projector(Node):
    def __init__(self) -> None:
        super().__init__('bbox_projector')
        self.target_pos = None
        self.bridge = CvBridge()
        self.cams = []

        self.pos = [None]*3
        self.cam_imgs = {name: None for name in cam_names}

        self.pose_sub_ = self.create_subscription(
            PoseStamped, 
            '/qualisys/hb1/pose', 
            self.pose_sub_cb,
            10,
            )

        self.img_sub_0 = self.create_subscription(
            Image,
            '/camera_0/image',
            lambda msg: self.img_sub_cb(msg, ["camera_0"]),
            5
        )

        self.img_sub_1 = self.create_subscription(
            Image,
            '/camera_1/image',
            lambda msg: self.img_sub_cb(msg, ["camera_1"]),
            5
        )

        self.img_sub_3 = self.create_subscription(
            Image,
            '/camera_3/image',
            lambda msg: self.img_sub_cb(msg, ["camera_3"]),
            5
        )

        self.timer = self.create_timer(0.01, self.timer_cb)

        self.img_pub_0 = self.create_publisher(
            Image,
            '/camera_0/projected/image',
            10
        )

        self.msmt_pub_ = self.create_publisher(
            CamMsmt,
            '/'
        )

        self.img_pub_1 = self.create_publisher(
            Image,
            '/camera_1/projected/image',
            10
        )

        self.img_pub_3 = self.create_publisher(
            Image,
            '/camera_3/projected/image',
            10
        )
        self.img_pubs = {
            'camera_0': self.img_pub_0,
            'camera_1': self.img_pub_1,
            'camera_3': self.img_pub_3,
        }

    def timer_cb(self):
        for i, name in enumerate(cam_names):
            cv_img = self.cam_imgs[name]
            if cv_img is not None and self.pos[0] is not None:
                pix = cams[i]._get_pixel_pos(np.array([self.pos[0]]).reshape(3,1)).reshape(-1).astype(np.int32)
                cv_img = cv2.undistort(cv_img, cams_pinhole_K[i].reshape(3,3), cams_dist[i], None, cams[i].K)
                cv_img = cv2.rectangle(cv_img, pix-[15,15], pix+[15,15], (0,10,255), 3)
                x, y, w, h = dist_valid_roi
                # msg_ = self.bridge.cv2_to_imgmsg(cv_img[y:y+h, x:x+w], encoding="bgr8")
                # self.img_pubs[m-1].publish(msg_)
                msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
                self.img_pubs[name].publish(msg)        


    def pose_sub_cb(self, msg):
        # self.pos = msg
        # self.pos_pub_.publish(self.pos)
        self.pos.pop(0)
        self.pos.append(np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]))
    
    def img_sub_cb(self, msg, args):
        name = args[0]

        self.cam_imgs[name] = self.bridge.imgmsg_to_cv2(msg)

def main(args=None):
    rclpy.init(args=args)

    a_node = Projector()
    
    rclpy.spin(a_node)

    a_node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()