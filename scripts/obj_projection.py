import rospy
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

cam_names = [
    "camera_0",
    "camera_1",
    # "camera_2",
    "camera_3",
    ]
i = 0
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

cam = Camera(cam_K.reshape(-1), cam_pos, cam_att)

# IMPORTANT PARAMETER !!!
# Delay from camera to publish to the ros master
CAM_DELAY = 500 # ms?

class Projector():
    def __init__(self) -> None:
        self.target_pos = None
        self.bridge = CvBridge()
        self.pos = np.array([0,0,0])
        target_cb_args = []
        img_cb_args = []
        self.buffer = [None]*5
        target_pos_sub_ = rospy.Subscriber("/qualisys/cf0/pose", PoseStamped, self.target_pos_cb, target_cb_args)
        img_sub_ = rospy.Subscriber(f"{cam_name}/image_raw/compressed/compressed", CompressedImage, self.img_cb, img_cb_args)
        self.img_pub_ = rospy.Publisher(f"/{cam_name}/projected/image_raw", Image, queue_size=2)

    def target_pos_cb(self, data: PoseStamped, args):
        self.pos = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        self.buffer.append(self.pos)
        self.buffer.pop(0)

    def img_cb(self, data: CompressedImage, args):
        cv_img = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        pos = self.buffer[0]
        if pos is not None:
            pix = cam._get_pixel_pos(np.array([pos]).reshape(3,1)).reshape(-1).astype(np.int32)
            # print(pix)
            cv_img = cv2.undistort(cv_img, cam_pinhole_K.reshape(3,3), cam_dist, None, cam_K)
            cv_img = cv2.rectangle(cv_img, pix-[15,15], pix+[15,15], (0,10,255), 3)
            x, y, w, h = dist_valid_roi
            msg_ = self.bridge.cv2_to_imgmsg(cv_img[y:y+h, x:x+w])
            self.img_pub_.publish(msg_)

rospy.init_node("projection")

p = Projector()

rospy.spin()