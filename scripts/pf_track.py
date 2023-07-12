#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from vision_msgs.msg import BoundingBox2D
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Path
import sensor_msgs_py.point_cloud2 as pc2

import numpy as np
from functools import partial
import time, os, yaml, json, cv2

from lienp.SE3 import SE3
import pf_utils
from pf_utils import TargetTrack
from camera import CamGroup
cam_names = [
    "camera0",
    "camera1",
    "camera2",
    "camera3",
    ]
info_dir = '../camera_info'
with open(f'{info_dir}/camera_mocap_poses.json', 'r') as fs:
    camera_mocap_poses = json.load(fs)

cam_poses = []
cam_params = []
for name in cam_names:
    with open(f'{info_dir}/{name}.yaml', 'r') as fs:
        cam_info = yaml.safe_load(fs)
    with open(f"{info_dir}/{name}_mocap_calib.json") as fs:
        calib_info = json.load(fs)
    R_rel = np.array(calib_info['R'])
    c_rel = np.array(calib_info['c'])
    T_rel = SE3(R=R_rel, c=c_rel)
    T_mocap = np.array(camera_mocap_poses[name])
    T_cam = T_rel @ T_mocap
    K = np.array(cam_info["camera_matrix"]["data"])
    cam_dist = np.array(cam_info["distortion_coefficients"]["data"])
    K_opt, dist_valid_roi = cv2.getOptimalNewCameraMatrix(K.reshape(3,3), cam_dist, (1600,1200), 1, (1600, 1200))
    cam_poses.append(T_cam)
    cam_params.append(K_opt)

class Tracker(Node):
    def __init__(self):
        super().__init__('pf_tracker')

        self.bbox_subs_ = {name: None for name in cam_names}
        self.bbox_msmts_ = {name: None for name in cam_names}
        self.tracks = []

        self.paths = [[]]
        self.path_buffer_len = 50

        path_pub_0 = self.create_publisher(Path,"/drone1/path", 10)
        # cloud_pub_0 = self.create_publisher(PointCloud, "/drone1/cloud", 10)
        cloud_pub_0 = self.create_publisher(PointCloud2, "/drone1/cloud", 10)
        for name in cam_names:
            self.bbox_subs_[name] = self.create_subscription(
                BoundingBox2D,
                f'/{name}/bbox',
                partial(self.bbox_sub_cb, cam_name=name),
                10
            )
        self.cam_group = CamGroup(cam_params, cam_poses)
        time.sleep(1)
        self.tracks.append(TargetTrack(n_particles=1000, x0=np.zeros(3), label=1))
        self.path_pubs = [path_pub_0]
        self.cloud_pubs = [cloud_pub_0]
        self.pf_timer = self.create_timer(1/10.0, self.timer_cb)

    def bbox_sub_cb(self, msg: BoundingBox2D, cam_name):
        self.bbox_msmts_[cam_name] = np.array([[msg.center.position.x, msg.center.position.y, 1]])

    def timer_cb(self):
        # self.get_logger().info(f'{self.bbox_msmts_}')
        z = [msmt for msmt in self.bbox_msmts_.values()]
        # if z[0] is not None:
        self.run_pf(z)

    def run_pf(self, z):
        mean_states = np.array([track.mean_state[0:3] for track in self.tracks])
        mean_hypo = self.cam_group.get_group_measurement(mean_states, labels=np.array([1]))
        # print(mean_states)
        z_a = []
        for z_m, mean_hypo_m in zip(z, mean_hypo):
            if z_m is None:
                z_m = np.array([[-1, -1, -1]])
            # print("z_m: {} hypo_m: {}".format(z_m, mean_hypo_m))
            z_a_m, order = pf_utils.msmt_association(z_m, mean_hypo_m, sigma=40)
            z_a.append(z_a_m)
        
        for i, track in enumerate(self.tracks):
            msmt = [z_a_m[i] for z_a_m in z_a]
            particles = pf_utils.dynamics_d(track.particles, sigma=0.5)
            hypo = pf_utils.observe_fn(self.cam_group, track.particles)
            weights = pf_utils.weight_fn(msmt=msmt, hypo=hypo, sigma=40)
            weights = np.nan_to_num(weights, copy=False, nan=0)
            weights = np.clip(weights, 0, 1)
            track.update(weights, particles)

            self.paths[i].append(track.mean_state)
            if len(self.paths[i]) > self.path_buffer_len:
                self.paths[i].pop(0)
            
            path = Path()
            for state in self.paths[i]:
                pose = PoseStamped()
                x = state[0]
                y = state[1]
                z = state[2]
                pose.header.frame_id = 'qualisys'
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = z
                path.poses.append(pose)
            path.header.frame_id = 'qualisys'
            self.path_pubs[i].publish(path)

            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='weight', offset=16, datatype=PointField.FLOAT32, count=1)
            ]
            pc = np.hstack([particles[:,0:3],weights.astype(np.float32).reshape(1000,1)])

            header = Header()
            header.frame_id = 'qualisys'
            header.stamp = self.get_clock().now().to_msg()
            cloud = pc2.create_cloud(header,fields, pc)
            self.cloud_pubs[i].publish(cloud)

def main():
    rclpy.init(args=None)

    pf_tracker = Tracker()

    rclpy.spin(pf_tracker)

    pf_tracker.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()