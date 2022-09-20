import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point32, TransformStamped
from sensor_msgs.msg import CompressedImage, Image, PointCloud, ChannelFloat32
from nav_msgs.msg import Path
from tf2_ros import TransformBroadcaster

import numpy as np
import os
data_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), os.pardir)) + "/data/" 
info_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), os.pardir)) + "/camera_info/"
import cv2
import json
import yaml
from camera import Camera, CamGroup

from cv_bridge import CvBridge
from visnet.msg import CamMsmt
import util
from track import Track


cam_names = [
    "camera_0",
    "camera_1",
    # "camera_2",
    "camera_3",
    ]

calib_data = {}
with open(f"{data_dir}/calibration/calib_data.json", "r") as fs:
    calib_data = json.load(fs)
cam_poses_dict = calib_data["pose"]
cam_poses = np.array([val for val in cam_poses_dict.values()])

cams = []
cams_dist = []
cams_pinhole_K = []
cam_params = []

for i, cam_name in enumerate(cam_names):
    cam_name = cam_names[i]

    cam_info = {}
    with open(f"{info_dir}/{cam_name}.yaml", 'r') as fs: 
        cam_info = yaml.safe_load(fs)

    cam_pinhole_K = np.array(cam_info["camera_matrix"]["data"])
    cam_pos = np.array(cam_poses_dict[cam_name][0:3])
    cam_att = np.array(cam_poses_dict[cam_name][3:6])
    cam_dist = np.array(cam_info["distortion_coefficients"]["data"])
    cam_K, dist_valid_roi = cv2.getOptimalNewCameraMatrix(np.array(cam_pinhole_K).reshape(3,3), np.array(cam_dist), (1600,1200), 1, (1600, 1200))
    
    cams_pinhole_K.append(cam_pinhole_K)
    cams_dist.append(cam_dist)
    cams.append(Camera(cam_K.reshape(-1), cam_pos, cam_att))
    cam_params.append(cam_K.reshape(-1))

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


class Projector(Node):
    def __init__(self) -> None:
        super().__init__('bbox_projector')
        self.target_pos = None
        self.bridge = CvBridge()
        self.cams = []
        self.cam_group = CamGroup(cam_params, cam_poses)


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

        self.tracks = []
        self.cam_nodes = []
        self.msmt_subs = []
        self.paths = [[],[]]
        self.path_buffer_len = 200

        pose_msg: PoseStamped = wait_for_message(self, PoseStamped, "/qualisys/hb1/pose")
        x0_1 = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z])
        n_particles = 1000
        print(f'initial position {x0_1}')

        track1 = Track(n_particles=n_particles, x0=x0_1, label=1)
        self.tracks.append(track1)

        self.br = TransformBroadcaster(self)
        path_pub_0 = self.create_publisher(Path,"/drone_0/path", 10)
        cloud_pub_0 = self.create_publisher(PointCloud, "/drone_0/cloud", 10)

        self.path_pubs = [path_pub_0]
        self.cloud_pubs = [cloud_pub_0]

    def timer_cb(self):
        z = []

        for i, name in enumerate(cam_names):
            cv_img = self.cam_imgs[name]
            if cv_img is not None and self.pos[0] is not None:
                pix = cams[i]._get_pixel_pos(np.array([self.pos[0]]).reshape(3,1)).reshape(-1).astype(np.int32)
                cv_img = cv2.undistort(cv_img, cams_pinhole_K[i].reshape(3,3), cams_dist[i], None, cams[i].K)
                cv_img = cv2.rectangle(cv_img, pix-[25,25], pix+[25,25], (0,10,255), 3)
                x, y, w, h = dist_valid_roi
                # msg_ = self.bridge.cv2_to_imgmsg(cv_img[y:y+h, x:x+w], encoding="bgr8")
                # self.img_pubs[m-1].publish(msg_)
                msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
                self.img_pubs[name].publish(msg)
                z.append(np.array([[pix[0], pix[1], 1]]))
        
        if len(z) == len(cam_names):
            self.run_pf(z)


    def run_pf(self, z):       
        mean_states = np.array([track.mean_state[0:3] for track in self.tracks])
        mean_hypo = self.cam_group.get_group_measurement(mean_states, labels=np.array([1]))
        # print(mean_states)
        z_a = []
        for z_m, mean_hypo_m in zip(z, mean_hypo):
            if len(z_m) == 0:
                z_m = np.array([[-1, -1, -1]])
            # print("z_m: {} hypo_m: {}".format(z_m, mean_hypo_m))
            z_a_m, order = util.msmt_association(z_m, mean_hypo_m, sigma=40)
            z_a.append(z_a_m)
        
        for i, track in enumerate(self.tracks):
            msmt = [z_a_m[i] for z_a_m in z_a]
            particles = util.dynamics_d(track.particles)
            hypo = util.observe_fn(self.cam_group, track.particles)
            weights = util.weight_fn(msmt=msmt, hypo=hypo, sigma=40)
            weights = np.nan_to_num(weights, copy=False, nan=0)
            weights = np.clip(weights, 0, 1)
            track.update(weights, particles)

            # print("after update: ", track.mean_state)
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "map"
            t.child_frame_id = "track_{}".format(i)
            t.transform.translation.x = track.mean_state[0]
            t.transform.translation.y = track.mean_state[1]
            t.transform.translation.z = track.mean_state[2]
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0
            self.br.sendTransform(t)

            self.paths[i].append(track.mean_state)
            if len(self.paths[i]) > self.path_buffer_len:
                self.paths[i].pop(0)
            
            path = Path()
            for state in self.paths[i]:
                pose = PoseStamped()
                x = state[0]
                y = state[1]
                z = state[2]
                pose.header.frame_id = 'map'
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = z
                path.poses.append(pose)
            path.header.frame_id = 'map'
            self.path_pubs[i].publish(path)

            points = []
            for p in particles:
                point = Point32()
                point.x = p[0]
                point.y = p[1]
                point.z = p[2]
                points.append(point)
            
            channel = ChannelFloat32()
            channel.name = "intensity"
            channel.values = weights.tolist()

            cloud = PointCloud()
            cloud.header.frame_id = "map"
            cloud.header.stamp = self.get_clock().now().to_msg()
            cloud.points = points
            cloud.channels = [channel]
            self.cloud_pubs[i].publish(cloud)

    def pose_sub_cb(self, msg):
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