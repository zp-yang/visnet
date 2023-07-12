import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from lienp.SE3 import SE3
from lienp.SO3 import SO3
import numpy as np

import json

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

def main():
    rclpy.init(args=None)
    # node = Test()
    # rclpy.spin(node)
    # node.destroy_node()
    node = rclpy.create_node('ffff')
    cam_poses = {f'camera{i}': None for i in range(4)}
    for i in range(4):
        msg = wait_for_message(node, PoseStamped, f'/camera{i}/pose')
        pos = msg.pose.position
        ori = msg.pose.orientation
        c = np.array([pos.x, pos.y, pos.z])
        q = np.array([ori.w, ori.x, ori.y, ori.z])
        R = SO3.from_quat(q).T
        T_mocap = SE3(R=R, c=c)
        # print(T_mocap)
        cam_poses[f'camera{i}'] = T_mocap.M.tolist()
    print(cam_poses)
    with open('../camera_info/camera_mocap_poses.json', 'w') as fs:
        json.dump(cam_poses, fs)
    rclpy.shutdown()

if __name__=="__main__":
    main()