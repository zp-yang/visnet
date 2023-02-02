#!/usr/bin/env python3
import numpy as np

from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped

import numpy as np

import rclpy
from rclpy.node import Node

from tf2_ros import TransformBroadcaster

frame_names = [
    'camrea_0',
    'camera_1',
    # 'camera_2',
    'camera_3',
    'hb1',
]

def quaternion_from_euler(ai, aj, ak):
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = np.cos(ai)
    si = np.sin(ai)
    cj = np.cos(aj)
    sj = np.sin(aj)
    ck = np.cos(ak)
    sk = np.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    q[0] = cj*sc - sj*cs
    q[1] = cj*ss + sj*cc
    q[2] = cj*cs - sj*sc
    q[3] = cj*cc + sj*ss

    return q


class FramePublisher(Node):

    def __init__(self, frame_name):
        super().__init__(f'{frame_name}_frame_publisher')

        # Declare and acquire `turtlename` parameter
        self.frame_name = frame_name
        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribe to a turtle{1}{2}/pose topic and call handle_turtle_pose
        # callback function on each message
        self.subscription = self.create_subscription(
            PoseStamped,
            f'/{self.frame_name}/pose',
            self.pose_cb,
            1)

    def pose_cb(self, msg: PoseStamped):
        t = TransformStamped()

        # Read message content and assign it to
        # corresponding tf variables
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = self.frame_name

        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z

        t.transform.rotation.x = msg.pose.orientation.x
        t.transform.rotation.y = msg.pose.orientation.y
        t.transform.rotation.z = msg.pose.orientation.z
        t.transform.rotation.w = msg.pose.orientation.w

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    executor = rclpy.get_global_executor()
    
    for name in frame_names:
        node = FramePublisher(name)
        executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()

if __name__=="__main__":
    main()