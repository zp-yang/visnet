#!/usr/bin/env python3
import numpy as np
from numpy.core.arrayprint import printoptions
from numpy.linalg import pinv
from concurrent.futures import ThreadPoolExecutor

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Pose
import tf2_ros

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

def target_traj_circle(t, begin, end, duration):
    theta = 2 * np.pi * 0.1 * t / 3 + np.arctan2(begin[1],begin[0])
    x = 19*np.sin(theta)
    y = 14*np.cos(theta)
    z = begin[2]
    yaw = -(theta + np.pi/2)
    pos = np.array([x, y, z])
    return np.array([x,y,z, 0, 0, yaw])

def target_stationary(t, begin):
    return begin

def target_traj_straight(t, begin, end, duration):
    trip = int(t / duration)

    if (trip % 2): # odd trip
        temp = begin
        begin = end
        end = temp

    max_dist = np.linalg.norm(begin-end)
    v_max = (end-begin) / duration
    pos = begin + v_max * (t - trip*duration)

    att = np.array([0,0,0])
    # return pos
    return np.concatenate([pos, att])

def target_traj_f_file(t, begin):
    pass

class TargetTraj(Node):
    """
    Move a target entity(gazebo model) along a set trajectory defined by traj_f
    traj_f should always take @t and @begin_pose as the first two arguments
    """
    def __init__(self, target_name=None, traj_f=None, begin_pose=np.zeros(6, dtype=np.float64), *traj_args) -> None:
        super().__init__(f"{target_name}_trajectory")
        self.client = self.create_client(SetEntityState, "/set_entity_state")
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('gazebo set_entity_state service is not availabel, waiting...')
        
        self.target_name = target_name
        self.traj_f = traj_f
        self.begin_pose = begin_pose
        self.traj_args = traj_args

        timer_period = 0.01
        self.timer = self.create_timer(timer_period, self.traj_callback)
        self.elapsed = 0
        self.time_last = self.get_clock().now()

        print(f"setting state for {self.target_name}")
        self.state_msg = EntityState()
        self.state_msg.name = self.target_name
        # state_msg.reference_frame = "world" # default frame

        self.state_msg.pose.position.x = self.begin_pose[0]
        self.state_msg.pose.position.y = self.begin_pose[1]
        self.state_msg.pose.position.z = self.begin_pose[2]
    
        print(f"set initial states for {self.target_name}")

        self.request = SetEntityState.Request()

    def traj_callback(self):
        self.elapsed += (self.get_clock().now()-self.time_last).nanoseconds / 1e9
        self.time_last = self.get_clock().now()

        _pose = self.traj_f(self.elapsed, self.begin_pose, *self.traj_args)

        self.state_msg.pose.position.x = _pose[0]
        self.state_msg.pose.position.y = _pose[1]
        self.state_msg.pose.position.z = _pose[2]

        _q = quaternion_from_euler(_pose[3], _pose[4], _pose[5])
        self.state_msg.pose.orientation.x = _q[0]
        self.state_msg.pose.orientation.y = _q[1]
        self.state_msg.pose.orientation.z = _q[2]
        self.state_msg.pose.orientation.w = _q[3]
        
        self.request.state = self.state_msg
        future = self.client.call_async(self.request)
        if future.done():
            response = future.result()
            print("response: " + response)

def main(args=None):
    rclpy.init(args=args)
    executor = rclpy.get_global_executor()

    x0_1 = np.array([-20, -4, 20, 0, 0, 0], dtype=np.float64)
    x0_2 = np.array([20, 4, 20, 0, 0, 0], dtype=np.float64)

    traj_1 = TargetTraj("drone_0", target_traj_straight, x0_1, x0_1+[40,0,0,0,0,0], 30)
    traj_2 = TargetTraj("drone_1", target_traj_straight, x0_2, x0_2+[-40,0,0,0,0,0], 8)

    executor.add_node(traj_1)
    executor.add_node(traj_2)
    
    executor.spin()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
