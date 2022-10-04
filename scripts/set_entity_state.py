#!/usr/bin/env python3
from pydoc import cli
from telnetlib import RCP
from tkinter.messagebox import NO
import numpy as np
from numpy.core.arrayprint import printoptions
from numpy.linalg import pinv
from concurrent.futures import ThreadPoolExecutor

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
import geometry_msgs.msg
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
    # return pos
    return np.array([x,y,z, 0, 0, yaw])

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

class TargetTraj():
    def __init__(self, client, target_name, start, traj_f=None, *traj_args) -> None:
        self.client = client
        self.target_name = target_name
        self.traj_f = traj_f
        self.traj_args = traj_args

    def run(self):
        pass

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("set_target_traj")
    client = node.create_client(SetEntityState, "/set_entity_state")
    print("waiting for service")
    while not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('gazebo set_entity_state service is not availabel, waiting...')

    model_name = 'drone_0'
    begin = [0.0, 0.0, 15.0]

    state_msg = EntityState()
    state_msg.name = model_name
    # state_msg.reference_frame = "world" # default frame

    state_msg.pose.position.x = begin[0]
    state_msg.pose.position.y = begin[1]
    state_msg.pose.position.z = begin[2]
    state_msg.pose.orientation.x = 0.
    state_msg.pose.orientation.y = 0.
    state_msg.pose.orientation.z = 0.
    state_msg.pose.orientation.w = 1.

    request = SetEntityState.Request()
    
    while rclpy.ok():        
        request.state = state_msg
        future = client.call_async(request)
        print(future.done())
        if future.done():
            response = future.result()
            print("response: " + response)
        rclpy.spin_once(node=node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
