#!/usr/bin/env python3
import numpy as np
from numpy.core.arrayprint import printoptions
from numpy.linalg import pinv

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Pose
# import tf2_ros

from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity

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

def target_traj_f_file(t, traj_data):

    reverse = True

    duration = traj_data[-1,0]
    # if duration == 0:
    #     duration = traj_data[0,0]
    trip = int(t / duration)
    t = t - trip*duration


    if (trip % 2):
        traj_data_fliped = np.flip(traj_data, axis=0)
        t_list = duration - traj_data_fliped[:,0]
        x = np.interp(t, t_list, traj_data_fliped[:,1])
        y = np.interp(t, t_list, traj_data_fliped[:,2])
        z = 20.0        
    else:
        x = np.interp(t, traj_data[:,0], traj_data[:,1])
        y = np.interp(t, traj_data[:,0], traj_data[:,2])
        z = 20.0

    return np.array([x, y, z, 0., 0., 0.])

class TargetTraj(Node):
    """
    Move a target entity(gazebo model) along a set trajectory defined by traj_f
    traj_f should always take @t and @begin_pose as the first two arguments
    """
    def __init__(self, target_name=None, traj_f=None, *traj_args) -> None:
        super().__init__(f"set_{target_name}_trajectory")
        # self.client_cb_group = MutuallyExclusiveCallbackGroup()
        # self.timer_cb_group = MutuallyExclusiveCallbackGroup()

        # self.client_cb_group = ReentrantCallbackGroup()
        # self.timer_cb_group = ReentrantCallbackGroup()
        
        self.client = self.create_client(SetEntityPose, "/world/purdue/set_pose")
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('gazebo set_entity_state service is not availabel, waiting...')
        
        self.target_name = target_name
        self.traj_f = traj_f
        # self.begin_pose = begin_pose
        self.traj_args = traj_args

        timer_period = 0.01
        self.timer = self.create_timer(timer_period, self.traj_callback)
        self.elapsed = 0
        self.time_last = self.get_clock().now()

        print(f"setting state for {self.target_name}")
        self.entity = Entity()
        self.entity.name = self.target_name
        self.pose = Pose()
        
        self.request = SetEntityPose.Request()

    def traj_callback(self):
        self.elapsed += (self.get_clock().now()-self.time_last).nanoseconds / 1e9
        self.time_last = self.get_clock().now()

        # _pose = self.traj_f(self.elapsed, self.begin_pose, *self.traj_args)
        _pose = self.traj_f(self.elapsed, *self.traj_args)

        self.pose.position.x = _pose[0]
        self.pose.position.y = _pose[1]
        self.pose.position.z = _pose[2]

        _q = quaternion_from_euler(_pose[3], _pose[4], _pose[5])
        self.pose.orientation.x = _q[0]
        self.pose.orientation.y = _q[1]
        self.pose.orientation.z = _q[2]
        self.pose.orientation.w = _q[3]
        
        self.request.entity = self.entity
        self.request.pose = self.pose
        future = self.client.call_async(self.request)
        if future.done():
            response = future.result()
            print("response: " + response)

def get_traj_data(filename):
    traj_data = np.genfromtxt(filename, delimiter=',') 
    traj_data = traj_data[1:, 0:3]
    return traj_data.astype(np.float64)

def main(args=None):
    import os
    data_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), os.pardir)) + "/data/" 
    rclpy.init(args=args)
    
    # executor = rclpy.get_global_executor()
    executor = MultiThreadedExecutor()
    
    x0_1 = np.array([-40, 5, 20, 0, 0, 0], dtype=np.float64)
    x0_2 = np.array([20, 4, 20, 0, 0, 0], dtype=np.float64)

    # traj_data = get_traj_data(data_dir+"drone_traj.csv")
    # traj_1 = TargetTraj("hb1", target_traj_f_file, traj_data)

    traj_1 = TargetTraj("drone", target_traj_straight, x0_1, x0_1+[80,0,0,0,0,0], 16)
    # traj_2 = TargetTraj("hb2", target_traj_straight, x0_2, x0_2+[-40,0,0,0,0,0], 15)

    executor.add_node(traj_1)
    # executor.add_node(traj_2)
    
    executor.spin()

    try:
        rclpy.shutdown()
    except Exception():
        pass

if __name__ == '__main__':
    main()
