import numpy as np
from numpy.lib.function_base import vectorize
import roslib
roslib.load_manifest('visnet')
import sys
import rospy
import cv2
from std_msgs.msg import Int16MultiArray
from sensor_msgs.msg import CameraInfo
# from nav_msgs.msg import Odometry
import message_filters as mf

# these are modules custom made for this package
import pfilter_multi as pfilter
import util

class CameraNode:
    def __init__(self, cam_param, cam_pos, cam_att):
        self.param = cam_param
        self.pos = cam_pos
        self.att = cam_att
        self.K = util.get_cam_in(self.param)
        self.R = util.R_model2cam @ util.so3_exp(self.att)
        self.P = util.get_cam_mat_lie(self.param, self.pos, self.att)

    def __repr__(self):
        return "param: {}, pos: {}, att: {}".format(self.param, self.pos, self.att)
        
    def _get_pixel_pos_hom(self, target_pos):
        return self.P @ util.cart2hom(target_pos)
    
    def _get_pixel_pos(self, target_pos):
        return util.hom2cart(self._get_pixel_pos_hom(target_pos))

    def get_observation(self, target_pos):
        bearing = self._get_pixel_pos(target_pos)
        return bearing

class GzbPF():
    """
    A particle filter for Gazebo vision tracking
    """
    
    def __init__(self, n_cam=2):
        # [fx, fy, cx, cy, s] -- (should probably try to get these from gazebo instead, how to handle the update?)
        cam_param = [642.0926159343306, 642.0926159343306, 1000.5, 1000.5, 0]
        cam_poses = np.array([
            [20, 20, 12, 0, 0, -2.2],
            [20, -15, 12, 0, 0, 2.2],
            [-20, -20, 12, 0, 0, 0.7],
            [-20, 20, 12, 0, 0, -0.7],
        ])

        self.cam_nodes = []
        self.msmt_subs = []
        scb_args = []
        for i in range(n_cam):
            camera_name = "camera_"+str(i+1)
            cam_pose = cam_poses[i]
            cam_node = CameraNode(cam_param=cam_param, cam_pos=cam_pose[0:3], cam_att=cam_pose[3:6])
            self.cam_nodes.append(cam_node)
            cb_args = [camera_name, cam_node]
            msmt_sub = mf.Subscriber(camera_name+"_msmt", Int16MultiArray)
            self.msmt_subs.append(msmt_sub)
            
            # cam_info_cb_args = [camera_name, cam_node]
            # cam_info_sub = rospy.Subscriber(camera_name+"/"+camera_name+"_info", CameraInfo, self.cam_info_callback, cam_info_cb_args)
            # self.cam_info_subs.append(cam_info_sub)
        
        ts = mf.TimeSynchronizer(self.msmt_subs, queue_size=10)
        ts.registerCallback(self.synced_callback, scb_args)

    
    # def callback(self, data, cb_args):
    #     cam_name = cb_args[0]
    #     print(cam_name)


    def synced_callback(self, scb_data, scb_args):
        print(len(scb_data))
        for arg in scb_args:
            print(arg[0])

    def cam_info_callback(self, data, cb_args):
        print(type(data), data.K)
        pass
    
    # def observe_fn(x):
    #     """
    #     Parameters:
    #         x : all particles at a step
    #     Returns:
    #         observations/measurements from x using the camera projection matrix
    #     """
    #     msmt = []
    #     for xi in x:
    #         vec = []
    #         for node in cam_nodes:
    #             p1 = node.get_measurement(xi[0:3])
    #             p2 = node.get_measurement(xi[6:9])
    #             vec = np.hstack([vec,p1, p2])
    #         msmt.append(vec)
    #     return np.array(msmt)/2000.0

    # def dynamics_fn(x):
    #     """
    #     Parameters:
    #         x : all particles at a time step
    #     Returns:
    #         x1: propagated particles
    #     """
        
    #     dt = 1/10
    #     A = np.block([
    #         [np.eye(3), dt*np.eye(3)],
    #         [np.zeros((3,3)), np.eye(3)]
    #     ])
    #     w = np.block([np.zeros(3), np.random.normal(0, 1, 3)])
    #     n = x.shape[0]
    #     A_block = np.block([[A, np.zeros((6,6))], [np.zeros((6,6)), A]])
    #     w = np.zeros((12, n))
    #     w[3:6, :] = np.random.normal(0, 1, (3, n))
    #     w[9:12, :] = np.random.normal(0, 1, (3,n))
    #     x1 = A_block @ x.T + w

    #     return x1.T

    # def prior_fn(n):
    #     pos_x = np.random.uniform(-20, 20, (n,1))
    #     pos_y = np.random.normal(-20, 20, (n,1))
    #     pos_z = np.random.normal(10, 25, (n,1))
    #     vel_x = np.random.normal(0, 1, (n,1))
    #     vel_y = np.random.normal(0, 1, (n,1))
    #     vel_z = np.zeros((n,1))
    #     particles = np.hstack([pos_x, pos_y, pos_z, vel_x, vel_y, vel_z])
    #     # particles = np.hstack([particles,particles])
    
    #     return particles

def main():
    gzb_pf = GzbPF(n_cam=2)

    rospy.init_node('particle_filter')

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shuttin down...")

if __name__=="__main__":
    main()