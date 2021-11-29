import numpy as np
from numpy.lib.function_base import vectorize
import roslib
roslib.load_manifest('visnet')
import sys
import rospy
import cv2
from visnet.msg import CamMsmt
from sensor_msgs.msg import CameraInfo
# from nav_msgs.msg import Odometry
import message_filters as mf

# these are modules custom made for this package
import pfilter_multi as pfilter
import util

import camera


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
        
        self.cam_group = camera.CamGroup(cam_param, cam_poses)
        self.tracks = []
        self.cam_nodes = []
        self.msmt_subs = []
        
        x0_1 = np.array([-20, -5, 20])
        x0_2 = np.array([20, 5, 20])
        tra
        tracks = []
        for i in range(n_cam):
            camera_name = "camera_"+str(i)
            cam_pose = cam_poses[i]
            cam_node = CameraNode(cam_param=cam_param, cam_pos=cam_pose[0:3], cam_att=cam_pose[3:6])
            self.cam_nodes.append(cam_node)
            cb_args = [camera_name, cam_node]
            msmt_sub = mf.Subscriber(camera_name+"_msmt", CamMsmt)
            self.msmt_subs.append(msmt_sub)
            
            # cam_info_cb_args = [camera_name, cam_node]
            # cam_info_sub = rospy.Subscriber(camera_name+"/"+camera_name+"_info", CameraInfo, self.cam_info_callback, cam_info_cb_args)
            # self.cam_info_subs.append(cam_info_sub)
        
        ts = mf.TimeSynchronizer(self.msmt_subs, queue_size=10)
        ts.registerCallback(self.synced_callback)

    
    # def callback(self, data, cb_args):
    #     cam_name = cb_args[0]
    #     print(cam_name)


    def synced_callback(self, *args):
        z = []
        for m, data in enumerate(args):

            labels = np.array(data.labels)
            msmts = np.array(data.msmts)
            if msmts.shape[0] == 0:
                msmts = np.array([-1,-1])
            else:
                msmts = msmts.reshape(int(msmts.shape[0]/4), 4)
                
                x = msmts[:,0] + 0.5*msmts[:,2]
                y = msmts[:,1] + 0.5*msmts[:,3]
                a = msmts[:,2] * msmts[:,3]
                msmts = np.vstack([x,y, a]).T
            z.append(msmts)
            print("camera {}: {}".format(m, msmts))


    # def cam_info_callback(self, data, cb_args):
    #     print(type(data), data.K)
    #     pass
    
    
def main():
    rospy.init_node('particle_filter')
    gzb_pf = GzbPF(n_cam=4)

    

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shuttin down...")

if __name__=="__main__":
    main()