#!/usr/bin/env python3
import numpy as np

import roslib
import tf2_py
roslib.load_manifest('visnet')
import rospy
from visnet.msg import CamMsmt
import message_filters as mf

# these are modules custom made for this package
import pfilter_multi as pfilter
import util

import camera
from track import Track

import tf2_ros
import geometry_msgs.msg
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point32
from sensor_msgs.msg import PointCloud, ChannelFloat32


class GzbPF():
    """
    A particle filter for Gazebo vision tracking
    """
    
    def __init__(self, n_cam=4):
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
        self.paths = [[]]
        self.path_buffer_len = 200

        x0_1 = np.array([-20, -5, 20])
        x0_2 = np.array([20, 5, 20])
        
        n_particles = 5000

        track1 = Track(n_particles=n_particles, x0=x0_1, label=1)
        self.tracks.append(track1)

        for i in range(n_cam):
            camera_name = "camera_"+str(i)
            msmt_sub = mf.Subscriber(camera_name+"_msmt", CamMsmt)
            self.msmt_subs.append(msmt_sub)
            
        ts = mf.TimeSynchronizer(self.msmt_subs, queue_size=10)
        ts.registerCallback(self.synced_callback)

        self.br = tf2_ros.TransformBroadcaster()
        self.path_pub = rospy.Publisher("/drone_0/path", Path, queue_size=10)
        self.cloud_pub = rospy.Publisher("/drone_0/cloud", PointCloud, queue_size=10)

    # This is where we process particle fiters
    def synced_callback(self, *args):
        z = []
        ct = 0
        for m, data in enumerate(args):
            ct += 1
        
            labels = np.array(data.labels)
            msmts = np.array(data.msmts)
            msmts = msmts.reshape(int(msmts.shape[0]/4), 4)

            x = msmts[:,0] + 0.5*msmts[:,2]
            y = msmts[:,1] + 0.5*msmts[:,3]
            a = msmts[:,2] * msmts[:,3]
            msmts = np.vstack([x, y, labels]).T

            drone_indices = np.where(labels==1)[0]
            bird_indices = np.where(labels==0)[0]
            blob_indices = np.where(labels==-1)[0]
            
            drone_msmts = msmts[drone_indices, :]
            bird_msmts = msmts[bird_indices, :]
            blob_msmts = msmts[blob_indices, :]

            n_drone = drone_indices.shape[0]
            n_bird = bird_indices.shape[0]
            n_blob = blob_indices.shape[0]
            
            track_msmts = np.vstack([drone_msmts, blob_msmts])
            z.append(track_msmts)
        
        mean_states = np.array([track.mean_state[0:3] for track in self.tracks])
        mean_hypo = self.cam_group.get_group_measurement(mean_states, labels=np.array([1]))

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
            weights = np.clip(weights, 0, 1)
            track.update(weights, particles)

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
            cloud.points = points
            cloud.channels = [channel]
            self.cloud_pub.publish(cloud)

            # print("after update: ", track.mean_state)
            t = geometry_msgs.msg.TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "map"
            t.child_frame_id = "track_{}".format(i)
            t.transform.translation.x = track.mean_state[0]
            t.transform.translation.y = track.mean_state[1]
            t.transform.translation.z = track.mean_state[2]
            t.transform.rotation.x = 0
            t.transform.rotation.y = 0
            t.transform.rotation.z = 0
            t.transform.rotation.w = 1
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
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = z
                path.poses.append(pose)
            path.header.frame_id = 'map'
            self.path_pub.publish(path)



    
def main():
    rospy.init_node('particle_filter')
    gzb_pf = GzbPF(n_cam=4)

    

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shuttin down...")

if __name__=="__main__":
    main()