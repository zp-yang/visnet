#!/usr/bin/env python3

# from io import open_code
import numpy as np
from numpy.linalg import pinv
from genpy.rostime import Duration
import rospy 
import rospkg 
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
import geometry_msgs.msg
import tf2_ros
from concurrent.futures import ThreadPoolExecutor

def target_traj_circle(t, begin, end, duration):
    x = 15*np.sin(2 * np.pi * 0.01 * t / 3)
    y = 10*np.cos(2 * np.pi * 0.01 * t / 3)
    z = 20
    return np.array([x,y,z])

def target_traj_straight(t, begin, end, duration):
    max_dist = np.linalg.norm(begin-end)
    v_max = (end-begin) / duration
    pos = begin + v_max * t

    if t > duration:
        pos = end

    return pos

def set_drone_state(*args):
    args = args[0]

    model_name = args[0]
    traj_fn = args[1]
    begin = args[2]
    end = args[3]
    duration = args[4]

    br = tf2_ros.TransformBroadcaster()
    state_msg_0 = ModelState()
    state_msg_0.model_name = model_name
    state_msg_0.pose.position.x = begin[0]
    state_msg_0.pose.position.y = begin[1]
    state_msg_0.pose.position.z = begin[2]
    state_msg_0.pose.orientation.x = 0
    state_msg_0.pose.orientation.y = 0
    state_msg_0.pose.orientation.z = 0
    state_msg_0.pose.orientation.w = 1

    rospy.wait_for_service('/gazebo/set_model_state')
    start = rospy.get_rostime()
    elapsed = 0
    end_time = 60
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
    # while elapsed <= end_time:
        now = rospy.get_rostime()
        elapsed = (now - start).to_sec()
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            # state_msg_0.pose.position.x = 10*np.sin(2*np.pi*elapsed*0.05)
            # state_msg_0.pose.position.y = 15*np.cos(2*np.pi*elapsed*0.05)
            pos = traj_fn(elapsed, begin, end, duration)
            # print(elapsed, pos)
            state_msg_0.pose.position.x = pos[0]
            state_msg_0.pose.position.y = pos[1]
            state_msg_0.pose.position.z = pos[2]

            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( state_msg_0 )
            t = geometry_msgs.msg.TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "map"
            t.child_frame_id = model_name
            t.transform.translation.x = state_msg_0.pose.position.x
            t.transform.translation.y = state_msg_0.pose.position.y
            t.transform.translation.z = state_msg_0.pose.position.z
            t.transform.rotation.x = state_msg_0.pose.orientation.x
            t.transform.rotation.y = state_msg_0.pose.orientation.y
            t.transform.rotation.z = state_msg_0.pose.orientation.z
            t.transform.rotation.w = state_msg_0.pose.orientation.w
            br.sendTransform(t)

        except rospy.ServiceException as e:
            print("Service call failed: {:s}".format(str(e)))
        rate.sleep()
    pass

def main():
    rospy.init_node('set_pose')
    x0_1 = np.array([-20, -5, 20])
    x0_2 = np.array([20, 5, 20])

    executor_args = [
        ["drone_0", target_traj_straight, x0_1, x0_1+[40,0,0], 30], 
        ["drone_1", target_traj_straight, x0_2, x0_2+[-40,0,0], 30],
        ]
    
    with ThreadPoolExecutor(max_workers=2) as tpe:
        tpe.map(set_drone_state, executor_args)

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
