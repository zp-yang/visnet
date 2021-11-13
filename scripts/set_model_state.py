#!/usr/bin/env python3

# from io import open_code
import numpy as np
import rospy 
import rospkg 
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
import geometry_msgs.msg
import tf2_ros


def main():
    rospy.init_node('set_pose')

    br = tf2_ros.TransformBroadcaster()

    state_msg = ModelState()
    state_msg.model_name = 'drone'
    state_msg.pose.position.x = 0
    state_msg.pose.position.y = 0
    state_msg.pose.position.z = 20
    state_msg.pose.orientation.x = 0
    state_msg.pose.orientation.y = 0
    state_msg.pose.orientation.z = 0
    state_msg.pose.orientation.w = 1

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
            state_msg.pose.position.x = 10*np.sin(2*np.pi*elapsed*0.01)
            state_msg.pose.position.y = 15*np.cos(2*np.pi*elapsed*0.01)
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( state_msg )
            t = geometry_msgs.msg.TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "map"
            t.child_frame_id = "drone"
            t.transform.translation.x = state_msg.pose.position.x
            t.transform.translation.y = state_msg.pose.position.y
            t.transform.translation.z = state_msg.pose.position.z
            t.transform.rotation.x = state_msg.pose.orientation.x
            t.transform.rotation.y = state_msg.pose.orientation.y
            t.transform.rotation.z = state_msg.pose.orientation.z
            t.transform.rotation.w = state_msg.pose.orientation.w
            br.sendTransform(t)

        except rospy.ServiceException as e:
            print("Service call failed: {:s}".format(str(e)))
        rate.sleep()
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
