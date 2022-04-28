import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def img_cb(data, args):
    bridge: CvBridge = args[0]
    cv_img = bridge.imgmsg_to_cv2(data)
    print(cv_img.shape)

bridge= CvBridge()
cb_args = [bridge]

img_sub = rospy.Subscriber("/cv_camera/image_raw", Image, callback=img_cb, callback_args=cb_args)

rospy.init_node("yeet")
rospy.spin()