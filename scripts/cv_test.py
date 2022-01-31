#!/usr/bin/env python3
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import cv2
import rospy
import sys

import numpy as np
import roslib
roslib.load_manifest('visnet')


class GzbCamViewer:

    def __init__(self, scale=0.5, cam_no='0'):
        self.bridge = CvBridge()
        self.image_sub = []
        self.scale = scale

        for i in cam_no:
            camera_name = "camera_"+i
            cb_args = [camera_name]
            img_sub = rospy.Subscriber("tracked_view_"+i, Image, self.callback, cb_args)
        

    def callback(self, data, cb_args):
        name = cb_args[0]
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            width = int(cv_image.shape[1] * self.scale)
            height = int(cv_image.shape[0] * self.scale)
            dim = (width, height)
            
            # resize image
            resized = cv2.resize(cv_image, dim, interpolation = cv2.INTER_AREA)

            cv2.imshow(name, resized)
            k = cv2.waitKey(3) & 0xff
        except CvBridgeError as e:
            print(e)

        k = cv2.waitKey(3) & 0xff


def main(args):
    scale = float(args[1])
    print(args)
    cam_no = args[2:len(args)]
    print(cam_no)
    rospy.init_node('viewer', anonymous=False)
    gzb_tracker = GzbCamViewer(scale=scale, cam_no=cam_no)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
