import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
import rospy
import cv2

import os
data_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), os.pardir)) + "/data/" 

def img_cb(data, args):
    cam_name = args[0]
    bridge: CvBridge = args[1]
    start_time = args[2]
    # ..... args[1]
    elapsed = (rospy.get_rostime() - start_time).to_sec()
    print(cam_name)
    cv_img = bridge.compressed_imgmsg_to_cv2(data, "bgr8")

    elapsed_condition = 1
    if elapsed_condition: 
        filename = f"{data_dir}/saved_images/{cam_name}_{elapsed}.jpg"
        cv2.imwrite(filename, cv_img)

    # cv2.imshow("image", cv_img)
    # k = cv2.waitKey(3) & 0xff

def main():
    rospy.init_node("save_images")
    cam_names = [
        "camera_0",
        "camera_1",
        # "camera_2",
        "camera_3",
    ]
    i = 0
    bridge = CvBridge()

    start_time = rospy.get_rostime()

    img_cb_args = [cam_names[i], bridge, start_time]
    img_sub_ = rospy.Subscriber(f"{cam_names[i]}/image_raw/compressed/compressed", CompressedImage, img_cb, img_cb_args)
    rospy.spin()

    print("finished subscription ...")

if __name__=="__main__":
    main()