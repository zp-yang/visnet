import cv2
import sys
import json
# import labelme
import base64
import os
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr

from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

import rclpy

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
#rename to bag file that you want to make into images

data_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), os.pardir)) + "/data/" 
print(data_dir)
bridge: CvBridge = CvBridge()

if __name__ == '__main__' : 
    # Define an initial bounding box
    counter = 0
    a=1017 #horizontal
    b=938 #vertical
    c=40 #to the right
    d=22 #down
    bbox = (a, b, c, d)

    cam_name = "camera_3"

    with Reader('/home/zpyang/rosbags/multicam-09-09-172453') as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == f'/{cam_name}/camera/image_raw/compressed':
                msg = deserialize_cdr(rawdata, connection.msgtype)
                frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")                
                
                # print(counter) #frame counter
                if counter % 10 == 0:
                    filename = f"{data_dir}/{cam_name}/frame{counter:04d}" #Directory to send images to

                    cv2.imwrite(filename + '.jpg', frame)

                    # Display result
                    cv2.imshow("tracking",frame)

                    # Exit if ESC pressed
                    k = cv2.waitKey(1) & 0xff
                    if k == 27 : break
                counter = counter + 1