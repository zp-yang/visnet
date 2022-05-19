import numpy as np
import cv2
import os
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

img_path = "/home/zpyang/Downloads/crazyflie_example.jpg"

image = cv2.imread(img_path)


rospy.init_node("ass")
img_pub = rospy.Publisher("test/image_raw", Image, queue_size=1)
bridge = CvBridge()
img_msg = bridge.cv2_to_imgmsg(image, "bgr8")
while not rospy.is_shutdown():
    img_pub.publish(img_msg)    
#     cv2.imshow("Detection Result", image)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()
