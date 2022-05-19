import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
# 1. needs to subcribe to camera pose
# 2. needs to select other camera in the frame ROI
# 3. calibrate this camera's orientation using other camera's position and ROI position
# 4. save the new projectin matrix to somewhere ?.yaml?
# Notes: qualisys node should be running

cam_names = [
    "camera_0",
    "camera_1",
    "camera_2",
    "camera_3",
    ]
rospy.init_node("calibration")
i = 0
img_msg = rospy.wait_for_message(f"{cam_names[i]}/image_raw/compressed/compressed", CompressedImage)
bridge = CvBridge()
cv_img = bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")
window_name = f"{cam_names[i]} calibration"
cv2.imshow(window_name, cv_img)
roi = cv2.selectROIs(window_name, cv_img, showCrosshair=True)
print(roi)
if cv2.waitKey(0) & 0xFF == ord('q'):
    print("q is pressed")
    cv2.destroyAllWindows()
