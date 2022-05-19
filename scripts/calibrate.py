import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
# 1. needs to subcribe to camera pose
# 2. needs to select other camera in the frame ROI
# 3. calibrate this camera's orientation using other camera's position and ROI position
# 4. save the new projectin matrix to somewhere ?.yaml?
# Notes: qualisys node should be running
roi = cv2.selectROI()