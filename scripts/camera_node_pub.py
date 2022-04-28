#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

rospy.init_node("camera_streamer")
# /dev/video0
camera = cv2.VideoCapture(0)

if (camera.isOpened()== False):
    print("Error opening video file")

bridge = CvBridge()
img_pub = rospy.Publisher("camera0/image_raw", Image, queue_size=10)
camera.set(cv2.CAP_PROP_FPS, 30.0)
# camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
a = camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
b = camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
print("w / h", a, b)
# Read until video is completed
while(camera.isOpened()):
    ret, frame = camera.read()
    try:
        if ret == True:
            frame = bridge.cv2_to_imgmsg(frame, "bgr8")
            img_pub.publish(frame)
        else:
            print("cannot read from camera...")
            raise Exception
    except KeyboardInterrupt:
        print("ctcl+c...")
        # When everything done, release the video capture object
        camera.release()
        break
    except Exception as e:
        print(e)
        camera.release()
        break