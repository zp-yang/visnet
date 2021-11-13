#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('visnet')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class GzbObjTracker:

  def __init__(self, tracker, tracker_name):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("camera_1/image_raw", Image, self.callback)
    self.tracker = tracker
    self.tracker_name = tracker_name

  def callback(self, data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      self.obj_detect(cv_image)
    except CvBridgeError as e:
      print(e)

    # (rows,cols,channels) = cv_image.shape
    # if cols > 60 and rows > 60 :
    #   cv2.circle(cv_image, (50,50), 10, 255)
    
    # cv2.imshow("Image window", cv_image)
    # cv2.waitKey(3)

    # try:
    #   self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    # except CvBridgeError as e:
    #   print(e)

  def obj_detect(self, frame):
    # Define an initial bounding box
    bbox = (1508, 608, 63, 57)
    # Uncomment the line below to select a different bounding box
    # bbox = cv2.selectROI(frame, False)
    print(bbox)
    tracker = self.tracker
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    
    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else :
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Display tracker name on frame
    cv2.putText(frame, self.tracker_name + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(3) & 0xff
    # if k == 27 : break

def main(args):
  # Set up tracker.
  # Instead of MIL, you can also use

  tracker_names = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
  tracker_name = tracker_names[1]

  if tracker_name == 'BOOSTING':
      tracker = cv2.legacy.TrackerBoosting_create()
  if tracker_name == 'MIL':
      tracker = cv2.legacy.TrackerMIL_create()
  if tracker_name == 'KCF':
      tracker = cv2.legacy.TrackerKCF_create()
  if tracker_name == 'TLD':
      tracker = cv2.legacy.TrackerTLD_create()
  if tracker_name == 'MEDIANFLOW':
      tracker = cv2.legacy.TrackerMedianFlow_create()
  if tracker_name == 'GOTURN':
      tracker = cv2.legacy.TrackerGOTURN_create()
  if tracker_name == 'MOSSE':
      tracker = cv2.legacy.TrackerMOSSE_create()
  if tracker_name == "CSRT":
      tracker = cv2.legacy.TrackerCSRT_create()

  gzb_tracker = GzbObjTracker(tracker=tracker, tracker_name=tracker_name)

  rospy.init_node('image_converter', anonymous=False)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)