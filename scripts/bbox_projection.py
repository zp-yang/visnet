#!/usr/bin/env python3
from inspect import getmembers
import numpy as np
import camera
import sys, os

import roslib
roslib.load_manifest('visnet')
import rospy
import cv2
from visnet.msg import CamMsmt
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from gazebo_msgs.srv import GetModelState, GetModelStateRequest

class BboxProjector:
    def __init__(self, n_cam, cam_params, cam_poses, model_name, end_time=-1, view=False, record=False) -> None:
        self.bridge = CvBridge()
        self.n_cam = n_cam
        self.view = view
        self.record = record
        self.end_time = end_time
        
        self.model = GetModelStateRequest()
        self.model.model_name = model_name
        # self.model.relative_entity_name = "world"

        self.cameras = []

        self.img_subs = {}
        self.start_time = rospy.get_rostime()
        self.unreg_status = {}
        rospy.wait_for_service("/gazebo/get_model_state")
        self.get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

        for i in range(n_cam):
            camera_name = "camera_"+str(i)
            cam = camera.Camera(cam_param=cam_params, cam_pos=cam_poses[i][0:3], cam_att=cam_poses[i][3:6])
            self.cameras.append(cam)

            cb_args = [camera_name, cam]

            img_sub = rospy.Subscriber(camera_name + "/image_raw", Image, self.img_sub_callback, cb_args)
            self.img_subs[camera_name] = img_sub
            self.unreg_status[camera_name] = 0
    
    def img_sub_callback(self, data, cb_args):
        cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        camera_name = cb_args[0]
        cam = cb_args[1]
        this_sub = self.img_subs[camera_name]

        # TODO: Test projection and check if images needed to be processed when using real cameras (cv_img)
        
        model_state = self.get_model_state(self.model)
        model_pos = np.array([model_state.pose.position.x, model_state.pose.position.y, model_state.pose.position.z])

        pixel_pos = cam._get_pixel_pos(model_pos.reshape(3,1)).reshape(2,)
        pixel_pos = pixel_pos.astype("int16")
        
        x, y = pixel_pos
        w, h = (50, 50)

        start_pt = (x-w//2, y-h//2)
        end_pt = (x+w//2, y+h//2)
        
        cropped_img = cv_img[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0]]
        
        elapsed = (rospy.get_rostime() - self.start_time).to_sec()
        
        if self.record is True:
            self.save_img(cropped_img, camera_name+"/"+str(elapsed)+".jpg")
        
        if self.view:
            cv2.imshow("cropped", cropped_img)

            cv2.rectangle(cv_img, start_pt, end_pt, (0, 0, 255), 2)
            cv2.putText(cv_img, "x", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255))
            cv2.imshow("Projection", cv_img)
            
            k = cv2.waitKey(3) & 0xff
        
        
        if elapsed > self.end_time and self.end_time > 0:
            this_sub.unregister()
            self.unreg_status[camera_name] = 1
            print("unregistered {} in callback".format(camera_name))
            if sum(self.unreg_status.values()) >= self.n_cam:
                rospy.signal_shutdown("reached time limit")
    
    def save_img(self, img, img_name):
        data_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), os.pardir)) + "/data/" 
        cv2.imwrite(data_dir+img_name, img)


    def set_start_time(self, time):
        self.start_time = time

def main(args):
    rospy.init_node("bbox_projection")
    print(rospy.get_rostime())
    cam_params = [642.0926, 642.0926, 1000.5, 1000.5,0]
    cam_poses = np.array([
        [20, 20, 12, 0, 0, -2.2],
        [20, -15, 12, 0, 0, 2.2],
        [-20, -20, 12, 0, 0, 0.7],
        [-20, 20, 12, 0, 0, -0.7],
    ])
    
    model_name = "drone_1"
    print(rospy.get_rostime())
    bp = BboxProjector(n_cam=1, 
                        cam_params=cam_params, 
                        cam_poses=cam_poses, 
                        model_name=model_name,
                        view=True)
    bp.set_start_time(rospy.get_rostime())

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down...")
    cv2.destroyAllWindows()

if __name__=="__main__":
    main(sys.argv)
