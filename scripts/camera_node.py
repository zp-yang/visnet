#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.srv import SetCameraInfo
from cv_bridge import CvBridge, CvBridgeError
from camera_info_manager import CameraInfoManager

cv_extra_param_dict = {
    "cv_cap_prop_pos_msec": cv2.CAP_PROP_POS_MSEC,
    "cv_cap_prop_pos_avi_ratio": cv2.CAP_PROP_POS_AVI_RATIO,
    "cv_cap_prop_frame_width": cv2.CAP_PROP_FRAME_WIDTH,
    "cv_cap_prop_frame_height": cv2.CAP_PROP_FRAME_HEIGHT,
    "cv_cap_prop_fps": cv2.CAP_PROP_FPS,
    "cv_cap_prop_fourcc": cv2.CAP_PROP_FOURCC,
    "cv_cap_prop_frame_count": cv2.CAP_PROP_FRAME_COUNT,
    "cv_cap_prop_format": cv2.CAP_PROP_FORMAT,
    "cv_cap_prop_mode": cv2.CAP_PROP_MODE,
    "cv_cap_prop_brightness": cv2.CAP_PROP_BRIGHTNESS,
    "cv_cap_prop_contrast": cv2.CAP_PROP_CONTRAST,
    "cv_cap_prop_saturation": cv2. CAP_PROP_SATURATION,
    "cv_cap_prop_hue": cv2. CAP_PROP_HUE,
    "cv_cap_prop_gain": cv2.CAP_PROP_GAIN,
    "cv_cap_prop_gamma": cv2.CAP_PROP_GAMMA,
    "cv_cap_prop_exposure": cv2.CAP_PROP_EXPOSURE,
    "cv_cap_prop_convert_rgb": cv2.CAP_PROP_CONVERT_RGB,
    "cv_cap_prop_rectification": cv2.CAP_PROP_RECTIFICATION,
    "cv_cap_prop_iso_speed": cv2.CAP_PROP_ISO_SPEED,
    "cv_cap_prop_white_balance_u": cv2.CAP_PROP_WHITE_BALANCE_BLUE_U,
    "cv_cap_prop_white_balance_v": cv2.CAP_PROP_WHITE_BALANCE_RED_V,
    "cv_cap_prop_buffersize": cv2.CAP_PROP_BUFFERSIZE,
    "cv_cap_prop_auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
}

class CamNode():
    def __init__(self, cam_name="", device_id=0, url="",
                 namespace="", frame_id="", rescale_cam_info_=False,
                 fps=30.0, width=640, height=480) -> None:
        if cam_name=="":
            self.cam_name = rospy.get_param("cam_name")
        else:
            self.cam_name = cam_name
        
        print("cam_name: {}".format(self.cam_name))
        rospy.init_node(self.cam_name)

        self.device_id = device_id
        self.fps = fps
        self.width = width
        self.height = height
        self.url = url
        rospy.loginfo_once("Node for {} is started".format(self.cam_name))
        
        if namespace=="":
            self.namespace = self.cam_name
        if frame_id=="":
            self.frame_id = self.cam_name
        self.camera = None
        
        self.load_basic_params()

        self.info_manager = CameraInfoManager(
            cname=self.cam_name, 
            url=self.url,
            namespace=self.namespace,
        )
        self.info_manager.loadCameraInfo()
        self.info_ = self.info_manager.getCameraInfo()
        self.rescale_cam_info = rescale_cam_info_
    
        self.bridge = CvBridge()
        self.img_pub_ = rospy.Publisher(self.cam_name+"/image_raw", Image, queue_size=10)
        self.info_pub_ = rospy.Publisher(self.cam_name+"/camera_info", CameraInfo, queue_size=1)

    def __del__(self):
        if self.camera:
            rospy.loginfo_once("closing camera streaming...")
            self.camera.release()

    def load_basic_params(self):
        self.device_id = rospy.get_param("~device_id")
        self.width = rospy.get_param("~image_width")
        self.height = rospy.get_param("~image_height")
        self.url = rospy.get_param("~camera_info_url")


    def load_cv_cap_params(self):
        server_param_list = rospy.get_param_names()

        for cv_param in cv_extra_param_dict.keys():
            resolved_param = rospy.resolve_name("~"+cv_param)

            if resolved_param in server_param_list:
                rospy.loginfo("Setting {}".format(resolved_param))
                param_val = rospy.get_param(resolved_param)
                ret = self.camera.set(cv_extra_param_dict[cv_param], param_val)
                if ret:
                    rospy.loginfo("Successfully set {}".format(cv_param))
                else:
                    rospy.logwarn("Cannot set {}".format(cv_param))

    def rescale_info(self, width, height):
        width_coeff = width / self.info_.width
        height_coeff = height / self.info_.height

        self.info_.K[0] *= width_coeff
        self.info_.K[2] *= width_coeff
        self.info_.K[4] *= height_coeff
        self.info_.K[5] *= height_coeff

        self.info_.P[0] *= width_coeff
        self.info_.P[2] *= width_coeff
        self.info_.P[5] *= height_coeff
        self.info_.P[6] *= height_coeff
        
    def open(self) -> bool:
        self.camera = cv2.VideoCapture(self.device_id)
        if (self.camera.isOpened()==False):
            rospy.logerr("Error opening video stream")
            return False
        fps_ok = self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        fourcc_ok = self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
        width_ok = self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        height_ok = self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        self.load_cv_cap_params()
        return fps_ok and fourcc_ok and width_ok and height_ok

    def capture(self) -> bool:
        ret, frame = self.camera.read()
        if ret==True:           
            frame_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            frame_msg.encoding = "bgr8" if frame.shape[2] == 3 else "mono8"
            frame_msg.header.stamp = rospy.get_rostime()
            frame_msg.header.frame_id = self.frame_id

            self.info_ = self.info_manager.getCameraInfo()
            if (self.info_.height == 0 and self.info_.width == 0):
                self.info_.height = frame_msg.height
                self.info_.width = frame_msg.width
            elif (self.info_.height != frame_msg.height or self.info_.width != frame_msg.width):
                if self.rescale_cam_info:
                    old_width = self.info_.width
                    old_height = self.info_.height
                    self.rescale_info(frame_msg.width, frame_msg.height)
                    rospy.loginfo_once("Camera calibration automatically rescaled from [{} {}] to [{} {}]".format(
                        old_width, old_height, frame_msg.width, frame_msg.height
                    ))
                else:
                    rospy.loginfo_once("Calibration resolution [{} {}] does not match camera resolution[{} {}]".format(
                        self.info_.width, self.info_.height, frame_msg.width, frame_msg.height
                    ))
            self.img_pub_.publish(frame_msg)
            self.info_pub_.publish(self.info_)
        else:
            rospy.logerr("Cannot read from camera...")
        return ret

def main():
    node = CamNode(
        device_id=0,
        fps=30.0,
        width=1600,
        height=1200,
    )

    if not node.open():
        rospy.logerr("Cannot open video stream...")
        return
    else:
        rospy.loginfo_once("Camera open on " + str(node.device_id))
    
    rate = rospy.get_param("~pub_rate")
    r = rospy.Rate(rate)
    rospy.loginfo_once("pub rate: {}".format(rate))
    while not rospy.is_shutdown():
        is_captured = node.capture()
        r.sleep()

if __name__=="__main__":
    main()