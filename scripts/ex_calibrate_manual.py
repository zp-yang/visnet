# %%
import numpy as np
import os
data_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), os.pardir)) + "/data/" 
info_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), os.pardir)) + "/camera_info/"
import json
import yaml
import matplotlib.pyplot as plt
import cv2
import util

calib_data = None
with open(f"{data_dir}/calibration/calib_data.json", 'r') as fs:
    calib_data = json.load(fs)
cam_poses = calib_data["pose"]
cam_rois = calib_data["roi"]

cam_names = [
    "camera_0",
    "camera_1",
    # "camera_2",
    "camera_3",
    ]
i = 0
cam_name = cam_names[i]
other_cam_names = cam_names.copy()
other_cam_names.remove(cam_name)

cam_info = None
with open(f"{info_dir}/{cam_name}.yaml", 'r') as fs: 
    cam_info = yaml.safe_load(fs)

#%%
def cart2hom(point):
    return np.hstack([point,1.0])

global cam_P

cam_pinhole_K = np.array(cam_info["camera_matrix"]["data"])
cam_pos = np.array(cam_poses[cam_name][0:3])
cam_att = np.zeros(3)
cam_dist = np.array(cam_info["distortion_coefficients"]["data"])

rois = np.array(cam_rois[cam_name])

if __name__=="__main__":
    windowName = "attitude_calibration"
    maxRoll = 2*np.pi
    maxPitch = 2*np.pi
    maxYaw = 2*np.pi
    maxAlpha = 1

    tb_val_roll = "Roll"
    tb_val_pitch = "Pitch"
    tb_val_yaw = "Yaw"
    tb_val_alpha = "alpha"
    # Read picture

    # Create a window to display the results and set the flag to "cv2.WINDOW_AUTOSIZE"
    cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
    obj_pos_1 = np.array(cam_poses[other_cam_names[0]][0:3])
    obj_pos_2 = np.array(cam_poses[other_cam_names[1]][0:3])

    orig_img = cv2.imread(f"{data_dir}/calibration/{cam_name}_view.jpg")
    orig_img = cv2.rectangle(orig_img, rois[0,0:2], (rois[0,0:2] + rois[0,2:4]), (255,0,0), 2)
    orig_img = cv2.rectangle(orig_img, rois[1,0:2], (rois[1,0:2] + rois[1,2:4]), (0,255,0), 2)

    cam_att = np.zeros(3)
    cam_pos = np.array(cam_poses[cam_name][0:3])
    cam_P, dist_valid_roi = cv2.getOptimalNewCameraMatrix(cam_pinhole_K.reshape(3,3), cam_dist, (1600,1200), 0.5, (1600, 1200))

    slider_scale = 10000
    undistort_img = cv2.undistort(orig_img, cam_pinhole_K.reshape(3,3), cam_dist, None, newCameraMatrix=cam_P)

    print(dist_valid_roi)
    print(undistort_img.shape)

    def project_obj(cam_mat, obj_pos_1, obj_pos_2):
        pix1 = np.array(cam_mat @ np.hstack([obj_pos_1, 1])).reshape(-1)
        pix1 = pix1[0:2]/pix1[2]

        pix2 = np.array(cam_mat @ np.hstack([obj_pos_2, 1])).reshape(-1)
        pix2 = pix2[0:2]/pix2[2]
        
        start = (pix1 - [15, 15]).astype(np.int32)
        end = (pix1 + [15, 15]).astype(np.int32)
        
        drawn_img = undistort_img.copy()
        cv2.rectangle(drawn_img, start, end, (255,120,0), 2)
        
        start = (pix2 - [15, 15]).astype(np.int32)
        end = (pix2 + [15, 15]).astype(np.int32)
        
        cv2.rectangle(drawn_img, start, end, (120,255,0), 2)

        return drawn_img

    # Callback function
    def roll_slider_cb(*args):
        roll = args[0] / slider_scale
        cam_att[0] = roll
        cam_mat = util.get_cam_mat_euler(cam_P.reshape(-1), cam_pos, cam_att)
        
        drawn_img = project_obj(cam_mat, obj_pos_1, obj_pos_2)
        
        cv2.imshow(windowName, drawn_img)

    def pitch_slider_cb(*args):
        pitch = args[0] / slider_scale
        cam_att[1] = pitch
        cam_mat = util.get_cam_mat_euler(cam_P.reshape(-1), cam_pos, cam_att)

        drawn_img = project_obj(cam_mat, obj_pos_1, obj_pos_2)    
        cv2.imshow(windowName, drawn_img)

    def yaw_slider_cb(*args):
        yaw = args[0] / slider_scale
        cam_att[2] = yaw
        cam_mat = util.get_cam_mat_euler(cam_P.reshape(-1), cam_pos, cam_att)
    
        drawn_img = project_obj(cam_mat, obj_pos_1, obj_pos_2)
        cv2.imshow(windowName, drawn_img)

    def alpha_slider_cb(*args):
        alpha = args[0] / slider_scale
        cam_P, _ = cv2.getOptimalNewCameraMatrix(np.array(cam_pinhole_K).reshape(3,3), np.array(cam_dist), (1600,1200), alpha, (1600, 1200))
        cam_mat = util.get_cam_mat_euler(cam_P.reshape(-1), cam_pos, cam_att)
        drawn_img = project_obj(cam_mat, obj_pos_1, obj_pos_2)
        cv2.imshow(windowName, drawn_img)

        

    # Create a slider and associate a callback function
    cv2.createTrackbar(tb_val_roll, windowName, 0, int(maxRoll*slider_scale), roll_slider_cb)
    cv2.createTrackbar(tb_val_pitch, windowName, 0, int(maxPitch*slider_scale), pitch_slider_cb)
    cv2.createTrackbar(tb_val_yaw, windowName, 0, int(maxYaw*slider_scale), yaw_slider_cb)
    cv2.createTrackbar(tb_val_alpha, windowName, 0, int(maxAlpha*slider_scale), alpha_slider_cb)

    # Display image
    cv2.imshow(windowName, undistort_img)
    c = cv2.waitKey(0)
    cv2.destroyAllWindows()
