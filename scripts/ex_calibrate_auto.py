# %%
import casadi as ca
import numpy as np
import os
data_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), os.pardir)) + "/data/" 
info_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), os.pardir)) + "/camera_info/"
import json
import yaml
import matplotlib.pyplot as plt
import cv2
import util

from camera_casadi_expr import *
# %%
# load calibration data and camera info
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

i = 2

cam_name = cam_names[i]
other_cam_names = cam_names.copy()
other_cam_names.remove(cam_name)

cam_info = None
with open(f"{info_dir}/{cam_name}.yaml", 'r') as fs: 
    cam_info = yaml.safe_load(fs)
#%%
# project points with casadi expression

def cart2hom(point):
    return np.hstack([point,1.0])

cam_pinhole_K = np.array(cam_info["camera_matrix"]["data"])
cam_pos = np.array(cam_poses[cam_name][0:3])
cam_att = np.array(cam_poses[cam_name][3:6])

cam_dist = np.array(cam_info["distortion_coefficients"]["data"])
cam_P, dist_valid_roi = cv2.getOptimalNewCameraMatrix(np.array(cam_pinhole_K).reshape(3,3), np.array(cam_dist), (1600,1200), 1, (1600, 1200))
print(dist_valid_roi)

obj_pos_1 = np.array(cam_poses[other_cam_names[0]][0:3])
obj_pos_2 = np.array(cam_poses[other_cam_names[1]][0:3])

cam_mat_ca = get_cam_mat_ca(cam_P.reshape(-1), cam_pos, cam_att_ca) 
cam_mat_lie_ca = get_cam_mat_lie_ca(cam_P.reshape(-1), cam_pos, cam_lie_ca)

cam_mat_ca = cam_mat_lie_ca

p1_hom_ca = cam_mat_ca @ cart2hom(obj_pos_1)
p2_hom_ca = cam_mat_ca @ cart2hom(obj_pos_2)
p1_ca = p1_hom_ca[0:2]/p1_hom_ca[2]
p2_ca = p2_hom_ca[0:2]/p2_hom_ca[2]

#%%
# undistort pictures and check other objects' projection
plt.figure(figsize=(10,10))
img = cv2.imread(f"{data_dir}/calibration/{cam_name}_view.jpg")
plt.imshow(img)

plt.figure(figsize=(10,10))
undistort_img = cv2.undistort(img, cam_pinhole_K.reshape(3,3), cam_dist, None, newCameraMatrix=cam_P)
x, y, w, h = dist_valid_roi

rois = np.array(cam_rois[cam_name])
points = rois[:, 0:2].astype(np.float64)
undistort_points = cv2.undistortPoints(np.expand_dims(points, axis=1), cam_pinhole_K.reshape(3,3), cam_dist, P=cam_P)
undistort_rois = np.hstack([np.squeeze(undistort_points.astype(np.int64)), rois[:,2:4]])

undist_img = cv2.rectangle(undistort_img, undistort_rois[0,0:2], (undistort_rois[0,0:2] + undistort_rois[0,2:4]), (255,0,0), 2)
undist_img = cv2.rectangle(undistort_img, undistort_rois[1,0:2], (undistort_rois[1,0:2] + undistort_rois[0,2:4]), (0,0,255), 2)
plt.imshow(undist_img)

p1 = np.array([
    undistort_rois[0,0] + undistort_rois[0,2]/2,
    undistort_rois[0,1] + undistort_rois[0,3]/2,
    ])

p2 = np.array([
    undistort_rois[1,0] + undistort_rois[1,2]/2,
    undistort_rois[1,1] + undistort_rois[1,3]/2,
    ])
#%%
# solve for the camera's orientation (so3), given the other two object's image position and wold position
nlp = {'x':cam_lie_ca, 'f': ca.norm_2(ca.vertcat(p1-p1_ca, p2-p2_ca))**2, 'g':0}
S = ca.nlpsol('S', 'ipopt', nlp, {
    'print_time': 0,
        # 'ipopt': {
        #     'sb': 'yes',
        #     'print_level': 0,
        #     }
})

# intitialize either positive or negative so3
r = S(x0=[-.1, -.1, -.1], lbg=0, ubg=0)
x_opt = r['x']

# r = S(x0=[.1, .1, .1], lbg=0, ubg=0)
# x_opt = r['x']

# %%
# compare the rotation matrix from auto calibration with manual calibration

print('x_opt: ', x_opt)
print(so3_exp(x_opt))
print(euler2dcm_ca(cam_att))
print(so3_log(so3_exp(x_opt)))
# %%
# plot object projection after calibration
# cam_mat = util.get_cam_mat_euler(cam_P.reshape(-1), cam_pos, cam_att)
cam_mat = util.get_cam_mat_lie(cam_P.reshape(-1), cam_pos, np.array(x_opt).reshape(-1))
pix1 = np.array(cam_mat @ np.hstack([obj_pos_1, 1])).reshape(-1)
pix1 = pix1[0:2]/pix1[2]

pix2 = np.array(cam_mat @ np.hstack([obj_pos_2, 1])).reshape(-1)
pix2 = pix2[0:2]/pix2[2]

start = (pix1 - [15, 15]).astype(np.int32)
end = (pix1 + [15, 15]).astype(np.int32)
plt.figure(figsize=(10,10))

drawn_img = undistort_img.copy()
cv2.rectangle(drawn_img, start, end, (255,120,0), 2)

start = (pix2 - [15, 15]).astype(np.int32)
end = (pix2 + [15, 15]).astype(np.int32)

cv2.rectangle(drawn_img, start, end, (120,255,0), 2)
plt.figure(figsize=(10,10))
plt.imshow(drawn_img)
