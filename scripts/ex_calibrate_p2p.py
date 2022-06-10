# %%
import casadi as ca
import numpy as np
np.set_printoptions(precision=4)

import os
data_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), os.pardir)) + "/data/" 
info_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), os.pardir)) + "/camera_info/"
import json
import yaml
import matplotlib.pyplot as plt
import cv2
import util
import so3

from camera_casadi_expr import *
import enum
class RotRepr(enum.Enum):
    EULER = 0
    LIE = 1
    MRP = 2
    QUAT = 3
# %%
# load calibration data and camera info
calib_data = None
calib_dir = f"{data_dir}/calibration_circle/"
with open(f"{calib_dir}calib_data.json", 'r') as fs:
    calib_data = json.load(fs)
cam_rois = calib_data["roi"]
cam_poses = calib_data["pose"]
# with open(f"{calib_dir}/calib_pose.json", 'r') as fs:
#     cam_poses = json.load(fs)

cam_names = [
    "camera_0",
    "camera_1",
    # "camera_2",
    "camera_3",
    ]

i = 2
method = RotRepr.LIE

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

K = np.array(cam_info["camera_matrix"]["data"])
cam_pos = np.array(cam_poses[cam_name][0:3])
cam_att = np.array(cam_poses[cam_name][3:7])

cam_dist = np.array(cam_info["distortion_coefficients"]["data"])
K_opt, dist_valid_roi = cv2.getOptimalNewCameraMatrix(K.reshape(3,3), cam_dist, (1600,1200), 1, (1600, 1200))
print(f"dist: {dist_valid_roi}")
print(f"K: {K.reshape(3,3)}")
print(f"K_opt: {K_opt}")
obj_pos_1 = np.array(cam_poses[other_cam_names[0]][0:3])
obj_pos_2 = np.array(cam_poses[other_cam_names[1]][0:3])

if method==RotRepr.EULER:
    cam_mat_ca = get_cam_mat_ca(K_opt.reshape(-1), cam_pos, cam_att_ca) 
elif method==RotRepr.LIE:
    cam_mat_ca = get_cam_mat_lie_ca(K_opt.reshape(-1), cam_pos, cam_lie_ca)
elif method==RotRepr.MRP:
    cam_mat_ca = get_cam_mat_mrp_ca(K_opt.reshape(-1), cam_pos, cam_mrp_ca)
elif method==RotRepr.QUAT:
    cam_mat_ca = get_cam_mat_quat_ca(K_opt.reshape(-1), cam_pos, cam_quat_ca)

p1_hom_ca = cam_mat_ca @ cart2hom(obj_pos_1)
p2_hom_ca = cam_mat_ca @ cart2hom(obj_pos_2)
p1_ca = p1_hom_ca[0:2]/p1_hom_ca[2]
p2_ca = p2_hom_ca[0:2]/p2_hom_ca[2]

#%%
# undistort pictures and check other objects' projection
img = cv2.imread(f"{calib_dir}/{cam_name}_view.jpg")

undistort_img = cv2.undistort(img, K.reshape(3,3), cam_dist, None, newCameraMatrix=K_opt)
# x, y, w, h = dist_valid_roi

rois = np.array(cam_rois[cam_name])
points = rois[:, 0:2].astype(np.float64)
undistort_points = cv2.undistortPoints(np.expand_dims(points, axis=1), K.reshape(3,3), cam_dist, P=K_opt)
undistort_rois = np.hstack([np.squeeze(undistort_points.astype(np.int64)), rois[:,2:4]])

undist_img = cv2.rectangle(undistort_img, undistort_rois[0,0:2], (undistort_rois[0,0:2] + undistort_rois[0,2:4]), (255,0,0), 2)
undist_img = cv2.rectangle(undistort_img, undistort_rois[1,0:2], (undistort_rois[1,0:2] + undistort_rois[0,2:4]), (0,0,255), 2)

p1_hom = np.array([
    undistort_rois[0,0] + undistort_rois[0,2]/2,
    undistort_rois[0,1] + undistort_rois[0,3]/2,
    1])
p1 = p1_hom[0:2]
p2_hom = np.array([
    undistort_rois[1,0] + undistort_rois[1,2]/2,
    undistort_rois[1,1] + undistort_rois[1,3]/2,
    1])
p2 = p2_hom[0:2]

# %%
# get the unit vec pointing to the other two objects
vec1 = obj_pos_1 - cam_pos
vec1 = vec1 / np.linalg.norm(vec1)

vec2 = obj_pos_2 - cam_pos
vec2 = vec2 / np.linalg.norm(vec2)

# vec3 = obj_pos_3 - cam_pos
# vec3 = vec3 / np.linalg.norm(vec3)

vec1_ca = get_ray_ca(K_opt.reshape(-1), cam_lie_ca, cart2hom(p1))
vec2_ca = get_ray_ca(K_opt.reshape(-1), cam_lie_ca, cart2hom(p2))
# vec3_ca = get_ray_ca(K_opt.reshape(-1), cam_lie_ca, cart2hom(p3))

# angle between two vectors
def get_vecs_angle(v1, v2):
    # angle is measured from v1 to v2
    # np.arccos(np.dot(v1, v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    n = np.cross(v1, v2)
    d = n / np.linalg.norm(n)
    theta = np.arcsin(np.dot(d, np.cross(v1, v2)) / (np.linalg.norm(v1)*np.linalg.norm(v2)))
    return theta

#%%
# define 'a' as the point on the left of the image plane
p_a = np.max([p1, p2], axis=0)
p_b = np.min([p1, p2], axis=0)
p_c = np.array([K_opt[0,2], K_opt[1,2]]) # image center (after undistortion)

# vectors in camera frame
Bp = np.linalg.inv(util.inv_SE2 @ K_opt @ util.R_model2cam)
v_a = Bp @ cart2hom(p_a)
v_a = v_a / np.linalg.norm(v_a)
v_b = Bp @ cart2hom(p_b)
v_b = v_b / np.linalg.norm(v_b)
v_c = Bp @ cart2hom(p_c)
v_c = v_c / np.linalg.norm(v_c)

v_u = v_a
v_w = np.cross(v_u, v_b)
v_w = v_w / np.linalg.norm(v_w)
v_v = np.cross(v_w, v_u)
v_v = v_v / np.linalg.norm(v_v)
R_0 = np.vstack([v_u, v_v, v_w])

#########################
# vectors in world frame
#########################
vec1 = obj_pos_1 - cam_pos
vec1 = vec1 / np.linalg.norm(vec1)

vec2 = obj_pos_2 - cam_pos
vec2 = vec2 / np.linalg.norm(vec2)

# this should be a function to match the world vec to image (a,b) 
####################
if cam_name=="camera_0" or cam_name=="camera_3":
    vec_a = vec2
    vec_b = vec1
elif cam_name=="camera_1" or cam_name=="camera_2":
    vec_a = vec1
    vec_b = vec2
####################

vec_u = vec_a
vec_w = np.cross(vec_a, vec_b)
vec_w = vec_w / np.linalg.norm(vec_w)

vec_v = np.cross(vec_w, vec_u)
vec_v = vec_v / np.linalg.norm(vec_v)

R_1 = np.vstack([vec_u, vec_v, vec_w])

print(f"R_0: camera body to uvw\n{R_0}")
w0 = np.array(f_dcm2lie(R_0))
theta_0 = np.linalg.norm(np.array(w0))
print(f"axis: {w0/theta_0}, theta_0: {theta_0}")
print(f"roll pitch yaw (deg): \n{np.rad2deg(np.array(f_dcm2euler(R_0.T))).ravel()}\n")

print(f"R_1: world to uvw\n{R_1}")
w1 = np.array(f_dcm2lie(R_1))
theta_1 = np.linalg.norm(w1)
print(f"axis: {w1/theta_1}, theta_0: {theta_1}")
print(f"roll pitch yaw (deg): \n{np.rad2deg(np.array(f_dcm2euler(R_1.T))).ravel()}\n")

# approximated R_cam (manual/mocap)
R_ = np.array(f_euler2dcm(cam_att)).T
print(f"R_approx: \n{R_}")
w_ = util.so3_log(R_)
theta_ = np.linalg.norm(w_)
euler_ = np.array(f_dcm2euler(R_.T))
print(f"axis: {w_/theta_}, theta_0: {theta_}")
print(f"euler(deg): \n{np.rad2deg(euler_).ravel()}\n")

R_cam = R_0.T @ R_1
print(f"R_cam: calculated\n {R_cam}")

w_cam_0 = util.so3_log(R_cam.T)
theta_cam_0 = np.linalg.norm(w_cam_0)
print(f"axis: {w_cam_0/theta_cam_0}, theta_0: {theta_cam_0}")
print(f"euler: \n{np.rad2deg(np.array(f_dcm2euler(R_cam.T))).ravel()}")
#%%
##########################################
# cam_mat = util.get_cam_mat_lie(K_opt.reshape(-1), cam_pos, w_cam_0)
# cam_mat = util.get_cam_mat_euler(K_opt.reshape(-1), cam_pos, cam_att)
cam_mat = util.get_cam_mat_dcm(K_opt.reshape(-1), cam_pos, R_cam)

pix1 = np.array(cam_mat @ np.hstack([obj_pos_1, 1])).reshape(-1)
pix1 = pix1[0:2]/pix1[2]

pix2 = np.array(cam_mat @ np.hstack([obj_pos_2, 1])).reshape(-1)
pix2 = pix2[0:2]/pix2[2]

drawn_img = undistort_img.copy()

start = (pix1 - [15, 15]).astype(np.int32)
end = (pix1 + [15, 15]).astype(np.int32)
cv2.rectangle(drawn_img, start, end, (255,120,0), 2)

start = (pix2 - [15, 15]).astype(np.int32)
end = (pix2 + [15, 15]).astype(np.int32)
cv2.rectangle(drawn_img, start, end, (120,255,0), 2)

nx, ny, nz = (11, 1, 9)
x = np.linspace(-2,2, nx)
y = np.linspace(0,1, ny)
z = np.linspace(0, 3, nz)
# vx, vy, vz = np.meshgrid(x, y, z)

# drawn_img = undistort_img.copy()
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            p = np.array([x[i], y[j], z[k]])

            pix_pos = np.array(cam_mat @ np.hstack([p, 1])).reshape(-1)
            pix_pos = pix_pos[0:2] / pix_pos[2]
            color = (int(i*20), int(j*5), int(k*20))
            cv2.circle(drawn_img, pix_pos.astype(np.int32), 8, color, 5)
plt.figure(figsize=(15,15))
plt.imshow(drawn_img) 
#%%
# solve for the camera's orientation (so3), given the other two object's image position and wold position
if method == RotRepr.EULER:
    x_des = cam_att_ca
    x0 = [0.1, 0.1, 0.1]
elif method == RotRepr.LIE:
    x_des = cam_lie_ca
    # x0 = [0.1, 0.1, 0.1] # camera_1
    # x0 = [-0.1, -0.1, -0.1] # camera_0
    x0 = w_cam_0
elif method == RotRepr.MRP:
    x_des = cam_mrp_ca
    x0 = [-0.1, -0.1, -0.1, 0]
elif method == RotRepr.QUAT:
    x_des = cam_quat_ca
    x0 = [1, 0, 0, 0]

f = ca.vertcat(p1-p1_ca, p2-p2_ca)
# f = ca.vertcat(ca.cross(vec1, vec1_ca), ca.cross(vec2, vec2_ca))

g = 0

nlp = {'x': x_des, 'f': ca.dot(f, f), 'g': g}
S = ca.nlpsol('S', 'ipopt', nlp, {
    'print_time': 0,
        'ipopt': {
            # 'sb': 'yes',
            # 'print_level': 0,
            }
})

# intitialize either positive or negative so3
# r = S(x0=[.1, -.1, 1.1], lbg=0, ubg=0)
x0 = x0
r = S(x0=x0, lbg=np.zeros(1), ubg=np.zeros(1))
x_opt = r['x']

# %%
# compare the rotation matrix from auto calibration with manual calibration
# print(so3_log(euler2dcm_ca(cam_att)))

if method == RotRepr.LIE:
    print('lie_opt: ', x_opt)
    x_opt = x_opt[0:3]
    print('x0: ', x0)
    print('convert to euler: ', f_lie2euler(-x_opt))
    print('in degrees: ', np.rad2deg(f_lie2euler(-x_opt)).reshape(-1))
    
    print(so3_exp(x_opt))
    
elif method == RotRepr.MRP:
    print('mrp_opt: ', x_opt)
    print(so3.Dcm.from_mrp(x_opt))
elif method == RotRepr.QUAT:
    print('quat_opt: ', x_opt)
    print(so3.Dcm.from_quat(x_opt))
elif method == RotRepr.EULER:
    pass

# print('\nquaternion -> dcm: ', f_q2dcm(cam_att))


# %%
# plot object projection after calibration
if method == RotRepr.EULER:
    cam_mat = util.get_cam_mat_euler(K_opt.reshape(-1), cam_pos, np.array(x_opt).reshape(-1))
elif method == RotRepr.LIE:
    cam_mat = util.get_cam_mat_lie(K_opt.reshape(-1), cam_pos, np.array(x_opt).reshape(-1))
elif method == RotRepr.MRP:    
    cam_mat = util.get_cam_mat_mrp(K_opt.reshape(-1), cam_pos, np.array(x_opt).reshape(-1))
elif method == RotRepr.QUAT:    
    cam_mat = f_cam_mat_quat(K_opt.reshape(-1), cam_pos, np.array(x_opt).reshape(-1))

cam_mat = util.get_cam_mat_lie(K_opt.reshape(-1), cam_pos, w_cam_0)
pix1 = np.array(cam_mat @ np.hstack([obj_pos_1, 1])).reshape(-1)
pix1 = pix1[0:2]/pix1[2]

pix2 = np.array(cam_mat @ np.hstack([obj_pos_2, 1])).reshape(-1)
pix2 = pix2[0:2]/pix2[2]

drawn_img = undistort_img.copy()

start = (pix1 - [15, 15]).astype(np.int32)
end = (pix1 + [15, 15]).astype(np.int32)
cv2.rectangle(drawn_img, start, end, (255,120,0), 2)

start = (pix2 - [15, 15]).astype(np.int32)
end = (pix2 + [15, 15]).astype(np.int32)
cv2.rectangle(drawn_img, start, end, (120,255,0), 2)

nx, ny, nz = (11, 1, 9)
x = np.linspace(-2,2, nx)
y = np.linspace(0,1, ny)
z = np.linspace(0, 3, nz)
# vx, vy, vz = np.meshgrid(x, y, z)

# drawn_img = undistort_img.copy()
pix_poss = []
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            p = np.array([x[i], y[j], z[k]])

            pix_pos = np.array(cam_mat @ np.hstack([p, 1])).reshape(-1)
            pix_pos = pix_pos[0:2] / pix_pos[2]
            pix_poss.append(pix_pos)
            color = (int(i*20), int(j*5), int(k*20))
            cv2.circle(drawn_img, pix_pos.astype(np.int32), 8, color, 5)
plt.figure(figsize=(15,15))
plt.imshow(drawn_img) 
# %%
