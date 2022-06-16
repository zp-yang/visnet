# python implementation of Nakano's p3p solver, originally in matlab 
# https://github.com/g9nkn/p3p_problem
# @inproceedings{nakano2019simple,
#   title={A Simple Direct Solution to the Perspective-Three-Point Problem},
#   author={Nakano, Gaku},
#   numpages={12},
#   booktitle={Proceedings of the British Machine Vision Conference (BMVC)},
#   publisher={BMVA Press},
#   year={2019}
# }

# INPUTS:
#   m - 3x3 matrix of 2D points represented by homogeneous coordinates.
#       Each column m(:,i) corresponds to the 3D point X(:,i),
#       [u1, u2, u3
#        v1, v2, v3
#        w1, w2, w3],
#       where each column is normalized by sqrt(u^2+v^2+w^2)=1.
#   |^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^|
#   | Above is what the paper says |
#   | But actually columns of m are the back projected ray 
#   | from 2d points using K and normalized (in camera body frame)
#
#   X - 3x3 matrix of 3D points.
#       Each column X(:,i) corresponds to the 2D point m(:,i),
#       [x1, x2, x3
#        y1, y2, y3
#        z1, z2, z3].
#
#   polishing - (optional) an integer to set the number of iterations of
#               root polishing. If <= 0, the root polishing is not performed.
#               (default: 1)
#
# OUTPUS:
#   R - 3x3xM rotation matrix (1<= M <= 4). 
#       R(:,:,i) corresponds to t(:,i).
#   t - 3xM translation vector.
# Note:
#   d_i * m_i = R_w * X_i + t_w 
#   K is implied in R_w
# In the paper f = 800, c = (320,240) for camera's K
# %%
import numpy as np
from numpy.linalg import norm, inv
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

# load calibration data and camera info
calib_data = None
calib_dir = f"{data_dir}/calibration_p3p/"

with open(f"{calib_dir}calib_data.json", 'r') as fs:
    calib_data = json.load(fs)
cam_rois = calib_data["roi"]
cam_poses = calib_data["pose"]
# with open(f"{calib_dir}/calib_pose.json", 'r') as fs:
#     cam_poses = json.load(fs)

cam_names = [
    "camera_0",
    "camera_1",
    "camera_2",
    "camera_3",
    ]

i = 3

cam_name = cam_names[i]
other_cam_names = cam_names.copy()
other_cam_names.remove(cam_name)

cam_info = None
with open(f"{info_dir}/{cam_name}.yaml", 'r') as fs: 
    cam_info = yaml.safe_load(fs)

K = np.array(cam_info["camera_matrix"]["data"])
cam_pos = np.array(cam_poses[cam_name][0:3])
cam_att = np.array(cam_poses[cam_name][3:7])

cam_dist = np.array(cam_info["distortion_coefficients"]["data"])
K_opt, dist_valid_roi = cv2.getOptimalNewCameraMatrix(K.reshape(3,3), cam_dist, (1600,1200), 1, (1600, 1200))
print(f"dist: {dist_valid_roi}")
print(f"K: \n{K.reshape(3,3)}")
print(f"K_opt: \n{K_opt}")
obj_pos_1 = np.array(cam_poses[other_cam_names[0]][0:3])
obj_pos_2 = np.array(cam_poses[other_cam_names[1]][0:3])
obj_pos_3 = np.array(cam_poses[other_cam_names[2]][0:3])
print("objects points: \n", np.array([obj_pos_1, obj_pos_2, obj_pos_3]))

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
undist_img = cv2.rectangle(undistort_img, undistort_rois[1,0:2], (undistort_rois[1,0:2] + undistort_rois[0,2:4]), (0,255,0), 2)
undist_img = cv2.rectangle(undistort_img, undistort_rois[2,0:2], (undistort_rois[2,0:2] + undistort_rois[0,2:4]), (0,0,255), 2)

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

p3_hom = np.array([
    undistort_rois[2,0] + undistort_rois[2,2]/2,
    undistort_rois[2,1] + undistort_rois[2,3]/2,
    1])
p3 = p3_hom[0:2]

Bp = inv(util.inv_SE2 @ K_opt @ util.R_model2cam)

v1 = Bp @ p1_hom
v1 = v1 / norm(v1)
v2 = Bp @ p2_hom
v2 = v2 / norm(v2)
v3 = Bp @ p3_hom
v3 = v3 / norm(v3)

print("image points: \n", np.array([v1, v2, v3]))
plt.figure(figsize=(15,15))
plt.imshow(undist_img)

# %%
def p3p_nakano(m, X, polishing=0):
    m = np.array([v1, v2, v3]).T
    X = np.array([obj_pos_1, obj_pos_2, obj_pos_3]).T
    m = np.array(m)
    X = np.array(X)
    assert m.shape == (3,3)
    assert X.shape == (3,3)

    x12 = X[:,1] - X[:,0]
    x23 = X[:,2] - X[:,1]
    x13 = X[:,2] - X[:,0]


    d = np.sqrt(np.sum(np.array([X[:,1] - X[:,0], X[:,2] - X[:,1], X[:,0] - X[:,2]]).T**2, axis=0))
    indx = np.argmax(d)
    print(d, indx)
    # %%
    print("X:\n", X, "\nm: \n", m)
    if indx==1:
        X = X[:, [1,2,0]]
        m = m[:, [1,2,0]]
    elif indx==2:
        X = X[:, [0,2,1]]
        m = m[:, [0,2,1]]
    print("X:\n", X, "\nm: \n", m)
    #%%
    # rigid transformation so that all points are on a plane z=0.
    # Xg = [ 0 a b
    #        0 0 c
    #        0 0 0]
    # a~=b, b>0, c>0

    X21 = X[:,1] - X[:,0]
    X31 = X[:,2] - X[:,0]
    nx = X21
    nx = nx / norm(nx)
    nz = np.cross(nx, X31)
    nz = nz / norm(nz)
    ny = np.cross(nz, nx)
    N = np.array([nx, ny, nz]).T
    print("N: \n", N)
    # %%
    # calculate coefficients of the polynomial for sovling projective depth
    a = N[:,0] @ X21
    b = N[:,0] @ X31
    c = N[:,1] @ X31

    M12 = m[:,0] @ m[:,1]
    M13 = m[:,0] @ m[:,2]
    M23 = m[:,1] @ m[:,2]
    p = b/a
    q = (b**2 + c**2) / a**2

    f = np.array([p, -M23, 0, -M12*(2*p-1), M13, p-1])
    g = np.array([q, 0, -1, -2*M12*q, 2*M13, q-1])

    h = np.array([
        -f[0]**2 + g[0]*f[1]**2,
        f[1]**2*g[3] - 2*f[0]*f[3] - 2*f[0]*f[1]*f[4] + 2*f[1]*f[4]*g[0],
        f[4]**2*g[0] - 2*f[0]*f[4]**2 - 2*f[0]*f[5] + f[1]**2*g[5] - f[3]**2 - 2*f[1]*f[3]*f[4] + 2*f[1]*f[4]*g[3],
        f[4]**2*g[3] - 2*f[3]*f[4]**2 - 2*f[3]*f[5] - 2*f[1]*f[4]*f[5] + 2*f[1]*f[4]*g[5],
        - 2*f[4]**2*f[5] + g[5]*f[4]**2 - f[5]**2,
    ])
    print("f: ", f)
    print("g: ", g)
    print("h: ", h)
    #%%
    x = np.roots(h)
    indx = (np.real(x) > 0) & (np.abs(np.imag(x))<1e-8)
    x = np.real(x[indx])
    y = -((f[0]*x + f[3])*x + f[5])/(f[4] + f[1]*x)
    print("x: ", x)
    print("y: ", y)
    #%%
    # recover motion
    nsols = x.shape[0]
    A = m * np.array([-1, 1, 0])
    B = m * np.array([-1, 0, 1])
    C = B - p * A

    R = np.zeros((3,3, nsols))
    t = np.zeros((3,nsols))

    for i in range(nsols):
        lam = np.array([1, x[i], y[i]])
        s = norm(A @ lam) / a
        d = lam / s

        r1 = (A @ d) / a
        r2 = (C @ d) / c
        r3 = np.cross(r1, r2)
        Rc = np.array([r1, r2, r3]).T
        tc = d[0] * m[:,0]

        R[:,:,i] = Rc @ N.T
        t[:,i] = tc - Rc @ N.T @ X[:,0]

    print(R, t)
# %%
