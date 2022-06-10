import numpy as np
import casadi as ca
import so3

rot_model2cam = ca.SX(np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ]))

SE2_pix2image = ca.SX([
    [-1, 0,  1600],
    [0, -1, 1200],
    [0, 0,  1],
])

inv_SE2 = SE2_pix2image

def euler2dcm_ca(euler):
    #euler 321
    from casadi import sin,cos
    phi = euler[0]
    theta = euler[1]
    psi = euler[2]
    
    dcm = ca.SX(3,3)
    dcm[0,0] = cos(theta)*cos(psi)
    dcm[0,1] = cos(theta)*sin(psi)
    dcm[0,2] = -sin(theta)
    dcm[1,0] = -cos(phi)*sin(psi)+sin(phi)*sin(theta)*cos(psi)
    dcm[1,1] = cos(phi)*cos(psi)+sin(phi)*sin(theta)*sin(psi)
    dcm[1,2] = sin(phi)*cos(theta)
    dcm[2,0] = sin(phi)*sin(psi)+cos(phi)*sin(theta)*cos(psi)
    dcm[2,1] = -sin(phi)*cos(psi)+cos(phi)*sin(theta)*sin(psi)
    dcm[2,2] = cos(phi)*cos(theta)
#     dcm = ca.MX([
#         [cos(theta)*cos(psi), cos(theta)*sin(psi), -sin(theta)],
#         [-cos(phi)*sin(psi)+sin(phi)*sin(theta)*cos(psi), cos(phi)*cos(psi)+sin(phi)*sin(theta)*sin(psi), sin(phi)*cos(theta)],
#         [sin(phi)*sin(psi)+cos(phi)*sin(theta)*cos(psi), -sin(phi)*cos(psi)+cos(phi)*sin(theta)*sin(psi), cos(phi)*cos(theta)]
#     ])
    return dcm

def get_cam_in_ca(cam_param):
    if type(cam_param)==type([]):
        cam_param = np.array(cam_param)
    
    if not cam_param.shape[0] == 9:
        print("invalid camera parameters please check again!!!!")
        return None

    fx = cam_param[0]
    fy = cam_param[4]
    cx = cam_param[2]
    cy = cam_param[5]
    s = cam_param[1]

    cam_in = ca.SX(3,3)
    cam_in[0,0] = fx
    cam_in[0,1] = s
    cam_in[0,2] = cx
    cam_in[1,1] = fy
    cam_in[1,2] = cy
    cam_in[2,2] = 1
    return cam_in

def get_cam_mat_ca(cam_param, cam_pos, cam_att):    
    pos = cam_pos
    euler = cam_att
    # intrinsic matrix
    cam_in = get_cam_in_ca(cam_param)

    # extrinsic matrix
    rot_world2model = euler2dcm_ca(euler)
#     rot_world2model = so3.Dcm.from_euler(euler)
    
    cam_ex = rot_model2cam @ rot_world2model
    block = ca.SX(3,4)
    block[:,0:3] = cam_ex
    block[:,3] = -(cam_ex @ pos)
    cam_mat = inv_SE2 @ cam_in @ block
    return cam_mat

#%%
# Lie group algebra
def so3_wedge(v):
    X = ca.SX(3,3)
    X[0, 1] = -v[2]
    X[0, 2] = v[1]
    X[1, 0] = v[2]
    X[1, 2] = -v[0]
    X[2, 0] = -v[1]
    X[2, 1] = v[0]
    return X

def vee(X):
    v = ca.SX(3, 1)
    v[0, 0] = X[2, 1]
    v[1, 0] = X[0, 2]
    v[2, 0] = X[1, 0]
    return v

eps = 1e-7
x = ca.SX.sym('x')
C1 = ca.Function('a', [x], [ca.if_else(ca.fabs(x) < eps, 1 - x ** 2 / 6 + x ** 4 / 120, ca.sin(x)/x)])
C2 = ca.Function('b', [x], [ca.if_else(ca.fabs(x) < eps, 0.5 - x ** 2 / 24 + x ** 4 / 720, (1 - ca.cos(x)) / x ** 2)])
C3 = ca.Function('d', [x], [ca.if_else(ca.fabs(x) < eps, 0.5 + x**2/12 + 7*x**4/720, x/(2*ca.sin(x)))])
del x

def so3_exp(v):
    theta = ca.norm_2(v)
    X = so3_wedge(v)
    R = ca.SX.eye(3) + C1(theta)*X + C2(theta)*ca.mtimes(X, X)
    # NOTE: WHY DOES R need a negative sign to match euler321 to DCM ????
    return R
    # return -R

def so3_log(R):
    # R = -R
    theta = ca.arccos((ca.trace(R) - 1) / 2)
    return vee(C3(theta) * (R - R.T))

def get_cam_mat_lie_ca(cam_param, cam_pos, cam_lie):  
    pos = cam_pos
    w = cam_lie # so3
    
    # intrinsic matrix
    cam_in = get_cam_in_ca(cam_param)
    # extrinsic matrix
    rot_world2model = so3_exp(-w)
    
    cam_ex = rot_model2cam @ rot_world2model
    block = ca.SX(3,4)
    block[:,0:3] = cam_ex
    block[:,3] = -(cam_ex @ pos)
    cam_mat = inv_SE2 @ cam_in @ block
    return cam_mat

def get_cam_mat_mrp_ca(cam_param, cam_pos, cam_mrp):   
    pos = cam_pos

    # intrinsic matrix
    cam_in = get_cam_in_ca(cam_param)

    # extrinsic matrix
    rot_world2model = so3.Dcm.from_mrp(cam_mrp)
    
    cam_ex = rot_model2cam @ rot_world2model
    block = ca.SX(3,4)
    block[:,0:3] = cam_ex
    block[:,3] = -(cam_ex @ pos)
    cam_mat = inv_SE2 @ cam_in @ block
    return cam_mat

def get_cam_mat_quat_ca(cam_param, cam_pos, cam_quat):
    pos = cam_pos

    # intrinsic matrix
    cam_in = get_cam_in_ca(cam_param)

    # extrinsic matrix
    rot_world2model = so3.Dcm.from_quat(cam_quat)
    
    cam_ex = rot_model2cam @ rot_world2model
    block = ca.SX(3,4)
    block[:,0:3] = cam_ex
    block[:,3] = -(cam_ex @ pos)
    cam_mat = inv_SE2 @ cam_in @ block
    return cam_mat

def get_cam_mat_R_ca(cam_param, cam_pos, cam_R):
    pos = cam_pos

    # intrinsic matrix
    cam_in = get_cam_in_ca(cam_param)

    # extrinsic matrix
    rot_world2model = cam_R
    
    cam_ex = rot_model2cam @ rot_world2model
    block = ca.SX(3,4)
    block[:,0:3] = cam_ex
    block[:,3] = -(cam_ex @ pos)
    cam_mat = inv_SE2 @ cam_in @ block
    return cam_mat

def get_ray_ca(cam_param, cam_lie, pix_hom):
    K = get_cam_in_ca(cam_param)
    R = rot_model2cam @ so3_exp(cam_lie) 
    Bp = ca.inv(K @ R)

    vec = Bp @ pix_hom
    vec = vec / ca.norm_2(vec)
    return vec

cam_param_ca = ca.SX.sym('param', 9)

cam_pos_ca = ca.SX.sym('pos', 3)
cam_att_ca = ca.SX.sym('att', 3)
cam_lie_ca = ca.SX.sym('lie', 3)
cam_mrp_ca = ca.SX.sym("mrp", 4)
cam_quat_ca = ca.SX.sym("quat", 4)
cam_dcm_ca = ca.SX.sym("dcm", (3,3))

rho_ca = ca.SX.sym("rho", 2) # scaling of image coordinate after projection

cam_mat_ca = get_cam_mat_ca(cam_param_ca, cam_pos_ca, cam_att_ca)

cam_mat_lie_ca = get_cam_mat_lie_ca(cam_param_ca, cam_pos_ca, cam_lie_ca)
cam_mat_quat_ca = get_cam_mat_quat_ca(cam_param_ca, cam_pos_ca, cam_quat_ca)

f_cam_mat_euler = ca.Function('f_cam_mat',[cam_param_ca, cam_pos_ca, cam_att_ca], [cam_mat_ca])
f_cam_mat_lie = ca.Function('f_cam_mat_lie',[cam_param_ca, cam_pos_ca, cam_lie_ca], [cam_mat_lie_ca])
f_cam_mat_quat = ca.Function('f_cam_mat_quat', [cam_param_ca, cam_pos_ca, cam_quat_ca], [cam_mat_quat_ca])
f_lie2euler = ca.Function('f_lie2euler', [cam_lie_ca], [so3.Euler.from_dcm(so3.Dcm.exp(cam_lie_ca))])
f_lie2dcm = ca.Function('f_lie2dcm', [cam_lie_ca], [so3.Dcm.exp(cam_lie_ca)])
f_q2dcm = ca.Function('f_q2dcm', [cam_quat_ca], [so3.Dcm.from_quat(cam_quat_ca)])
f_euler2dcm = ca.Function('f_euler2dcm', [cam_att_ca], [so3.Dcm.from_euler(cam_att_ca)])
f_dcm2euler = ca.Function('f_dcm2euler', [cam_dcm_ca], [so3.Euler.from_dcm(cam_dcm_ca)])
f_dcm2lie = ca.Function('dcm2lie', [cam_dcm_ca], [so3.Dcm.log(cam_dcm_ca)])
# opencv getOptimalNewCamera Matrix 
# https://github.com/opencv/opencv/blob/8b4fa2605e1155bbef0d906bb1f272ec06d7796e/modules/calib3d/src/calibration.cpp#L2714
#
# opencv undistort
# https://github.com/egonSchiele/OpenCV/blob/edc96c2a8c3266d0da839df362b5978bb590bfbd/modules/imgproc/src/undistort.cpp#L251

# Re-write these code in casadi to run optimization/calibration

def getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imgSize, alpha, newImgSize):
    newCameraMatrix = ca.SX.zeros(3,3)
    validRoi = ca.SX.zeros(1,4)

    return newCameraMatrix, validRoi

def undistortPoints(points, cameraMatrix, k, P):
    # points: points to be undistorted, i.e. other camera's image position [[u,v],...] (n by 2)
    # cameraMatrix: pinhole_K (3 by 3)
    # k: distortion coefficient: k1 k2 p1 p2 k3
    # P: new optimal camera matrix (3 by 3)
    fx = cameraMatrix[0,0]
    fy = cameraMatrix[1,1]
    cx = cameraMatrix[0,2]
    cy = cameraMatrix[1,2]
    n = points.shape[0]
    iter_num = 5

    undistort_points = ca.SX(points.shape)
    for i in range(n):
        x = points[i, 0]
        y = points[i, 1]
        x = (x - cx) / fx
        y = (y - cy) / fy
        x0 = x
        y0 = y
        for _ in range(iter_num):
            r2 = x*x + y*y
            icdist = 1 / (1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2)
            delta_x = 2*k[2]*x*y + k[3]*(r2 + 2*x*x)
            delta_y = k[2]*(r2 + 2*y*y) + 2*k[3]*x*y
            x = (x0 - delta_x)*icdist
            y = (y0 - delta_y)*icdist
        
        xx = P[0,0]*x + P[0,1]*y + P[0,2]
        yy = P[1,0]*x + P[1,1]*y + P[1,2]
        ww = P[2,0]*x + P[2,1]*y + P[2,2]
        x = xx / ww
        y = yy / ww
        undistort_points[i,0] = x
        undistort_points[i,1] = y

    return undistort_points
# %%
