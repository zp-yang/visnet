import numpy as np

# rotation correction between gazebo cam coordinates and literature convention
R_model2cam = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],        
])

# SE2, transform image coordiantes to match the book
SE2_pix2image = np.array([
    [-1, 0,  1600],
    [0, -1, 1200],
    [0, 0,  1],
])

inv_SE2 = SE2_pix2image

# 321 Euler sequence
def euler2dcm(euler):
    from numpy import sin,cos
    phi = euler[0]
    theta = euler[1]
    psi = euler[2]
    
    R1 = np.array([
        [1, 0, 0],
        [0, cos(phi), sin(phi)],
        [0, -sin(phi), cos(phi)],
    ])
    R2 = np.array([
        [cos(theta), 0, -sin(theta)],
        [0, 1, 0],
        [sin(theta), 0, cos(theta)],
    ])
    R3 = np.array([
        [cos(psi), sin(psi), 0],
        [-sin(psi), cos(psi), 0],
        [0, 0 , 1]
    ])
    dcm = R1 @ R2 @ R3
    return dcm

def mrp2dcm(r):
        assert r.shape == (4, 1) or r.shape == (4,)
        a = r[:3]
        X = so3_wedge(a)
        n_sq = np.dot(a, a)
        X_sq = X @ X
        R = np.eye(3) + (8 * X_sq - 4 * (1 - n_sq) * X) / (1 + n_sq) ** 2
        # return transpose, due to convention difference in book
        return R.T

def cart2hom(points):
    """
    points shape should be (3, n)
    """
    n_points = points.shape[1]
    return np.vstack([points, np.ones(n_points)])

def hom2cart(coord):
    coord = coord[0:-1]/coord[-1]
    return coord

def so3_wedge(w):
    wx = np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])
    return wx

def so3_vee(wx):
    w = np.array([wx[2,1], wx[0,2], wx[1,0]])
    return w

eps = 1e-7
def so3_exp(w):
    theta = np.linalg.norm(w)
    C1 = 0
    C2 = 0
    if np.abs(theta) > eps:
        C1 = np.sin(theta)/theta
        C2 = (1 - np.cos(theta))/theta**2
    else:
        C1 = 1 - theta**2/6 + theta**4/120 - theta**6/5040
        C2 = 1/2- theta**2/24 + theta**4/720 - theta**5/40320
    wx = so3_wedge(w)
    R = np.eye(3) + C1 * wx + C2 * wx @ wx
    return R

def so3_log(R):
    theta = np.arccos((np.trace(R) - 1) / 2)
    C3 = 0
    if np.abs(theta) > eps:
        C3 = theta/(2*np.sin(theta))
    else:
        C3 = 0.5 + theta**2/12 + 7*theta**4/720
    return so3_vee(C3 * (R - R.T))

def get_cam_in(cam_param):
    fx = cam_param[0]
    fy = cam_param[4]
    cx = cam_param[2]
    cy = cam_param[5]
    s = cam_param[1]
    cam_in = np.array([
        [fx, s, cx],
        [0, fy, cy],
        [0, 0, 1],
    ])
    return cam_in

# this is the euler angle version
def get_cam_ex_euler(cam_pos, cam_euler):    
    R_world2model = euler2dcm(cam_euler)
    cam_ex = R_model2cam @ R_world2model @ np.block([np.eye(3), -cam_pos.reshape(-1,1)])
    return cam_ex
    
def get_cam_mat_euler(cam_param, cam_pos, cam_euler):
    cam_in = get_cam_in(cam_param)    
    cam_ex = get_cam_ex_euler(cam_pos, cam_euler)

    cam_mat = inv_SE2 @ cam_in @ cam_ex
    return cam_mat

def get_cam_ex_lie(cam_pos, cam_lie):
    R_world2model = so3_exp(-cam_lie)
    cam_ex = R_model2cam @ R_world2model @ np.block([np.eye(3), -cam_pos.reshape(-1,1)])
    return cam_ex

def get_cam_mat_lie(cam_param, cam_pos, cam_lie):
    cam_in = get_cam_in(cam_param)
    cam_ex = get_cam_ex_lie(cam_pos, cam_lie)
    return inv_SE2 @ cam_in @ cam_ex

def get_cam_ex_mrp(cam_pos, cam_mrp):
    R_world2model = mrp2dcm(cam_mrp)
    cam_ex = R_model2cam @ R_world2model @ np.block([np.eye(3), -cam_pos.reshape(-1,1)])
    return cam_ex

def get_cam_mat_mrp(cam_param, cam_pos, cam_mrp):
    cam_in = get_cam_in(cam_param)
    cam_ex = get_cam_ex_mrp(cam_pos, cam_mrp)
    return inv_SE2 @ cam_in @ cam_ex

def get_cam_ex_dcm(cam_pos, R):
   cam_ex = R_model2cam @ R @ np.block([np.eye(3), -cam_pos.reshape(-1,1)])
   return cam_ex

def get_cam_mat_dcm(cam_param, cam_pos, R):
    cam_in = get_cam_in(cam_param)
    cam_ex = get_cam_ex_dcm(cam_pos, R)
    return inv_SE2 @ cam_in @ cam_ex

def get_ray(K, R, p_h):
    # x_h = K R [I | -C] X_h
    # x_h = KR(X - C)
    # inv(KR) x_h = X-C = v
    R_ = R_model2cam @ R
    Bp = np.linalg.inv(inv_SE2 @ K @ R_)
    v = Bp @ p_h
    v = v / np.linalg.norm(v)
    return v
## Resampling based on the examples at: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
## originally by Roger Labbe, under an MIT License
def systematic_resample(weights):
    n = len(weights)
    positions = (np.arange(n) + np.random.uniform(0, 1)) / n

    indices = np.zeros(n, 'i')
    cumsum = np.cumsum(weights)
    i, j = 0, 0
    while i < n:
        if positions[i] < cumsum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    return indices


def stratified_resample(weights):
    n = len(weights)
    positions = (np.random.uniform(0, 1, n) + np.arange(n)) / n
    return create_indices(positions, weights)


def residual_resample(weights):
    n = len(weights)
    indices = np.zeros(n, np.uint32)
    # take int(N*w) copies of each weight
    num_copies = (n * weights).astype(np.uint32)
    k = 0
    for i in range(n):
        for _ in range(num_copies[i]):  # make n copies
            indices[k] = i
            k += 1
    # use multinormial resample on the residual to fill up the rest.
    residual = weights - num_copies  # get fractional part
    residual /= np.sum(residual)
    cumsum = np.cumsum(residual)
    cumsum[-1] = 1
    indices[k:n] = np.searchsorted(cumsum, np.random.uniform(0, 1, n - k))
    return indices


def create_indices(positions, weights):
    n = len(weights)
    indices = np.zeros(n, np.uint32)
    cumsum = np.cumsum(weights)
    i, j = 0, 0
    while i < n:
        if positions[i] < cumsum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1

    return indices


### end rlabbe's resampling functions

def squared_error(x, y, sigma=1):
    """
    Use gaussian probability distribution
    x is the measurements, and y is the mean
    """  
    dx = (x - y) ** 2
    d = np.ma.sum(dx, axis=1)
    return np.exp(-d / (2.0 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma**2)


def msmt_association(measurements, map_hypothesis, sigma=30):
    """
    Parameters:
    -----------
    measurements: shape: (n_measurements, 2+1)
        From the m-th camera, 
    map_hypothesis: shape: (n_targets, 2+1)
        State hypothesis on cameras at t-1 of all tracked objects, 

    RETURN:
    msmt_m: 
        Associated measurements of m-th camera, order in the same way as the targets. |msmt_m[i] <-> tracks[i]|
    """

    msmt_m = []
    argmaxes = []
    for hi in map_hypothesis:
        w = squared_error(measurements, hi, sigma=sigma)
        argmax_w = np.argmax(w)
        argmaxes.append(argmax_w)
        msmt_m.append(measurements[argmax_w])
    
    return np.array(msmt_m), argmaxes

def observe_fn(cam_group, particles):
        """
        Parameters:
            @ particles : all particles of a track
        Returns:
            @ hypothesis from x using the camera projection matrix
            This returns list of arrays which corresponds to hypothesis of particles on each camera
            |hypo[i] <-> camera[i]|
        """

        pos = particles[:, 0:3]
        labels = particles[:, -1]
        hypothesis = cam_group.get_group_measurement(pos, labels)

        return hypothesis

def weight_fn(msmt, hypo, sigma, verbose=False):
        """
        Iterate through each camera, update weights
        @ msmt: [{yj}_0, {yj}_1, ...{yj}_n]    n cameras
        @ hypo: [bearing_0, bearing_1, ....bearing_n]   n_cameras
        """

        n_particles = hypo[0].shape[0]
        weights = np.ones(n_particles)

        for zi, hi in zip(msmt, hypo):
            if zi[0] < 0: # got no measurements so weights are unchanged
                wi = np.ones(n_particles)
            else:
                wi = squared_error(hi, zi, sigma=sigma)
            weights *= wi
        weights_sum = np.sum(weights)
        if verbose:
            # print(weights)
            print(weights_sum)
        # print(weights_sum)
        return weights / weights_sum

def resample(weights_):
    n = len(weights_)
    indices = np.random.choice(n, n, p=weights_)
    return indices

dt = 1/10
A = np.block([
    [np.eye(3), dt*np.eye(3)],
    [np.zeros((3,3)), np.eye(3)]
])
A = np.block([
    [A, np.zeros((6,1))],
    [np.zeros((1,6)), 1]
])

def dynamics_d(x, sigma=0.2):
    """
    Discrete dynamics, last state is the target label
    """
    n, d = x.shape
    w = np.zeros((d, n))
    w[3:6] = np.random.normal(0, sigma, (3,n))
    x_1 = A @ x.T + w

    return x_1.T

def get_iou(bb1, bb2):
    # bounding box format (top-left | bottom-right) (x1, y1, x2, y2)
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
    bb2_area = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou