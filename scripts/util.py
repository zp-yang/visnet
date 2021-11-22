import numpy as np

# rotation correction between gazebo cam coordinates and literature convention
R_model2cam = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],        
    ])

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

def cart2hom(point):
#     print(point.shape, point)
    return np.hstack([point,1.0])

def hom2cart(coord):
    coord = coord[0:-1]/coord[-1]
    return coord

def so3_wedge(w):
    wx = np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])
    return -wx

def so3_vee(wx):
    w = np.array([wx[2,1], wx[0,2], wx[1,0]])
    return w

eps = 1e-7    
def so3_exp(w):
    theta = np.linalg.norm(w)
    C1 = 0
    C2 = 0
    if theta > eps:
        C1 = np.sin(theta)/theta
        C2 = (1 - np.cos(theta))/theta**2
    else:
        C1 = 1 - theta**2/6 + theta**4/120 - theta**6/5040
        C2 = 1/2- theta**2/24 + theta**4/720 - theta**5/40320
    wx = so3_wedge(w)
    return np.eye(3)  + C1 * wx + C2 * wx @ wx

def so3_log(R):
    theta = np.arccos((np.linalg.trace(R) - 1) / 2)
    return so3_vee(C3(theta) * (R - R.T))

def get_cam_in(cam_param):
    fx = cam_param[0]
    fy = cam_param[1]
    cx = cam_param[2]
    cy = cam_param[3]
    s = cam_param[4]
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

    cam_mat = cam_in @ cam_ex
    return cam_mat

def get_cam_ex_lie(cam_pos, cam_att):
    R_world2model = so3_exp(cam_att)
    cam_ex = R_model2cam @ R_world2model @ np.block([np.eye(3), -cam_pos.reshape(-1,1)])
    return cam_ex

def get_cam_mat_lie(cam_param, cam_pos, cam_att):
    cam_in = get_cam_in(cam_param)
    cam_ex = get_cam_ex_lie(cam_pos, cam_att)
    return cam_in @ cam_ex

## Resampling based on the examples at: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
## originally by Roger Labbe, under an MIT License
def systematic_resample(weights):
    n = len(weights)
    positions = (np.arange(n) + np.random.uniform(0, 1)) / n
    return create_indices(positions, weights)


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

# resample function from http://scipy-cookbook.readthedocs.io/items/ParticleFilter.html
def resample(weights):
    n = len(weights)
    indices = []
    C = [0.0] + [np.sum(weights[: i + 1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0 + i) / n for i in range(n)]:
        while u > C[j]:
            j += 1
        indices.append(j - 1)
    return indices

def squared_error(x, y, sigma=1):
#     """
#         RBF kernel, supporting masked values in the observation
#         Parameters:
#         -----------
#         x : array (N,D) array of values | hypotheses
#         y : array (N,D) array of values | measurements

#         Returns:
#         -------

#         distance : scalar
#             Total similarity, using equation:

#                 d(x,y) = e^((-1 * (x - y) ** 2) / (2 * sigma ** 2))

#             summed over all samples. Supports masked arrays.
#     """  
    dx = (x - y) ** 2
    d = np.ma.sum(dx, axis=1)
    return np.exp(-d / (2.0 * sigma ** 2))