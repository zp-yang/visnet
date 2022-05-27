# %%
import jax.numpy as np
import jax
from jax import jit
import util
np.set_printoptions(precision=4)

#%%
# [x y z roll pitch yaw]
cam_poses = np.array([
    [20, 20, 12, 0, 0, -2.2],
    [20, -15, 12, 0, 0, 2.2],
    [-20, -20, 12, 0, 0, 0.7],
    [-20, 20, 12, 0, 0, -0.7],
])
cam_param = [642.0926, 642.0926, 1000.5, 1000.5,0]
cam = 3

R_model2cam = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],        
    ])

def cart2hom(point):
    return np.hstack([point,1.0])

def hom2cart(coord):
    coord = coord[0:-1]/coord[-1]
    return coord

def wrap(x):
    return np.where(np.abs(x) >= np.pi, (x + np.pi) % (2 * np.pi) - np.pi, x)

def wrap_vec(v):
    return np.array([wrap (vi) for vi in v])

def so3_wedge(w):
    wx = np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])
    return -wx

def so3_vee(wx):
    w = np.array([wx[2,1], wx[0,2], w[1,0]])
    return w

eps = 1e-7
def so3_exp(w):
    theta = np.linalg.norm(w)
    
    wx = so3_wedge(w)
    C1 = np.where(theta > eps, np.sin(theta)/theta, 1 - theta**2/6 + theta**4/120 - theta**6/5040)
    C2 = np.where(theta > eps, (1 - np.cos(theta))/theta**2, 1/2 - theta**2/24 + theta**4/720 - theta**6/40320)
    
    return np.eye(3) + C1 * wx + C2 * wx @ wx

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

def get_cam_ex_lie(cam_pos, cam_att):
    R_world2model = so3_exp(cam_att)
    cam_ex = R_model2cam @ R_world2model @ np.block([np.eye(3), -cam_pos.reshape(-1,1)])
    return cam_ex

def get_cam_mat_lie(cam_param, cam_pos, cam_att):
    cam_in = get_cam_in(cam_param)
    cam_ex = get_cam_ex_lie(cam_pos, cam_att)
    return cam_in @ cam_ex

def euler2dcm(euler):
    from jax.numpy import sin,cos
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
# %%
cam_mat_true = util.get_cam_mat_euler(cam_param, cam_poses[cam][0:3], cam_poses[cam][3:6])
p1_hom = cam_mat_true @ util.cart2hom(cam_poses[1][0:3])
p2_hom = cam_mat_true @ util.cart2hom(cam_poses[2][0:3])
p2_hom.astype('int32')
#%%
def f_calib_euler(cam_att):
    cam_pos = cam_poses[cam][0:3]
    cam_mat = get_cam_mat_euler(cam_param, cam_poses[cam][0:3], cam_att)
    p1_hom_hat = cam_mat @ cart2hom(cam_poses[1][0:3])
    p2_hom_hat = cam_mat @ cart2hom(cam_poses[2][0:3])                                                                                  

    err = np.linalg.norm(np.hstack([p1_hom-p1_hom_hat, p2_hom-p2_hom_hat]))**2
    return err

df_calib_euler = jax.jacobian(f_calib_euler)
df_calib_euler(np.array([.0, .0, -0.0]))

def hessian(f):
    return jax.jacfwd(jax.jacrev(f))

H = hessian(f_calib_euler)
J = jax.jacfwd(f_calib_euler)

@jit
def minHessian(x): 
    return x - 0.1*np.linalg.inv(H(x)) @ J(x)  

key = jax.random.PRNGKey(42)
domain = jax.random.uniform(key, shape=(50,3), dtype='float32', minval=-np.pi, maxval=np.pi)

vfuncHS = jax.vmap(minHessian)
for epoch in range(200):
  domain = vfuncHS(domain)

minfunc = jax.vmap(f_calib_euler)
minimums = minfunc(domain)

# After running the loop, we look for the argmin and the objective minimum

arglist = np.nanargmin(minimums)
argmin = domain[arglist]
minimum = minimums[arglist]

print("The minimum is {} \nthe arg min is ({},{},{})".format(minimum,argmin[0],argmin[1],argmin[2]))

# %%
def f_calib_lie(cam_att):
    cam_pos = cam_poses[cam][0:3]
    cam_mat = get_cam_mat_lie(cam_param, cam_poses[cam][0:3], cam_att)
    p1_hom_hat = cam_mat @ cart2hom(cam_poses[1][0:3])
    p2_hom_hat = cam_mat @ cart2hom(cam_poses[2][0:3])                                                                                  

    err = np.linalg.norm(np.hstack([p1_hom-p1_hom_hat, p2_hom-p2_hom_hat]))**2
    return err

df_calib_lie = jax.jacobian(f_calib_euler)
df_calib_lie(np.array([.0, .0, -0.0]))

H = hessian(f_calib_lie)
J = jax.jacfwd(f_calib_lie)

@jit
def minHessian(x):
    return x - 0.5*np.linalg.inv(H(x)) @ J(x)

minfunc = jax.vmap(f_calib_euler)

key = jax.random.PRNGKey(42)
domain = jax.random.uniform(key, shape=(50,3), dtype='float32', minval=-np.pi, maxval=np.pi)

vfuncHS = jax.vmap(minHessian)
for epoch in range(100):
  domain = vfuncHS(domain)


minimums = minfunc(domain)

# After running the loop, we look for the argmin and the objective minimum

arglist = np.nanargmin(minimums)
argmin = domain[arglist]
minimum = minimums[arglist]
argmin = wrap_vec(argmin)
print("The minimum is {} \nthe arg min is ({},{},{})".format(minimum,argmin[0],argmin[1],argmin[2]))