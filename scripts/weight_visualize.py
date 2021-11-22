#%%
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
sys.path.insert(0, "../scripts")
import util
np.set_printoptions(precision=4)

#%%
class Camera:
    def __init__(self, cam_param, cam_pos, cam_att):
        self.param = cam_param
        self.pos = cam_pos
        self.att = cam_att
        self.R_cam = util.so3_exp(self.att)
        self.K = util.get_cam_in(self.param)
        self.R = util.R_model2cam @ self.R_cam # this rotation is kinda goofy but works, TODO: maybe
        self.P = util.get_cam_mat_lie(self.param, self.pos, self.att)
        self.range = 20 # meters
        self.fov = np.deg2rad(90) #degrees, both directions
        
    def __repr__(self):
        return "param: {}, pos: {}, att: {}".format(self.param, self.pos, self.att)
        
    def _get_pixel_pos_hom(self, target_pos):
        return self.P @ util.cart2hom(target_pos)
    
    def _get_pixel_pos(self, target_pos):
        return util.hom2cart(self._get_pixel_pos_hom(target_pos))
    
    def _get_distance(self, target_pos):
        """
        !!!Should not be used directly in estimation!!!
        """
        dist =  np.linalg.norm(target_pos-self.pos)
        return dist
    
    def get_bearing(self, target_pos):
        """
        considers the FOV and range of the camera
        Outside FOV - no measurement
        Outside range - worse measurement, more noise?
        """
        # check FOV on x & y axis of the camera frame
        dist = self._get_distance(target_pos)
        vec = self.R_cam @ (target_pos - self.pos) / dist
        ang1 = np.arctan2(vec[2], vec[0])
        ang2 = np.arctan2(vec[1], vec[0])
        if not(ang1 < self.fov/2 and ang1 > -self.fov/2) or not(ang2 < self.fov/2 and ang2 > -self.fov/2) :
            return np.array([-1, -1])
        
        bearing = self._get_pixel_pos(target_pos)
        if dist > self.range:
            return self.add_pixel_noise(bearing, sigma=20)
        return self.add_pixel_noise(bearing)
        
    def add_pixel_noise(self, pixel_coord, sigma=10):
        return pixel_coord + np.random.normal(0, sigma)
    
class CamNode(Camera):
    def __init__(self, cam_param, cam_pos, cam_att):
        super(CamNode, self).__init__(cam_param, cam_pos, cam_att)
        
    def _get_vec(self, pixel_pos):
        """
        Back projected ray from the camera to the target (approximate)
        """
        pixel_coord = self.add_pixel_noise(pixel_pos)
        K = self.K
        R = self.R
        Bp = np.linalg.inv(K @ R)
        vec = Bp @ util.cart2hom(pixel_pos).astype('int32') # cast to integer because actual pixels are discrete
        vec = vec/np.linalg.norm(vec)
        return vec
    
    def get_measurement(self, target_pos, label):
        """
        Bearing measurements in pixel coordinate (and target identity) 
        The input here is temporary until we have trained the neural net to actually identify targets
        """
        
        bearing = self.get_bearing(target_pos)
        if bearing[0] < 0:
            return None
        
        # target is unknow if it is too far
        if self._get_distance(target_pos) > self.range:
            label = -1
        # return np.hstack([bearing, label])
        return (bearing, label)

class CamGroup():
    def __init__(self, cam_param, cam_poses):
        self.n_cams = cam_poses.shape[0]
        self.cam_param = cam_param
        self.cam_poses = cam_poses
        self.cam_nodes = [CamNode(cam_param, poses[0:3], poses[3:6]) for poses in cam_poses]
        
    def get_group_measurement(self, target_pos):
        z = []
        for node in self.cam_nodes:
            z.append(node.get_bearing(target_pos))
        return z

def prior_fn(n, spawn_region=None):
    if spawn_region is not None:
        pos = np.random.normal(spawn_region, 5*np.ones(3), (n,3))
        vel = np.random.normal(np.zeros(3), np.ones(1), (n,3))
        label = np.random.randint(0, 2, (n,1))
        particles = np.hstack([pos, vel, label])

        print("particles", particles.shape)
        return particles
    else:
        pos_x = np.random.uniform(-20, 20, (n,1))
        pos_y = np.random.uniform(-20, 20, (n,1))
        pos_z = np.random.uniform(10, 25, (n,1))
        vel_x = np.random.normal(0, 1, (n,1))
        vel_y = np.random.normal(0, 1, (n,1))
    
        vel_z = np.random.normal(0, 1, (n,1))
        label = np.random.randint(0, 2, (n,1))
        particles = np.hstack([pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, label])
        print("particles", particles.shape)
        return particles

def squared_error(x, y, sigma=2000):
    '''
    x: the set of particles hypothesis
    y: measurement from camera nodes 
    '''
    dx = (x - y) ** 2
    d = np.ma.sum(dx, axis=1)
    weights = np.exp(-d / (2.0 * sigma ** 2))
    return weights

def weight_fn(self, z_set, hypo, n_particles):
    """
    Iterate through each camera, find the measurement most likely to be associated with the target
    @ z_set: [[y0, y1, ...], [y0, y1, ...], ...]
    @ hypo: [bearing0, bearing1, ....]
    @ n_particles: number of particles
    """
    y = []
    weight = np.ones(n_particles)
    for i, zi in enumerate(z_set):
        Si = 0
        wi = None
        yi = None
        if zi is None: # no measurement from this camera
            wi = np.ones(n_particles)
            Si = n_particles
        else:
            for yj in zi:
                w_tmp = util.squared_error(hypo[:, i, :], yj, sigma=self.sigma)
                S_tmp = sum(w_tmp)
                if S_tmp > Si:
                    wi = w_tmp
                    Si = S_tmp
                    yi = yj
        weight *= wi
    return weight

def main():
    cam_param = [642.0926, 642.0926, 1000.5, 1000.5,0]
    # [x y z roll pitch yaw]
    cam_poses = np.array([
        # [20, 20, 12, 0, 0, -2.2],
        [20, -20, 12, 0, 0, 2.2],
        [-20, -20, 12, 0, 0, 0.7],
        [-20, 20, 12, 0, 0, -0.7],
    ])

    particles = prior_fn(2000)
    # plt.figure()
    # plt.plot(particles[:,0], particles[:,1], '.')
    # # plt.show()
    group = CamGroup(cam_param, cam_poses)
    targets = np.array([[15,0,20],[-10,5,10]])

    z_set = group.get_group_measurement(targets)
    print(z_set)
    weights = []
    hypothesis = []
    for p in particles:
        pos = p[0:3]
        h = group.get_group_measurement(pos)
        h = np.array(h)
        hypothesis.append(h)
    hypothesis = np.array(hypothesis)
    # print(type(h))
    print(hypothesis.shape)
    print(hypothesis.reshape(2000, 3*2).shape)
    for i in range(hypothesis.shape[1]):
        hi = hypothesis[:, i, :]
        print("hi", hi.shape)
        wi = util.squared_error(hi, z_set[i], sigma=300)
        Si = sum(wi)
        weights.append(wi/Si)
    weights = np.array(weights).T

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,4,1, projection="3d")
    # ax = plt.axes(projection="3d")
    # ax.plot_trisurf(particles[:,0], particles[:,1], weights[:,0], 
    #                 cmap=cm.jet, linewidth=0.2, antialiased=True)
    ax.scatter3D(particles[:,0], particles[:,1], weights[:,0],
            cmap=cm.turbo,c=weights[:,0])
    ax.scatter(group.cam_nodes[0].pos[0], group.cam_nodes[0].pos[1], 0, marker="s", s=200,)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')

    ax = fig.add_subplot(1,4,2, projection="3d")
    ax.scatter3D(particles[:,0], particles[:,1], weights[:,1],
            cmap=cm.turbo,c=weights[:,1])
    ax.scatter(group.cam_nodes[1].pos[0], group.cam_nodes[1].pos[1], 0, marker="s", s=200)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')

    ax = fig.add_subplot(1,4,3, projection="3d")
    ax.scatter3D(particles[:,0], particles[:,1], weights[:,2],
            cmap=cm.turbo,c=weights[:,2])
    ax.scatter(group.cam_nodes[2].pos[0], group.cam_nodes[2].pos[1], 0, marker="s", s=200)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')

    ax = fig.add_subplot(1,4,4, projection="3d")
    weights_final = weights[:,0] * weights[:,1] * weights[:,2]
    weights_final /= np.sum(weights_final)
    ax.scatter3D(particles[:,0], particles[:,1], weights_final,
            cmap=cm.turbo,c=weights_final)
    # ax.plot3D(group.cam_nodes[1].pos[0], group.cam_nodes[1].pos[1], 0, "s")
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    plt.show()

if __name__=="__main__":
    main()