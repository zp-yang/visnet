import util
import numpy as np
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
            return np.array([[-1, -1]])
        
        bearing = self._get_pixel_pos(target_pos)
        # if dist > self.range:
        #     return self.add_pixel_noise(bearing, sigma=20).reshape((1,2))
        return self.add_pixel_noise(bearing, sigma=10).reshape((1,2))
        
    def add_pixel_noise(self, pixel_coord, sigma=5):
        return pixel_coord + np.random.normal(0, sigma)
    
class CamNode(Camera):
    def __init__(self, cam_param, cam_pos, cam_att):
        super(CamNode, self).__init__(cam_param, cam_pos, cam_att)
        self.max_false_alert = 3
        
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
    
    def get_measurement(self, targets):
        """
        Bearing measurements in pixel coordinate (and target identity) 
        The input here is temporary until we have trained the neural net to actually identify targets
        """
        
        dim = targets.shape
        bearings = np.empty([0,2])
        for target_pos in targets:
            bearing = self.get_bearing(target_pos).reshape((1,2))
            bearings = np.vstack([bearings,bearing])

        # add false alerts
        # rand = np.random.uniform()
        # if rand < -0.1:
        #     n_false_alert = np.random.randint(0, self.max_false_alert)
        #     bearing = np.vstack([bearing, np.random.uniform(0, 2000, size=(n_false_alert,2))])
        
        # return np.hstack([bearing, label])
        return bearings

class CamGroup():
    def __init__(self, cam_param, cam_poses):
        self.n_cams = cam_poses.shape[0]
        self.cam_param = cam_param
        self.cam_poses = cam_poses
        self.cam_nodes = [CamNode(cam_param, poses[0:3], poses[3:6]) for poses in cam_poses]
        
    def get_group_measurement(self, targets, hypo_mode=False):
        z = []
        for node in self.cam_nodes:
            z.append(node.get_measurement(targets))
        if hypo_mode: # each cam only gives one measurement of a particle
            return z
        else: # for simulation where each cam can have false measurments
            # TODO Add noisy measurements?
            return z
            

cam_param = [642.0926, 642.0926, 1000.5, 1000.5,0]
# [x y z roll pitch yaw]
cam_poses = np.array([
    [20, 20, 12, 0, 0, -2.2],
    [20, -20, 12, 0, 0, 2.2],
    [-20, -20, 12, 0, 0, 0.7],
    [-20, 20, 12, 0, 0, -0.7],
])