from numpy.core.arrayprint import printoptions
import util
import numpy as np
class Camera:
    def __init__(self, cam_param, cam_pos, cam_att):
        self.param = cam_param
        self.pos = cam_pos
        self.att = cam_att
        self.R_cam = util.so3_exp(self.att)
        self.K = util.get_cam_in(self.param)

        # this rotation is kinda goofy but corrects the gazebo convention and literature convention
        self.R = util.R_model2cam @ self.R_cam 
        
        self.P = util.get_cam_mat_lie(self.param, self.pos, self.att)
        self.range = 25 # meters
        self.fov = np.deg2rad(90) #degrees, both directions
        self.bound = 2000
        
    def __repr__(self):
        return "param: {}, pos: {}, att: {}".format(self.param, self.pos, self.att)
        
    def _get_pixel_pos_hom(self, target_pos):
        return self.P @ util.cart2hom(target_pos)
    
    def _get_pixel_pos(self, target_pos):
        pix_hom = self._get_pixel_pos_hom(target_pos)
        pix_cart = pix_hom[0:-1, :] / pix_hom[-1,:]
        return pix_cart
    
    def _get_distance(self, target_pos):
        """
        !!!Should not be used directly in estimation!!!
        target_pos: (n, 3)
        """
        dist =  np.linalg.norm(target_pos-self.pos, axis=1)
        return dist
    
    def get_bearing(self, target_pos):
        """
        Check if projectd pixel is within the picture bounds
        target_pos should be (3, n) shaped
        """        
        bearing = self._get_pixel_pos(target_pos)
        
        mask = (bearing > 0) & (bearing < self.bound)
        mask = mask[0] & mask[1] # True if within bound, False if out of bound
        indices = np.where(mask==False)[0] # Find the indices of target that are out of bound
        
        n = bearing.shape[1]
        
        # add noise to the pixel locations due to varying bounding box size
        bearing = bearing + np.random.normal(0, 10, size=(2, n))
        
        bearing[:, indices] = np.array([-1, -1]).reshape(2,1)
        return bearing, indices
        
    def add_pixel_noise(self, pixel_coord, sigma=5):
        return pixel_coord + np.random.normal(0, sigma, size=[1,2])
    
class CamNode(Camera):
    def __init__(self, cam_param, cam_pos, cam_att):
        super(CamNode, self).__init__(cam_param, cam_pos, cam_att)
        self.max_false_alert = 3
        
    def _get_vec(self, pixel_pos):
        """
        Back projected ray from the camera to the target (approximate)
        """
        K = self.K
        R = self.R
        Bp = np.linalg.inv(K @ R)
        vec = Bp @ util.cart2hom(pixel_pos).astype('int32') # cast to integer because actual pixels are discrete
        vec = vec/np.linalg.norm(vec)
        return vec
    
    def get_measurement(self, targets, labels=None):
        """
        Bearing measurements in pixel coordinate (and target identity: drone, bird, unknown) 
        The input here is temporary until we have trained the neural net to actually identify targets
        """
        bearing, oob_indices = self.get_bearing(targets.T)

        dist = self._get_distance(targets)
        dist_mask = dist > self.range
        oor_indices = np.where(dist_mask==True) # find targets that are out of range

        labels[oob_indices] = -1
        labels[oor_indices] = -1
        msmts = np.vstack([bearing, labels])

        return msmts.T

class CamGroup():
    def __init__(self, cam_param, cam_poses):
        self.n_cams = cam_poses.shape[0]
        self.cam_param = cam_param
        self.cam_poses = cam_poses
        self.cam_nodes = [CamNode(cam_param, poses[0:3], poses[3:6]) for poses in cam_poses]
        
    def get_group_measurement(self, targets, labels):
        z = []
        for node in self.cam_nodes:
            z.append(node.get_measurement(targets, labels))    
        return z
            

cam_param = [642.0926, 642.0926, 1000.5, 1000.5,0]
# [x y z roll pitch yaw]
cam_poses = np.array([
    [20, 20, 12, 0, 0, -2.2],
    [20, -20, 12, 0, 0, 2.2],
    [-20, -20, 12, 0, 0, 0.7],
    [-20, 20, 12, 0, 0, -0.7],
])