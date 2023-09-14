import numpy as np

def cart2hom(points):
    """
    points shape should be (3, n)
    """
    n_points = points.shape[1]
    return np.vstack([points, np.ones(n_points)])

def hom2cart(coord):
    coord = coord[0:-1]/coord[-1]
    return coord

class Camera:
    def __init__(self, K, pose):
        self.K = K
        self.T_cam = pose

        self.range = 25 # meters
        self.fov = np.deg2rad(90) #degrees, both directions
        self.bound = 2000

    def update_pose(self, pose):
        self.pose = pose
    
    def _project_points(self, points):
        # points shape should be (3, n)
        points_h = np.vstack([points, np.ones((1,points.shape[1]))])
        points_fcam = self.T_cam.M @ points_h
        points_fcam = points_fcam[0:3, :] / points_fcam[2, :]
        pix_pts = self.K @ points_fcam
        return pix_pts[0:2, :]
    
    def _get_distance(self, target_pos):
        """
        !!!Should not be used directly in estimation!!!
        target_pos: (n, 3)
        """
        dist =  np.linalg.norm(target_pos-self.T_cam.c, axis=1)
        return dist
    
    def get_bearing(self, target_pos):
        """
        Check if projectd pixel is within the picture bounds
        target_pos should be (3, n) shaped
        """        
        bearing = self._project_points(target_pos)
        
        mask = (bearing > 0) & (bearing < self.bound)
        mask = mask[0] & mask[1] # True if within bound, False if out of bound
        indices = np.where(mask==False)[0] # Find the indices of target that are out of bound
        
        # n = bearing.shape[1]
        # add noise to the pixel locations due to varying bounding box size
        # bearing = bearing + np.random.normal(0, 5, size=(2, n))
        
        bearing[:, indices] = np.array([-1, -1]).reshape(2,1)
        return bearing, indices
        
    def add_pixel_noise(self, pixel_coord, sigma=5):
        return pixel_coord + np.random.normal(0, sigma, size=[1,2])
    
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
    def __init__(self, cam_params, cam_poses):
        self.n_cams = len(cam_poses)
        self.cam_params = cam_params
        self.cam_poses = cam_poses

        self.cam_nodes = []
        for i in range(self.n_cams):
            self.cam_nodes.append(Camera(self.cam_params[i], self.cam_poses[i]))

    def get_group_measurement(self, targets, labels):
        z = []
        for node in self.cam_nodes:
            z.append(node.get_measurement(targets, labels))    
        return z
    
    def update_poses(self, poses):
        pass