import numpy as np
import camera

class Track:
    """
    This is a data structure for representing tracked targets
    """
    def __init__(self, n_particles: int, x0=None, label=None):
        self.n_particles = n_particles
        self.map_state = None # MAP estimate, particle with the max weight
        self.mean = None
        self.cov = None
        self.weights = np.ones(n_particles)
        self.particles = None
        self.label = label
        self.trajectory = []
        
        self.cam_group = None
        self.hypothesis = None
        self.map_hypo = None # MAP estimate of pixel locations on different cameras

        if x0 is not None:
            x0 = x0 + np.random.normal(np.zeros(3), np.ones(3))
            pos = np.random.normal(x0, [1,1,1], size=(n_particles, 3))
            vel = np.random.normal([0,0,0], [1,1,1], size=(n_particles, 3))
            labels = np.ones((n_particles,1)) * label
            self.particles = np.hstack([pos, vel, labels])
            self.map_state = x0
        else:
            pos = np.random.uniform([-20,-20,0], [20,20,25], size=(n_particles, 3))
            vel = np.random.normal([0,0,0], [1,1,1], size=(n_particles, 3))
            self.particles = np.hstack([pos, vel])
    
    def set_map_hypo(self, map_hypo):
        self.map_hypo = map_hypo

    def set_map_state(self, map_state):
        self.map_state = map_state
    
    def update(self, weights, particles, resample_fn):
        """
        Update track after particle filter update
        """
        self.particles = particles
        self.weight_sum = np.sum(weights)
        self.weights = weights / self.weight_sum
        # mean and cov
        self.mean_state = np.sum(self.particles.T * self.weights, axis=-1).T
        self.cov_state = np.cov(self.particles, rowvar=False, aweights=self.weights)
        
        # Maximum-a-posterior MAP
        argmax_weight = np.argmax(self.weights)
        self.map_state = self.particles[argmax_weight]
        self.trajectory.append(self.map_state[0:3])
                
        indices = resample_fn(weights)
        self.particles = self.particles[indices, :]
        self.weights = np.ones(self.n_particles) / self.n_particles
        
        return particles, weights
    
    def set_cam_group(self, cam_group: camera.CamGroup):
        self.cam_group = cam_group