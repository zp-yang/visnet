import numpy as np
import camera
import util

class Track:
    """
    This is a data structure for representing tracked targets
    """
    def __init__(self, n_particles: int, x0=None, label=None):
        self.n_particles = n_particles
        self.mean_state = None
        self.cov_state = None
        self.weights = np.ones(n_particles)
        self.particles = None
        self.label = label
        self.trajectory = []
        self.particles_hist = []
        self.weights_hist = []
        
        if x0 is not None:
            x0 = x0 + np.random.normal(np.zeros(3), np.ones(3))
            pos = np.random.normal(x0, np.ones(3), size=(n_particles, 3))
            vel = np.random.normal(np.zeros(3), np.ones(3), size=(n_particles, 3))
            labels = np.ones((n_particles,1)) * label
            self.particles = np.hstack([pos, vel, labels])
            self.mean_state = x0
        else:
            pos = np.random.uniform([-20,-20,0], [20,20,25], size=(n_particles, 3))
            vel = np.random.normal([0,0,0], [1,1,1], size=(n_particles, 3))
            labels = np.ones((n_particles,1)) * label
            self.particles = np.hstack([pos, vel, labels])
            self.mean_state = np.random.normal(np.zeros(3), np.ones(3))
    
    def update(self, weights, particles):
        """
        Update track after particle filter update
        """
        self.particles = particles
        # normalizing weight
        self.weight_sum = np.sum(weights)
        self.weights = weights / self.weight_sum

        self.particles_hist.append(particles)
        self.weights_hist.append(weights)
        
        # mean and cov
        self.mean_state = np.sum(self.particles.T * self.weights, axis=-1).T
        self.cov_state = np.cov(self.particles, rowvar=False, aweights=self.weights)
        
        self.trajectory.append(self.mean_state[0:3])
        
        # resampling step and reset weights
        n = len(weights)
        indices = np.random.choice(n, n, p=weights)
        self.particles = self.particles[indices, :]
        self.weights = np.ones(self.n_particles) / self.n_particles
        

