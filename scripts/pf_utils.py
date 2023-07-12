import numpy as np

class TargetTrack:
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

def dynamics_d(x, sigma=1):
    """
    Discrete dynamics, last state is the target label
    """
    n, d = x.shape
    w = np.zeros((d, n))
    w[3:6] = np.random.normal(0, sigma, (3,n))
    x_1 = A @ x.T + w

    return x_1.T