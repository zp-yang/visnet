from math import *
import random
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches


class Robot:
    def __init__(self, length=20.0):
        self.x = random.random() * world_size
        self.y = random.random() * world_size
        self.orientation = random.random() * 2.0 * pi
        self.length = length
        self.bearing_noise = 0.0
        self.steering_noise = 0.0
        self.distance_noise = 0.0

    def __repr__(self):
        return '[x=%.6s y=%.6s orient=%.6s]' % (str(self.x), str(self.y),
             str(self.orientation))

    def set(self, new_x, new_y, new_orientation):
        if new_orientation < 0 or new_orientation >= 2 * pi:
            raise ValueError('Orientation must be in [0..2pi]')
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)

    def set_noise(self, new_b_noise, new_s_noise, new_d_noise):
        self.bearing_noise = float(new_b_noise)
        self.steering_noise = float(new_s_noise)
        self.distance_noise = float(new_d_noise)

    def sense(self):
        Z = []
        for i in range(len(landmarks)):
            dx = landmarks[i][0] - self.x
            dy = landmarks[i][1] - self.y
            bearing = atan2(dy, dx) - self.orientation
            if dy < 0:
                bearing += 2 * pi
            Z.append(bearing)
        return Z

    def measurement_prob(self, measurements):
        predicted_measurements = self.sense()  # Our sense function took 0 as an argument to switch off noise.
        error = 1.0
        for i in range(len(measurements)):
            error_bearing = abs(measurements[i] - predicted_measurements[i])
            error_bearing = (error_bearing + pi) % (2.0 * pi) - pi  # truncate
            error *= (exp(- (error_bearing ** 2) / (self.bearing_noise ** 2) / 2.0) /
                      sqrt(2.0 * pi * (self.bearing_noise ** 2)))
        return error

    def move(self, motion):
        alphainit = motion[0]
        alpha = random.gauss(alphainit, steering_noise) % (2 * pi)
        d = motion[1] + random.gauss(0, distance_noise)
        if alphainit > max_steering_angle:
            raise ValueError('Car cannot turn with angle greater than pi/4')
        while alpha > max_steering_angle:
            alpha = random.gauss(alphainit, steering_noise) % (2 * pi)
        if alpha == 0.0:
            beta = 0.0
        else:
            beta = (d / self.length) * tan(alpha)
            R = d / beta

        if abs(beta) < 0.001:
            # straight movement
            x = self.x + d * cos(self.orientation)
            y = self.y + d * sin(self.orientation)
            thetanew = (self.orientation + beta) % (2 * pi)
        else:
            CX = self.x - sin(self.orientation) * R
            CY = self.y + cos(self.orientation) * R
            x = CX + sin(self.orientation + beta) * R
            y = CY - cos(self.orientation + beta) * R
            thetanew = (self.orientation + beta) % (2 * pi)

        self.set(x, y, thetanew)
        self.set_noise(bearing_noise, steering_noise, distance_noise)


# TEST CASES:
max_steering_angle = pi / 4.0
bearing_noise = 0.1
steering_noise = 0.1
distance_noise = 5.0
length = 20.0

tolerance_xy = 15.0
tolerance_orientation = 0.25
landmarks = [[0.0, 100.0], [0.0, 0.0], [100.0, 0.0], [100.0, 100.0]]  # position of 4 landmarks in (y, x) format.
world_size = 100.0  # world is NOT cyclic. Robot is allowed to travel "out of bounds"


bearing1 = []
bearing2 = []
bearing3 = []
bearing4 = []
coords = [(76.0, 94.0)]
orientations = []
codes = [Path.MOVETO]

myrobot = Robot(length)
myrobot.set(76.0, 94.0, 2.0)
myrobot.set_noise(bearing_noise, steering_noise, distance_noise)

print('Robot:        ', myrobot)
print('Bearing measurements: ', myrobot.sense())

for i in range(100):
    myrobot.move([random.uniform(0,pi/4), random.randint(0,10)])
    bearings = myrobot.sense()
    bearing1.append(bearings[0])
    bearing2.append(bearings[1])
    bearing3.append(bearings[2])
    bearing4.append(bearings[3])
    coords.append((myrobot.x, myrobot.y))
    codes.append(Path.LINETO)
    orientations.append(myrobot.orientation)
    print('Robot:        ', myrobot)
    print('Bearing measurements: ', myrobot.sense())


path = Path(coords, codes)

fig, ax = plt.subplots()
patch = patches.PathPatch(path, facecolor="white", lw=2)
ax.add_patch(patch)
ax.set_xlim(0, 120)
ax.set_ylim(0, 120)
plt.show()
