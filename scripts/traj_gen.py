#%%
import numpy as np
import matplotlib.pyplot as plt
import bezier

#%%
## Straight segement
points = np.array([
    [20],
    [-40],
])

duration_nominal = 15
duration_attack = 15

nodes_nominal = np.asfortranarray([
    [20, 20, 10, -10, -38],
    [ 0, 12, 10, 10, 10]
])

nodes_attack = np.asfortranarray([
    [20, 20, 0,  -35,],
    [ 0, 15, 5,   20,]
])

curve_nominal = bezier.Curve(nodes_nominal, degree=nodes_nominal.shape[1]-1)
curve_attack = bezier.Curve(nodes_attack, degree=nodes_attack.shape[1]-1)


s = np.linspace(0.0, 1.0, 100)
t_list_nominal = np.hstack([0, duration_nominal * (1+s)])
t_list_attack = np.hstack([0, duration_attack * (1+s)])

points_nominal = np.hstack([points, curve_nominal.evaluate_multi(s)])
points_attack = np.hstack([points, curve_attack.evaluate_multi(s)])

fig = plt.figure()
ax = plt.axes()
ax.plot(points_nominal[0, :], points_nominal[1, :])
ax.scatter(nodes_nominal[0, :], nodes_nominal[1, :], marker="x", color="blue")
ax.plot(points_attack[0,:], points_attack[1, :])
ax.scatter(nodes_attack[0, :], nodes_attack[1, :], marker="s", color="red")
ax.set_aspect(1)
ax.grid()

point_diff_nominal = np.diff(points_nominal)
heading_nominal = np.arctan2(point_diff_nominal[1,:], point_diff_nominal[0,:])
heading_nominal = np.hstack([heading_nominal, heading_nominal[-1]])
heading_nominal[heading_nominal < 0] += 2*np.pi


point_diff_attack = np.diff(points_attack)
heading_attack = np.arctan2(point_diff_attack[1,:], point_diff_attack[0,:])
heading_attack = np.hstack([heading_attack, heading_attack[-1]])
heading_attack[heading_attack < 0] += 2*np.pi


plt.figure()
plt.plot(heading_nominal)
plt.plot(heading_attack)
# %%

data_nominal = np.vstack([t_list_nominal, points_nominal, heading_nominal]).T
data_attack = np.vstack([t_list_attack, points_attack, heading_attack]).T

import os
data_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), os.pardir)) + "/data/"
np.savetxt(data_dir+"hk_nominal_traj.csv", data_nominal, fmt='%.4e', delimiter=',')
np.savetxt(data_dir+"hk_attack_traj.csv", data_attack, fmt='%.4e', delimiter=',')


# %%
