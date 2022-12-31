"""
This script shows how to compute a more favorable initial orientation for the target, given its rotation rate.
The new orientation places the entry corridor closer to the chaser, and also gives the chaser enough time to reach the
corridor before the target starts turning away. The steps to compute the new orientation are as follows:
- Define the point on the surface of the KOZ along the corridor axis as the desired entry point (re)
- Compute the circle that the entry point traces as the target makes one revolution
- Find the point on that circle that is closest to the chaser (re_min)
- Compute the time (t1) that it would take the chaser to reach re_min
- Compute the angular displacement (theta) of the entry point during a time duration t1
- Place the new entry point theta degrees ahead of re_min
Setting this new orientation is the equivalent of the agent waiting for the moment when the entry corridor is turning
toward the chaser. It ensures that the chaser does not start its trajectory when the corridor is unnecessarily far away,
and it gives the chaser enough time to reach the corridor before it starts turning away.
"""

import numpy as np
from utils.environment_utils import make_new_env, print_state
from utils.general import rotate_vector_about_axis, angle_between_vectors
from utils.quaternions import rot2quat, quat_product
import matplotlib.pyplot as plt
from math import degrees


# Create a new environment:
config = dict(
    # qt0=np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]),
    # qt0_range=0,
    # wt0=np.radians(np.array([0, 2, 0.1])),  # Expressed in LVLH here only (env.wt is expressed in T)
    # wt0_range=0,
    dt=0.5,
    qt0_range=np.radians(180),
)
env = make_new_env(config)

# Reset the environment and print the initial state:
env.reset()
print_state(env)

# Define the entry point
rc = env.rc                                         # Position of the chaser [m] (LVLH)
koz_radius = env.koz_radius                         # radius of the KOZ [m]
wt = env.wt                                         # Target rotation rate [rad/s] (T frame)
wt_mag = np.linalg.norm(wt)                         # Magnitude of the target rot. rate [rad/s]
re = env.rd / np.linalg.norm(env.rd) * koz_radius   # Entry point (edge of the KOZ along corr axis) (T frame)
wt_lvlh = env.target2lvlh(wt)                       # Target rotation rate [rad/s] (LVLH)
re_lvlh = env.target2lvlh(re)                       # Entry point [m] (LVLH)
print(f"Initial entry point: {re_lvlh} (LVLH)")

# Compute the circle that the entry point traces throughout one rotation:
angles = np.linspace(0, 2*np.pi - 0.17, 72)
vecs = []
for angle in angles:
    vecs.append(rotate_vector_about_axis(re_lvlh, wt_lvlh, angle))
all_re_points = np.vstack(vecs)  # (LVLH)
x = all_re_points[:, 0]
y = all_re_points[:, 1]
z = all_re_points[:, 2]

# Find the point on the circle closest to the chaser initial position:
r_diff = np.linalg.norm(all_re_points - rc, axis=1)  # (LVLH)
index_min = np.argmin(r_diff)
re_min = all_re_points[index_min]  # (LVLH)
angle_min = angles[index_min]
print(f"min entry point: {re_min} m (LVLH)")
# print(f"min angle: {round(degrees(angle_min), 3)} deg")

# Compute how long it takes the chaser to reach that point:
a = env.thrust / env.mc * env.bt / env.dt
print(f"Axial acceleration from thruster: {round(a, 3)} m/s^2")
coef = 1  # Coefficient to add more time if necessary
min_time_to_entry = coef * np.sqrt(2*np.linalg.norm(rc - re_min)/a)
print(f"Minimum time for chaser to reach entry point: {round(min_time_to_entry, 3)} s")

# Compute the angular displacement of the entry point during that time:
angle_back = wt_mag * min_time_to_entry
print(f"Angle traced by entry point in that time: {round(degrees(angle_back), 3)} deg")

# Rotate the target to the new orientation:
qt_min = rot2quat(axis=wt, theta=angle_min)  # Rotation from qt0 to qt_min
qt_new = rot2quat(axis=wt, theta=-angle_back)  # Rotation from from qt_min to qt_new

env.qt = quat_product(env.qt, qt_min)  # First rotate the target to place the entry point at re_min
re_lvlh_min = env.target2lvlh(re)  # LVLH

env.qt = quat_product(env.qt, qt_new)  # Then rotate the target to place the entry point enough degrees ahead of re_min
re_lvlh_back = env.target2lvlh(re)  # LVLH

# Verify that the angular displacement between re_min and re_new is as large as expected:
component = wt_lvlh / wt_mag * np.linalg.norm(re_lvlh_min) * np.cos(angle_between_vectors(wt_lvlh, re_lvlh_min))
angle_btw_min_and_back = angle_between_vectors(re_lvlh_min-component, re_lvlh_back-component)
print(f"(Verify) Angle between re_min and re_back: "
      f"{round(degrees(angle_btw_min_and_back), 3)} deg")

# Plot the results:
fig2 = plt.figure(num='3D', clear=True, figsize=(9, 6))
ax2 = plt.axes(projection="3d")
ax2.plot3D([0], [0], [0], 'r+', label="Origin LVLH")
ax2.plot3D(x, y, z)
ax2.plot3D(x[0:1], y[0:1], z[0:1], 'k.', label="Old entry point")
# ax2.plot3D(x[-1:], y[-1:], z[-1:], 'rx', label="End")
wt_plot = wt_lvlh / np.linalg.norm(wt_lvlh) * koz_radius
ax2.plot3D([0, wt_plot[0]], [0, wt_plot[1]], [0, wt_plot[2]], 'k--',
           label=f"$\omega_T$ = {round(degrees(wt_mag), 3)} deg/s")
ax2.plot3D([rc[0]], rc[1], rc[2], "g*", label="$r_C$")
ax2.plot3D([rc[0], x[index_min]], [rc[1], y[index_min]], [rc[2], z[index_min]], 'g--')
ax2.plot3D(re_lvlh_min[0], re_lvlh_min[1], re_lvlh_min[2], 'gx', label="min")
ax2.plot3D(re_lvlh_back[0], re_lvlh_back[1], re_lvlh_back[2], 'rx', label="New entry point")
ax2.set_box_aspect((1, 1, 1))
lim = 11
ax2.set_xlim([-lim, lim])
ax2.set_ylim([-lim, lim])
ax2.set_zlim([-lim, lim])
plt.xlabel('$X_{LVLH}$ (m)'), plt.ylabel('$Y_{LVLH}$ (m)')
plt.legend()
plt.title("Computing a favorable initial target orientation")
plt.show()
plt.close()
