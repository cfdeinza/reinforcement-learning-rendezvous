# import matplotlib.pyplot as plt
# from matplotlib import cm
import numpy as np
from math import pi, e, tan, radians
# from sys import exit
import plotly.graph_objects as go

"""
This script plots the reward as a function of xy position relative to the chaser. 
It is useful to visualize where the highest rewards are (because in theory that is where the chaser would go).
"""


def distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the distance from the origin.\n
    :param x: x-values
    :param y: y-values
    """
    return np.sqrt(x**2 + y**2)


def angle_from_corridor(x, y):
    """
    Compute the angles between the points and the -y axis. Based on the cosine rule, simplified for the -y axis.\n
    :param x: x-values
    :param y: y-values
    """
    return np.arccos(-y / distance(x, y))  # np.arctan2(a, -b)


def current_reward(x: np.ndarray, y: np.ndarray) -> np.ndarray:

    # Distance component of the reward:
    dist = distance(x, y)
    rew_dist = (100 - dist) * 1e-2

    # Angle component of the reward:
    rew_angle = np.zeros(x.shape)
    angle = angle_from_corridor(x_mesh, y_mesh)
    angle_mask = np.abs(angle) < cone_half_angle
    rew_angle[angle_mask] += 0.1  # Current

    # Collision component of the reward
    rew_collision = np.zeros(x.shape)
    collision_mask = (dist < target_radius) & (np.abs(angle) > cone_half_angle)
    rew_collision[collision_mask] -= 0.1  # Current

    rew = rew_dist + rew_angle + rew_collision
    return rew


def phi(dist, dist_max):
    potential = (1 - dist / dist_max) ** 2
    return potential


def desired_reward(x, y):
    # # Distance component of the reward:
    # dist = distance(x, y)
    # rew_dist = (100 - dist) * 1e-3
    #
    # # Collision component of the reward
    # angle = angle_from_corridor(x_mesh, y_mesh)
    # angle_mask = angle < cone_half_angle
    # sigmoid = 0.1 * (1 / (1 + e ** (5 - dist)) - 1)  # The sigmoid function penalizes coming too close to the target
    # theta = angle  # This is a coefficient for the sigmoid function to avoid penalizing the corridor
    # theta[angle_mask] = (theta[angle_mask] / cone_half_angle) ** 2
    # theta[~angle_mask] = 1
    # rew_collision = sigmoid * theta

    # dist = distance(x_mesh, y_mesh)
    dist = distance(x, y)
    dist_max = 30
    angle = angle_from_corridor(x, y)

    # Distance component:
    # rew_distance = (1 - angle / pi) * (1 - dist / 100)**3
    rew_distance = phi(dist, dist_max)

    if dist < 5 and angle > cone_half_angle:
        rew_distance = 2 * phi(5, dist_max) - phi(dist, dist_max)

    # Collision component:
    sigmoid = 1 / (1 + e ** (5 - dist)) - 1  # Sigmoid function to penalize proximity to target
    rew_collision = (angle / pi) * sigmoid  # Multiply sigmoid with angle to avoid penalizing corridor

    k_distance = 1e-1  # Coefficient for the distance component
    k_collision = 1  # Coefficient for the collision component
    # rew = k_distance * rew_distance + k_collision * rew_collision
    rew = rew_distance
    return rew


# Define the meshgrid:
lim = 20  # Furthest point (in the positive and negative directions)
step = 0.1  # Step between gridpoints
x_vals = np.arange(-lim, lim + step, step)  # x-values
y_vals = np.arange(-lim, lim + step, step)  # y-values

x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)  # Meshgrid

# Define the geometric properties of the target and the approach corridor:
target_radius = 5
cone_half_angle = radians(30)

rows, cols = x_mesh.shape

z = x_mesh

for i in range(rows):
    for j in range(cols):
        z[i, j] = desired_reward(x_mesh[i, j], y_mesh[i, j])


# z = desired_reward(x_mesh, y_mesh)


def plot_cone(limit, half_angle, axis=None):
    """
    Plot a 2d cone to represent the approach corridor.\n
    """
    x_cone = np.array([-1, 0, 1]) * limit * tan(half_angle)
    y_cone = np.array([-1, 0, -1]) * limit
    z_cone = np.zeros(x_cone.shape)  # + np.mean(z)
    if axis is not None:
        axis.plot3D(x_cone, y_cone, z_cone, 'k-', linewidth=3)
    return x_cone, y_cone, z_cone


def plot_target(radius, axis=None):
    """Plot a circle to represent the target."""
    phi = np.linspace(0, 2*pi, 30)
    x_target = np.cos(phi) * radius
    y_target = np.sin(phi) * radius
    z_target = np.zeros(x_target.shape)  # + np.mean(z)
    if axis is not None:
        axis.plot3D(x_target, y_target, z_target, 'k-', linewidth=3)
    return x_target, y_target, z_target


def plot_chaser(r0):
    x = r0[0]
    y = r0[1]
    return np.array([x]), np.array([y]), desired_reward(np.array([x]), np.array([y]))


# Plotly figure:
cone = plot_cone(lim, cone_half_angle)
target = plot_target(target_radius)
chaser = plot_chaser(np.array([0, -20]))
f = go.Figure(data=[
              go.Surface(z=z, x=x_vals, y=y_vals, opacity=1),
              go.Scatter3d(x=cone[0], y=cone[1], z=cone[2],
                           marker=dict(size=4), line=dict(width=4), name='Corridor'),
              go.Scatter3d(x=target[0], y=target[1], z=target[2],
                           marker=dict(size=2), line=dict(width=4), name='Target'),
              go.Scatter3d(x=chaser[0], y=chaser[1], z=chaser[2],
                           marker=dict(size=2), line=dict(width=4), name="Chaser")
              ])
f.update_layout(title='Reward function',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                scene=dict(
                   xaxis_title='x (m)',
                   yaxis_title='y (m)',
                   zaxis_title='Reward',
                   camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.25, y=-1.25, z=1.25))
                ))
f.show()

# Matplotlib figure:
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# # p = ax.scatter(x, y, z, c=z)
# p = ax.plot_surface(x_mesh, y_mesh, z, cmap=cm.coolwarm, linewidth=0)
# # ax.contour3D(x, y, z, 50)
# ax.set_zlim3d([np.min(z), np.max(z)])
# # ax.set_zlim3d([0, np.max(z)])
# ax.set_xlabel('x (m)'), ax.set_ylabel('y (m)'), ax.set_zlabel('Reward')
# ax.set_title('Reward function')
# fig.colorbar(p)
# plot_cone(lim, cone_half_angle, ax)
# plot_target(target_radius, ax)
# plt.show()
