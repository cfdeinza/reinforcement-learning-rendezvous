"""
The purpose of this script is to plot the Rendezvous3DOF trajectories generated by the evaluate_policy function.
The callback in evaluate_policy generates a pickle dictionary for each trajectory.
Currently, the keys of the dictionary are 'x', 'y', 'z', and 'action' (switched to 'state' and 'action').
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
# from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from math import sin, cos, pi, radians
from scipy.spatial.transform import Rotation as R
import time
from plot_attitude_trajectory import generate_slider


def load_data(args):
    """
    Load the data from a pickle file.
    :param args: path to file. Namespace with arguments
    :return: dictionary with data
    """
    path = args.dir + args.file
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data


def anim(args):
    """
    Make a 3d plotly animation showing the trajectory of the chaser around the target.
    # :param args: Namespace containing arguments.
    :param args: path to the pickle file.
    :return: None
    """
    # Animate the trajectory in 3d
    data = load_data(args)
    x = data['state'][0, 1:]
    y = data['state'][1, 1:]
    z = data['state'][2, 1:]

    # Plot a sphere to represent the target:
    target_radius = 5
    rotation = R.from_quat([sin(pi / 4), 0, 0, cos(pi / 4)])  # Rotation that aligns the corridor with the +x axis.
    cone_half_angle = radians(30)
    target_x, target_y, target_z = create_target(target_radius, cone_half_angle)
    target_x, target_y, target_z = rotate_target(target_x, target_y, target_z, rotation)

    # Make surface object for the target:
    target = go.Surface(
        x=target_x,  # x_sphere,
        y=target_y,  # y_sphere,
        z=target_z,  # z_sphere,
        opacity=1,
        surfacecolor=target_x ** 2 + target_y ** 2 + target_z ** 2,
        colorscale=[[0, 'rgb(100,25,25)'], [1, 'rgb(200,50,50)']],
        showscale=False,
        name='Target',
        showlegend=True)

    # Make Scatter3d object for the trajectory (this trace will be updated in each frame)
    chaser = go.Scatter3d(x=x, y=y, z=z,
                          line={'color': 'rgb(50,150,50)', 'dash': 'solid', 'width': 4},
                          marker={'size': 2, 'color': 'rgb(50,50,50)'},
                          name='Trajectory',
                          showlegend=False)

    # Make figure:
    lim = 50
    fig_dict = {
        'data': [target, chaser],
        'layout': {
            'scene': dict(
                xaxis=dict(range=[-lim, lim]), xaxis_showspikes=False,
                yaxis=dict(range=[-lim, lim]), yaxis_showspikes=False,
                zaxis=dict(range=[-lim, lim]), zaxis_showspikes=False),
            'width': 800,
            'scene_aspectmode': 'cube',
            'scene_camera': define_camera(),
            'title': 'Rendezvous trajectory',
            'updatemenus': [{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 15},
                                        'mode': 'immediate', "fromcurrent": True,
                                        "transition": {"duration": 0}}],
                        "label": "Play",
                        "method": "animate"},
                    {
                        "args": [[None], {"frame": {"duration": 0},
                                          "mode": "immediate",
                                          "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"}],
                'type': 'buttons'}]},
        'frames': []}
    fig = go.Figure(fig_dict)

    # Create the frames:
    frames = []
    for i in range(0, len(x)):
        chaser = go.Scatter3d(x=x[0:i], y=y[0:i], z=z[0:i])
        frame = {"data": [chaser], "name": str(i), "traces": [1]}  # 'traces' indicates which trace we are updating.
        frames.append(frame)
    fig.frames = tuple(frames)
    sliders = generate_slider(fig)
    fig.update_layout(sliders=sliders)
    fig.show()

    return


def plot3d(args):
    """
    Make a 3d plotly figure showing the trajectory of the chaser around the target.
    :param args: Namespace containing the directory and name of the pickle file.
    :return: None
    """

    # Load the data:
    data = load_data(args)
    chaser_x = data['state'][0]
    chaser_y = data['state'][1]
    chaser_z = data['state'][2]

    # Plot the trajectory of the chaser:
    trajectory = go.Scatter3d(
        x=chaser_x, y=chaser_y, z=chaser_z,
        line={'color': 'rgb(50,150,50)', 'dash': 'solid', 'width': 4},
        marker={'size': 2, 'color': 'rgb(50,50,50)'},
        name='Trajectory')

    # Plot a sphere to represent the target:
    target_radius = 5
    rotation = R.from_quat([sin(pi/4), 0, 0, cos(pi/4)])
    cone_half_angle = radians(30)
    target_x, target_y, target_z = create_target(target_radius, cone_half_angle)
    target_x, target_y, target_z = rotate_target(target_x, target_y, target_z, rotation)

    # Make surface object:
    target = go.Surface(
        x=target_x,  # x_sphere,
        y=target_y,  # y_sphere,
        z=target_z,  # z_sphere,
        opacity=1,
        surfacecolor=target_x ** 2 + target_y ** 2 + target_z ** 2,
        colorscale=[[0, 'rgb(100,25,25)'], [1, 'rgb(200,50,50)']],
        showscale=False,
        # hovertext='Target',
        name='Target',
        showlegend=True)

    # Create the figure:
    fig = go.Figure()
    fig.add_trace(trajectory)
    fig.add_trace(target)

    # Update the figure's layout:
    lim = 50
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=8, range=[-lim, lim], zerolinecolor="black"),
            yaxis=dict(nticks=8, range=[-lim, lim], zerolinecolor="black"),
            zaxis=dict(nticks=8, range=[-lim, lim], zerolinecolor="black"),
            xaxis_title='x (m)', xaxis_showspikes=False,
            yaxis_title='y (m)', yaxis_showspikes=False,
            zaxis_title='z (m)', zaxis_showspikes=False),
        showlegend=True, width=800, margin=dict(r=10, l=10, b=10, t=10), scene_aspectmode='cube',
        scene_camera=define_camera()
    )

    fig.show()

    return


def create_target(r: float, angle: float):
    """
    This function creates the array of points for the target.
    The target is a sphere with a conical cut-out pointed in the +z direction.
    :param r: radius of the sphere
    :param angle: half-angle of the cone cut-out
    :return: 3 ndarrays (x, y, and z coordinates of the points on the sphere)
    """

    # Points on the sphere:
    u, v = np.mgrid[0:2 * pi:40j, 0:pi:20j]
    x_sphere = np.cos(u) * np.sin(v) * r
    y_sphere = np.sin(u) * np.sin(v) * r
    z_sphere = np.cos(v) * r

    # Make the cone (pointed in the +z direction):
    cone_mask = z_sphere > r * cos(angle)
    dist_from_z = np.sqrt(x_sphere[cone_mask] ** 2 + y_sphere[cone_mask] ** 2)  # Distance of each point from the z-axis
    z_sphere[cone_mask] = z_sphere[cone_mask] - r * (1 - dist_from_z / np.max(dist_from_z))

    return x_sphere, y_sphere, z_sphere


def rotate_target(x: np.ndarray, y: np.ndarray, z: np.ndarray, rot: R):
    """
    This function rotates a set of points representing the target.
    :param x: x-coordinates of the points
    :param y: y-coordinates of the points
    :param z: z-coordinates of the points
    :param rot: rotation to be applied
    :return: 3 ndarrays (x, y, and z coordinates of the rotated points)
    """

    # Rotate the points to the desired direction:
    shape = x.shape  # Original shape of the array
    points = np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))
    new_points = rot.apply(points)

    # Note: The new coordinates need to be reshaped to their original shape (otherwise the target cannot be plotted)
    new_x = new_points[:, 0].reshape(shape)
    new_y = new_points[:, 1].reshape(shape)
    new_z = new_points[:, 2].reshape(shape)
    return new_x, new_y, new_z


def define_camera(up: list=None, center: list=None, eye: list=None) -> dict:
    """
    Create a dictionary that defines the camera settings for a plotly 3d figure.
    :param up: Determines the 'up' direction on the page.
    :param center: The projection of the center point lies at the center of the view.
    :param eye: Determines the position of the camera.
    :return: Dictionary with camera settings.
    """
    if up is None:
        up = [0, 0, 1]
    if center is None:
        center = [0, 0, 0]
    if eye is None:
        eye = [1.25, -1.25, 1.25]

    camera = dict(
        up=dict(x=up[0], y=up[1], z=up[2]),
        center=dict(x=center[0], y=center[1], z=center[2]),
        eye=dict(x=eye[0], y=eye[1], z=eye[2])
    )
    return camera


def plot2d(args):
    """
    Plot a 2d matplotlib figure to show the position and the control inputs of the chaser over time.
    :param args: Namespace containing the directory and filename of the pickle file.
    :return: None
    """

    # Load the data:
    # path = args.dir + args.file
    data = load_data(args)
    # x = data['x']
    # y = data['y']
    # z = data['z']
    # x[0], y[0], z[0] = np.nan, np.nan, np.nan
    states = data['state']
    x = states[0]
    y = states[1]
    z = states[2]
    vx = states[3]
    vy = states[4]
    vz = states[5]

    # Create a matplotlib figure with two subplots:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

    # Plot the position of the chaser over time:
    t = range(len(x))
    make_2d_plot(ax1, t, x, y, z,
                 labels=['x', 'y', 'z'], xlabel='Time (s)', ylabel='Distance (m)', title='Position')

    # Plot the velocity of the chaser over time:
    make_2d_plot(ax2, t, vx, vy, vz,
                 labels=[r'$v_x$', r'$v_y$', r'$v_z$'], xlabel='Time (s)', ylabel='Velocity (m/s)', title='Velocity')

    # Plot the actions of the chaser over time:
    actions = data['action']
    action_x = actions[0]
    action_y = actions[1]
    action_z = actions[2]
    t = range(len(action_x))
    make_2d_plot(ax3, t, action_x, action_y, action_z,
                 labels=['x', 'y', 'z'], xlabel='Time (s)', ylabel='Delta V (m)', title='Action')

    plt.tight_layout()
    plt.show()
    plt.close()

    return


def make_2d_plot(ax, x, y1, y2, y3, labels=None, xlabel=None, ylabel=None, title=None):
    """
    Plot 3 sets of data onto a set of axes.
    """
    if y1 is not None:
        ax.plot(x, y1, label=labels[0])
    if y2 is not None:
        ax.plot(x, y2, label=labels[1])
    if y3 is not None:
        ax.plot(x, y3, label=labels[2])
    ax.set_xlabel(xlabel)
    ax.set_xlim([0, x[-1]])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid()
    return


def get_args():
    """
    Get the arguments (to run the script from the command line).
    :return: Namespace containing the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', dest='dir', type=str, default='logs/')  # Directory of the data file
    parser.add_argument('--file', dest='file', type=str, default='rdv_trajectory_0.pickle')  # Name of the file
    parser.add_argument('--type', dest='type', type=str, default='a')  # Type of plot (2d or 3d)
    args = parser.parse_args(args=[])

    return args


if __name__ == '__main__':
    start = time.perf_counter()
    arguments = get_args()
    if arguments.type == '3d':
        plot3d(arguments)
    elif arguments.type == '2d':
        plot2d(arguments)
    elif arguments.type == 'a':
        anim(arguments)
    else:
        plot3d(arguments)
        plot2d(arguments)

    print(f'Finished on {time.ctime()}. ({time.perf_counter()-start} seconds)')


# def plot3d(args):
#     """
#     [This function is no longer in use because it uses matplotlib (not good for 3d plots). I switched to plotly]
#     Plot the trajectory of a Rendezvous3DOF() environment.
#     :param args: Namespace containing the location and name of the pickle file.
#     :return:
#     """
#
#     path = args.dir + args.file
#
#     with open(path, 'rb') as handle:
#         data = pickle.load(handle)
#
#     fig = plt.figure(figsize=(10, 8))
#     ax = Axes3D(fig)
#
#     actions = data['action']
#     # states = data['state']
#     # x = states[0]
#     # y = states[1]
#     # z = states[2]
#     x = data['x']
#     y = data['y']
#     z = data['z']
#     # x[0], y[0], z[0] = np.nan, np.nan, np.nan
#     lim = 50
#
#     # Draw sphere to represent the target:
#     r = 5
#     u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:30j]
#     sphere_x = np.cos(u) * np.sin(v) * r
#     sphere_y = np.sin(u) * np.sin(v) * r
#     sphere_z = np.cos(v) * r
#     ax.plot_surface(sphere_x, sphere_y, sphere_z, color="r")
#
#     # Plot the trajectory of the chaser:
#     ax.plot3D(xs=x, ys=y, zs=z, label='Trajectory')
#     ax.plot3D(xs=x[0], ys=y[0], zs=z[0], marker='o', color='green', label='Start')
#     ax.plot3D(xs=x[-1], ys=y[-1], zs=z[-1], marker='*', color='black', label='End')
#
#     ax.set_xlim(-lim, lim), ax.set_xlabel('x (m)')
#     ax.set_ylim(-lim, lim), ax.set_ylabel('y (m)')
#     ax.set_zlim(-lim, lim), ax.set_zlabel('z (m)')
#     ax.legend()
#     # ax.set_aspect('equal')
#     ax.set_box_aspect((1, 1, 1))
#     ax.view_init(elev=30, azim=-45)
#     print('(3d plot: Use right-click to zoom)')
#     plt.show()
#     plt.close()
#
#     return
