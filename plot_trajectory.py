"""
The purpose of this script is to plot the Rendezvous3DOF trajectories generated by the evaluate_policy function.
The callback in evaluate_policy() generates a pickle dictionary that contains all the attributes of the environment.
Among these attributes, 'trajectory' contains the x-y-z coordinates of the chaser during the episode.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
# from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from math import sin, cos, pi, radians, degrees
from scipy.spatial.transform import Rotation as R
import time
from plot_attitude_trajectory import generate_slider
from os import path as os_path


def load_data(args):
    """
    Load the data from a pickle file.\n
    :param args: path to file.
    :return: dictionary with data
    """
    path = args.path

    if len(path) == 0:
        print('Please use the "--path" argument to specify the path to the trajectory file.\nExiting')
        exit()

    print(f'Loading file "{path}"...')
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data


def plot_animation(args, data):
    """
    Make a 3d plotly animation showing the trajectory of the chaser around the target.\n
    :param args: Namespace containing the arguments from the command line.
    :param data: Dictionary containing the trajectory data.
    :return: None
    """

    # Chaser position data:
    x = data['trajectory'][0]
    y = data['trajectory'][1]
    z = data['trajectory'][2]

    # Check that the required info about the target is available:
    if 'target_radius' in data:
        target_radius = data['target_radius']
    else:
        target_radius = 5
        print(f'Target radius not defined. Using default value: {target_radius} m')
    if 'cone_half_angle' in data:
        cone_half_angle = data['cone_half_angle']
    else:
        cone_half_angle = radians(30)
        print(f'Corridor angle not defined. Using default value: {round(degrees(cone_half_angle), 2)} deg')
    if 'corridor_axis' in data:
        corridor_vector = data['corridor_axis']
    else:
        corridor_vector = np.array([0, -1, 0])
        print(f'Corridor axis not defined. Using default value: {corridor_vector}')

    # Make surface object for the Keep-out zone:
    koz_x, koz_y, koz_z = create_koz_points(target_radius, cone_half_angle, corridor_vector)
    target = go.Surface(
        x=koz_x,
        y=koz_y,
        z=koz_z,
        opacity=1,
        opacityscale=[[0, 1], [0.99, 0.5], [1, 0.2]],
        surfacecolor=koz_x ** 2 + koz_y ** 2 + koz_z ** 2,
        colorscale=[[0, 'rgb(100,20,20)'], [1, 'rgb(200,30,30)']],
        showscale=False,
        name='Keep-out zone',
        showlegend=True)

    # Make a cube to represent the target body:
    target_x, target_y, target_z = create_cube_points(0, 0, 0, width=1)
    target_color = 'rgb(60,60,150)'
    target_body, target_edges = create_cube(target_x, target_y, target_z, name='Target', face_color=target_color)

    # Make Scatter3d object for the trajectory
    chaser_trajectory = go.Scatter3d(x=x, y=y, z=z,
                                     line={'color': 'rgb(50,150,50)', 'dash': 'solid', 'width': 4},
                                     marker={'size': 2, 'color': 'rgb(50,50,50)'},
                                     name='Trajectory',
                                     showlegend=True)

    # Make a cube to represent the chaser:
    chaser_x, chaser_y, chaser_z = create_cube_points(x[-1], y[-1], z[-1])
    chaser_body, chaser_edges = create_cube(chaser_x, chaser_y, chaser_z, name='Chaser')

    # Make the figure:
    if 'viewer_bounds' in data:
        lim = data['viewer_bounds']
    else:
        lim = 50
        print(f'Plot limits not defined. Using default value: {lim} m')
    fig_dict = {
        'data': [target, chaser_trajectory, chaser_body, chaser_edges, target_body, target_edges],
        'layout': {
            'scene': dict(
                xaxis=dict(range=[-lim, lim], zerolinecolor="black"), xaxis_showspikes=False,
                yaxis=dict(range=[-lim, lim], zerolinecolor="black"), yaxis_showspikes=False,
                zaxis=dict(range=[-lim, lim], zerolinecolor="black"), zaxis_showspikes=False),
            'width': 800,
            'scene_aspectmode': 'cube',
            'scene_camera': define_camera(),
            'title': 'Rendezvous trajectory',
            'updatemenus': [{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 0},
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

    # Update the time interval between the animation's frames:
    dt = data['dt']
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = dt * 500

    # Compute corridor rotation:
    if 'w_norm' in data and 'w_mag' in data:
        w_norm = data['w_norm']
        w_mag = data['w_mag']
    else:
        w_norm = np.array([0, 0, 1])
        w_mag = 0
        print('Target rotation not defined. Plotting static target instead.')
    euler_axis = w_norm
    theta = w_mag * dt
    quaternion = np.append(euler_axis * np.sin(theta/2), np.cos(theta/2))
    rot = R.from_quat(quaternion)  # Rotation of the target during each time step

    # Create the frames:
    frames = []
    for i in range(0, len(x)):
        # Update the keep-out zone:
        koz_x, koz_y, koz_z = rotate_points(koz_x, koz_y, koz_z, rot)
        current_koz = go.Surface(x=koz_x, y=koz_y, z=koz_z)
        # Update the target:
        target_x, target_y, target_z = rotate_points(target_x, target_y, target_z, rot)
        target_body, target_edges = create_cube(target_x, target_y, target_z, face_color=target_color)
        # Update the trajectory:
        current_trajectory = go.Scatter3d(x=x[0:i+1], y=y[0:i+1], z=z[0:i+1])
        # Update the chaser:
        chaser_x, chaser_y, chaser_z = create_cube_points(x[i], y[i], z[i])
        current_body, current_edges = create_cube(chaser_x, chaser_y, chaser_z)
        # Define new frame:
        frame = {"data": [current_koz, current_trajectory, current_body, current_edges, target_body, target_edges],
                 "name": str(i), "traces": [0, 1, 2, 3, 4, 5]}  # 'traces' indicates which trace we are updating.
        frames.append(frame)
    fig.frames = tuple(frames)
    sliders = generate_slider(fig)
    fig.update_layout(sliders=sliders)

    if args.save:  # Save the animation as an html file
        _, dataname = os_path.split(args.path)
        filename = str(dataname.split('.')[0]) + '_anim.html'
        fig.write_html(os_path.join('plots', filename))
        print(f'Animation saved in: "plots\{filename}"')

    if args.show:  # Display the animation
        fig.show()

    return


def create_koz_points(r: float, angle: float, vec: np.ndarray=None) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    This function creates the array of points for the target.
    The target is a sphere with a conical cut-out pointed in the +z direction.\n
    :param r: radius of the sphere
    :param angle: half-angle of the cone cut-out
    :param vec: vector in the direction of the corridor axis. If not given, corridor points into z-axis.
    :return: 3 numpy arrays (x, y, and z coordinates of the points on the sphere)
    """

    # Points on the sphere:
    u, v = np.mgrid[0:2 * pi:40j, 0:pi:20j]
    x_sphere = np.cos(u) * np.sin(v) * r
    y_sphere = np.sin(u) * np.sin(v) * r
    z_sphere = np.cos(v) * r

    # Make the cone (pointed in the +z direction by default):
    cone_mask = z_sphere > r * cos(angle)
    dist_from_z = np.sqrt(x_sphere[cone_mask] ** 2 + y_sphere[cone_mask] ** 2)  # Distance of each point from the z-axis
    z_sphere[cone_mask] = z_sphere[cone_mask] - r * (1 - dist_from_z / np.max(dist_from_z))

    # Rotate the cone to match the direction of the corridor:
    if vec is not None:
        z_vec = np.array([0, 0, 1])  # Vector pointing in the +z direction (initial direction of the cone)
        cross = np.cross(z_vec, vec)  # Cross product of the current cone direction and the desired corridor direction
        euler_axis = cross / np.linalg.norm(cross)
        theta = np.arccos(np.dot(z_vec, vec) / (np.linalg.norm(z_vec) * np.linalg.norm(vec)))
        quat = np.append(euler_axis * np.sin(theta/2), np.cos(theta/2))
        rot = R.from_quat(quat)  # Rotation from the current cone direction to the desired corridor direction
        x_sphere, y_sphere, z_sphere = rotate_points(x_sphere, y_sphere, z_sphere, rot)
    return x_sphere, y_sphere, z_sphere


def rotate_points(x: np.ndarray, y: np.ndarray, z: np.ndarray, rot: R) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    This function rotates a given set of points.\n
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


def create_cube_points(pos_x, pos_y, pos_z, width=1) -> (np.ndarray, np. ndarray, np.ndarray):
    """
    This function creates a set of points in the shape of a cube centered on a given location.\n
    :param pos_x: x-position of the center of the cube
    :param pos_y: y-position of the center of the cube
    :param pos_z: z-position of the center of the cube
    :param width: width of the cube
    :return: 3 ndarrays (x, y, and z coordinates of the points)
    """
    d = width / 2
    x = np.array([-d, -d, d, d, -d, -d, d, d]) + pos_x
    y = np.array([-d, d, d, -d, -d, d, d, -d]) + pos_y
    z = np.array([-d, -d, -d, -d, d, d, d, d]) + pos_z
    return x, y, z


def create_cube(x_points, y_points, z_points, name=None, face_color=None) -> (go.Mesh3d, go.Scatter3d):
    """
    This function creates a cube from a given set of points.\n
    :param x_points: x-coordinates of the points
    :param y_points: y-coordinates of the points
    :param z_points: z-coordinates of the points
    :param name: name shown in the legend of the plot
    :param face_color: color code for the faces of the cube, in rgb(#,#,#) format
    :return: two graph objects (one for the surface of the cube and one for the edges)
    """
    if face_color is None:
        face_color = 'rgb(50,50,50)'
    mesh = go.Mesh3d(
        # 8 vertices of the cube:
        x=x_points,
        y=y_points,
        z=z_points,
        # i, j and k give the vertices of the mesh triangles:
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        color=face_color,
        opacity=1,
        flatshading=True,
        name=name,
        showlegend=True
    )
    ind = [0, 3, 2, 1, 0, 4, 7, 3, 7, 6, 2, 6, 5, 1, 5, 4]  # indices of the points to join with lines
    lines = go.Scatter3d(
        x=[x_points[i] for i in ind],
        y=[y_points[i] for i in ind],
        z=[z_points[i] for i in ind],
        mode='lines',
        hoverinfo=None,
        showlegend=False,
        line=dict(color='black')
    )
    return mesh, lines


def define_camera(up: list=None, center: list=None, eye: list=None) -> dict:
    """
    Create a dictionary that defines the camera settings for a plotly 3d figure.\n
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


def plot2d(args, data):
    """
    Plot a 2d matplotlib figure to show the position and the control inputs of the chaser over time.\n
    :param args: Namespace containing the arguments from the command line.
    :param data: Dictionary containing the trajectory data.
    :return: None
    """

    if 'trajectory' in data:
        states = data['trajectory']
    else:
        states = data['state']
    x, y, z = states[0], states[1], states[2]
    vx, vy, vz = states[3], states[4], states[5]
    pos, vel = states[0:3], states[3:]

    # Create a matplotlib figure with two subplots:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

    # Plot the position of the chaser over time:
    t = range(len(x))
    plot_components(ax1, t, x, y, z,
                    labels=['x', 'y', 'z'], xlabel='Time (s)', ylabel='Distance (m)', title='Position')
    # Plot the overall distance from the target:
    dist = np.linalg.norm(pos, axis=0)
    ax1.plot(t, dist, 'k--', label='Overall'), ax1.legend()

    # Plot the velocity of the chaser over time:
    plot_components(ax2, t, vx, vy, vz,
                    labels=[r'$v_x$', r'$v_y$', r'$v_z$'], xlabel='Time (s)', ylabel='Velocity (m/s)', title='Velocity')
    # Plot the overall speed of the chaser:
    speed = np.linalg.norm(vel, axis=0)
    ax2.plot(t, speed, 'k--', label='Overall'), ax2.legend()

    # Plot the actions of the chaser over time:
    if 'actions' in data:
        actions = data['actions']
    else:
        actions = data['action']
    action_x, action_y, action_z = actions[0], actions[1], actions[2]
    t = range(len(action_x))
    sum_of_actions = sum(np.abs(action_x) + np.abs(action_y) + np.abs(action_z))
    plot_components(ax3, t, action_x, action_y, action_z,
                    labels=['x', 'y', 'z'], xlabel='Time (s)', ylabel='Delta V (m)',
                    title='Actions. Total ' + r'$\Delta V = $' + str(round(sum_of_actions, 2)))
    # Plot the overall control effort:
    ax3.plot(t, np.linalg.norm(actions, axis=0), 'k--', label='Overall'), ax3.legend()

    plt.tight_layout()

    if args.save:  # Save the 2d plot as a png file
        _, dataname = os_path.split(args.path)
        filename = str(dataname.split('.')[0]) + '_2d.png'
        plt.savefig(os_path.join('plots', filename))
        print(f'2d plot saved in: "plots\{filename}"')

    if args.show:  # Display the 2d plot
        plt.show()
        plt.close()

    return


def plot_components(ax: plt.Axes, x, y1, y2, y3,
                    labels: list=None, xlabel: str=None, ylabel: str=None, title: str=None):
    """
    Plot 3 sets of data onto a set of axes.
    Useful for plotting the individual components of a 3-dimensional quantity.\n
    :param ax: Axes object to plot the data on
    :param x: x-axis data
    :param y1: first y-axis data
    :param y2: second y-axis data
    :param y3: third y-axis data
    :param labels: list of labels to plot as legends
    :param xlabel: label for the x-axis
    :param ylabel: label for the y-axis
    :param title: title for the plot
    :return: None
    """
    if y1 is not None:
        ax.plot(x, y1, label=labels[0])
    if y2 is not None:
        ax.plot(x, y2, label=labels[1])
    if y3 is not None:
        ax.plot(x, y3, label=labels[2])
    ax.set_xlabel(xlabel)
    ax.set_xlim(left=0, right=x[-1])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid()
    return


def get_args():
    """
    Get the arguments (to run the script from the command line).\n
    :return: Namespace containing the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', type=str, default='',
                        help='Path to the data file.')
    parser.add_argument('--type', dest='type', type=str, default='all',
                        help='Type of plot. Default creates animation and 2d plot ')
    parser.add_argument('--save', dest='save', type=bool, nargs='?', const=True, default=False,
                        help='Use this flag to save the plots.')
    parser.add_argument('--show', dest='show', type=bool, nargs='?', const=True, default=False,
                        help='Use this flag to show the plots.')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    start = time.perf_counter()
    arguments = get_args()
    trajectory_data = load_data(arguments)
    if not arguments.show:
        print('"--show" was not called. Plots will not be displayed.')
    if not arguments.save:
        print('"--save" was not called. Plots will not be saved.')
    if arguments.type == '2d':
        plot2d(arguments, trajectory_data)
    elif arguments.type == 'anim':
        plot_animation(arguments, trajectory_data)
    else:
        plot_animation(arguments, trajectory_data)
        plot2d(arguments, trajectory_data)

    print(f'Finished on {time.ctime()}. ({round(time.perf_counter()-start, 2)} seconds)')
