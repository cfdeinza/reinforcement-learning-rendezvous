"""
This module contains utility functions for plotting 3D animations with plotly.

Written by C. F. De Inza Niemeijer.
"""

import numpy as np
from math import radians, degrees, pi, cos
from scipy.spatial.transform import Rotation as scipyRot
import plotly.graph_objects as go
import pickle
import matplotlib.pyplot as plt

# Compute corridor rotation:
# euler_axis = w_norm
# theta = w_mag * dt
# quaternion = np.append(euler_axis * np.sin(theta/2), np.cos(theta/2))
# rot = scipyRot.from_quat(quaternion)  # Rotation of the target during each time step


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

    print(f'Loading file "{path}"...', end=' ')
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    print('Successful.')
    return data


def get_trajectory_data(data):
    """
    Check if environment properties are present. If not replace with default values.\n
    :param data: Dictionary containing trajectory data.
    :return: Dictionary containing trajectory data.
    """
    if 'viewer_bounds' in data:
        lim = data['viewer_bounds'] if 'viewer_bounds' in data else 50
    else:
        lim = 50
        print(f'Plot limits not defined. Using default value: {lim} m')
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
    if 'w_norm' in data and 'w_mag' in data:
        w_norm = data['w_norm']
        w_mag = data['w_mag']
    else:
        w_norm = np.array([0, 0, 1])
        w_mag = 0
        print('Target rotation not defined. Plotting static target instead.')

    return lim, target_radius, cone_half_angle, w_norm, w_mag


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
        rot = scipyRot.from_quat(quat)  # Rotation from the current cone direction to the desired corridor direction
        x_sphere, y_sphere, z_sphere = rotate_points(x_sphere, y_sphere, z_sphere, rot)
    return x_sphere, y_sphere, z_sphere


def create_koz(x_points, y_points, z_points, opacity=1, legend=False):
    """
    Create a plotly graph_object for the keep-out zone.\n
    :param x_points: x-coordinates of the points
    :param y_points: y-coordinates of the points
    :param z_points: z-coordinates of the points
    :param opacity: Opacity of the surface
    :param legend: Display legend on figure
    :return: graph_object.Surface
    """
    surf = go.Surface(
            x=x_points,
            y=y_points,
            z=z_points,
            opacity=opacity,
            opacityscale=[[0, 1], [0.99, 0.9], [1, 0.1]],
            surfacecolor=x_points ** 2 + y_points ** 2 + z_points ** 2,
            colorscale=[[0, 'rgb(100,20,20)'], [1, 'rgb(200,30,30)']],
            showscale=False,
            name='Keep-out zone',
            showlegend=legend)

    return surf


def rotate_points(x: np.ndarray, y: np.ndarray, z: np.ndarray, rot: scipyRot) -> (np.ndarray, np.ndarray, np.ndarray):
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
    :return: 3 arrays (x, y, and z coordinates of the points)
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
        showlegend=False
    )
    x_edges, y_edges, z_edges = create_cube_edge_points(x_points, y_points, z_points)
    lines = go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        mode='lines',
        hoverinfo=None,
        name=name,
        showlegend=False,
        line=dict(color='black')
    )
    return mesh, lines


def create_cube_edge_points(x_points, y_points, z_points):
    """
    Create the x, y, and z coordinates of the points needed to draw the edges of the cube.\n
    :param x_points: x-coordinates of the satellite points
    :param y_points: y-coordinates of the satellite points
    :param z_points: z-coordinates of the satellite points
    :return: 3 arrays (x, y, and z coordinates of the edge points)
    """
    ind = [0, 3, 2, 1, 0, 4, 7, 3, 7, 6, 2, 6, 5, 1, 5, 4]  # indices of the points to join with lines
    x_edges = [x_points[i] for i in ind]
    y_edges = [y_points[i] for i in ind]
    z_edges = [z_points[i] for i in ind]
    return x_edges, y_edges, z_edges


def create_sat_points(pos_x, pos_y, pos_z, vec: np.ndarray=None):
    """
    Creates a set of points in the shape of a satellite centered on a given location.\n
    :param pos_x: x-coordinate of the center of the satellite
    :param pos_y: y-coordinate of the center of the satellite
    :param pos_z: z-coordinate of the center of the satellite
    :param vec: vector in the direction the satellite must be facing (points in -y direction by default)
    :return: 3 arrays (x, y, and z coordinates of the points)
    """
    w, h, d = 1.5, 1.5, 1
    pw, ph, pd = 2*2.5*w, 0.8*h, 0.1*d
    jw, jh, jd = 0.1*2*w, 0.2*h, 0.2*d
    x = np.array([w, -w, -w, w,                     # body (-y)
                  w, -w, -w, w,                     # body (+y)
                  w+jw, w, w, w+jw,                 # joint (+x)
                  w+jw+pw, w+jw, w+jw, w+jw+pw,     # panel (+x)
                  -w-jw, -w, -w, -w-jw,             # join (-x)
                  -w-jw-pw, -w-jw, -w-jw, -w-jw-pw  # panel (-x)
                  ]) / 2 + pos_x
    y = np.array([-d, -d, -d, -d,   # body (-y)
                  d, d, d, d,       # body (+y)
                  0, 0, 0, 0,       # joint (+x)
                  0, 0, 0, 0,       # panel (+x)
                  0, 0, 0, 0,       # joint (-x)
                  0, 0, 0, 0        # panel (-x)
                  ]) / 2 + pos_y
    z = np.array([h, h, -h, -h,         # body (-y)
                  h, h, -h, -h,         # body (+y)
                  jh, jh, -jh, -jh,     # joint (+x)
                  ph, ph, -ph, -ph,     # panel (+x)
                  jh, jh, -jh, -jh,     # joint (-x)
                  ph, ph, -ph, -ph      # panel (-x)
                  ]) / 2 + pos_z

    # Rotate the body to match the direction of the corridor:
    if vec is not None:
        vec_0 = np.array([0, -1, 0])  # Vector pointing in the -y direction (initial direction of the target)
        cross = np.cross(vec_0, vec)  # Cross product of the initial target direction and the desired corridor direction
        # print(f'Target: {cross}, {np.linalg.norm(cross)}')
        # print(f'Target: {vec}, {np.degrees(angle_between_vectors(vec_0, vec))}')
        euler_axis = cross / np.linalg.norm(cross)
        theta = np.arccos(np.dot(vec_0, vec) / (np.linalg.norm(vec_0) * np.linalg.norm(vec)))
        quat = np.append(euler_axis * np.sin(theta / 2), np.cos(theta / 2))
        rot = scipyRot.from_quat(quat)  # Rotation from the current cone direction to the desired corridor direction
        x, y, z = rotate_points(x, y, z, rot)
    return x, y, z


def create_sat(x_points, y_points, z_points, name=None):
    """
    Creates a plotly graph_object in the shape of a satellite.\n
    :param x_points: x-coordinates of the points
    :param y_points: y-coordinates of the points
    :param z_points: z-coordinates of the points
    :param name: name shown in the legend of the plot
    :return: two graph objects (one for the surface of the satellite and one for the edges)
    """
    b = 'rgb(255,240,0)'  # color for body
    p = 'rgb(0, 230, 250)'  # color for panel
    mesh = go.Mesh3d(
        x=x_points,
        y=y_points,
        z=z_points,
        # i, j and k give the vertices of the mesh triangles:
        # first 12 triangles are for the cube,
        # then 2 for right joint, 2 for right panel, 2 for left joint, 2 for left panel
        # | -y | +y  | +z  | -z  | +x  | -x  |joint| panel | joint | panel |
        i=[0, 0, 4, 4, 0, 0, 6, 6, 0, 0, 1, 1, 8, 8, 12, 12, 16, 16, 20, 20],
        j=[1, 3, 5, 7, 1, 4, 7, 2, 3, 4, 2, 5, 9, 11, 13, 15, 17, 19, 21, 23],
        k=[2, 2, 6, 6, 5, 5, 3, 3, 7, 7, 6, 6, 10, 10, 14, 14, 18, 18, 22, 22],
        facecolor=[b, b, b, b, b, b, b, b, b, b, b, b, b, b, p, p, b, b, p, p],
        opacity=1,
        flatshading=True,
        name=name,
    )
    x_edges, y_edges, z_edges = create_sat_edge_points(x_points, y_points, z_points)
    lines = go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        mode='lines',
        hoverinfo=None,
        showlegend=False,
        line=dict(color='black')
    )
    return mesh, lines


def create_sat_edge_points(x_points, y_points, z_points):
    """
    Create the x, y, and z coordinates of the points needed to draw the edges of the satellite.\n
    :param x_points: x-coordinates of the satellite points
    :param y_points: y-coordinates of the satellite points
    :param z_points: z-coordinates of the satellite points
    :return: 3 arrays (x, y, and z coordinates of the edge points)
    """
    ind = [0, 3, 2, 1, 0, 4, 7, 3, 7, 6, 2, 6, 5, 1, 5, 4]  # indices of the points to join with lines
    ind2 = [10, 11, 14, 15, 12, 13, 8, 9, 17, 16, 21, 20, 23, 22, 19, 18]
    x_edges = [x_points[i] for i in ind] + [0] + [x_points[i] for i in ind2]
    y_edges = [y_points[i] for i in ind] + [0] + [y_points[i] for i in ind2]
    z_edges = [z_points[i] for i in ind] + [0] + [z_points[i] for i in ind2]
    return x_edges, y_edges, z_edges


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
        eye = [1.25, -1.25, 1.25]  # default
        # eye = [0.52, -0.52, 0.52]  # more zoomed-in
        # eye = [-0.2, -0.2, 0.4]  # even more zoomed-in

    camera = dict(
        up=dict(x=up[0], y=up[1], z=up[2]),
        center=dict(x=center[0], y=center[1], z=center[2]),
        eye=dict(x=eye[0], y=eye[1], z=eye[2])
    )
    return camera


def generate_slider(figure):
    """
    Create a slider for the 3d animation.
    :param figure: A plotly figure (with frames already defined)
    :return: A list of dictionaries to define the sliders
    """
    duration = 0  # (ms?)
    frame_args = {
        'frame': {'duration': duration},
        'mode': 'immediate',
        'fromcurrent': True,
        'transition': {'duration': duration, 'easing': 'linear'}
    }
    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args],
                    "label": f.name,  # str(round(k*dt, 1)),
                    "method": "animate",
                }
                for k, f in enumerate(figure.frames)
            ],
        }
    ]
    return sliders


def plot_2dcomponents(ax: plt.Axes, x, y1, y2, y3,
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
