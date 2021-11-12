"""
The purpose of this script is to plot the trajectories generated by the evaluate_policy function for the Attitude() env.
The callback in evaluate_policy generates a pickle dictionary for each trajectory.
Currently, the keys of the dictionary are 'state', and 'action'.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go
from math import pi


def plot3d_animation(states):
    """
    Make a 3d animation of the rotation of the body frame.
    :param states: Array of states.
    :return
    """
    # Create the figure:
    fig = make_3d_plotly_figure(states[:, 0])

    # Generate frames:
    frames = []
    for t in range(len(states[0])):
        frame = {"data": generate_frame_data(states[:, t])[3:], "traces": [3, 4, 5, 6], "name": str(t)}
        frames.append(frame)
    fig.frames = tuple(frames)

    # Draw slider: (Issue: The sliders appear to slow the animation down)
    sliders = generate_slider(fig)
    fig.update_layout(sliders=sliders)

    # Plot:
    fig.show()
    return


def make_3d_plotly_figure(state):
    """
    Make the initial instance of the 3d figure.
    :param state: The initial state of the system
    :return: Plotly Figure object
    """
    # Set the camera settings:
    camera = dict(
        up=dict(x=0, y=0, z=1),  # Determines the 'up' direction on the page.
        center=dict(x=0, y=0, z=0),  # The projection of the center point lies at the center of the view.
        eye=dict(x=1.25, y=-1.25, z=1.25)  # The eye vector determines the position of the camera.
    )

    # Make figure:
    lim = 10
    fig_dict = {
        'data': generate_frame_data(state),
        'layout': {
            'scene': dict(
                xaxis=dict(nticks=8, range=[-lim, lim], zerolinecolor="black"), xaxis_showspikes=False,
                yaxis=dict(nticks=8, range=[-lim, lim], zerolinecolor="black"), yaxis_showspikes=False,
                zaxis=dict(nticks=8, range=[-lim, lim], zerolinecolor="black"), zaxis_showspikes=False),
            'width': 800,
            'scene_aspectmode': 'cube',
            'scene_camera': camera,
            'title': 'Rotation',
            'updatemenus': [{
                "buttons": [{
                    "args": [None, {"frame": {"duration": 15},
                                    'mode': 'immediate',
                                    "fromcurrent": True,
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
    return fig


def generate_frame_data(state, lim=10):
    """
    Generate the plotly traces for a frame of the animation.
    :param state: Current state
    :param lim: Limit for the graph
    :return: List containing the plotly traces for the frame
    """
    # Inertial axes:
    len_i = lim * 0.8  # Length of the inertial axes
    width_i = 4  # Width of the body axes
    ms = 1  # Marker size
    x_i = go.Scatter3d(
        x=[0, len_i],
        y=[0, 0],
        z=[0, 0],
        line={'color': 'darkred', 'dash': 'solid', 'width': width_i},
        marker={'size': ms},
        name='x_inertial')
    y_i = go.Scatter3d(
        x=[0, 0],
        y=[0, len_i],
        z=[0, 0],
        line={'color': 'darkgreen', 'dash': 'solid', 'width': width_i},
        marker={'size': ms},
        name='y_inertial')
    z_i = go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[0, len_i],
        line={'color': 'darkblue', 'dash': 'solid', 'width': width_i},
        marker={'size': ms},
        name='z_inertial')

    # Body axes:
    len_b = lim / 2  # Length of the body axes
    width_b = 8  # Width of the body axes
    vecs = np.array([
        [len_b, 0, 0],
        [0, len_b, 0],
        [0, 0, len_b]
    ])
    rotation = R.from_quat(state[0:4])
    rot_vecs = rotation.apply(vecs)  # Rotated vectors
    x_b = go.Scatter3d(
        x=[0, rot_vecs[0, 0]],
        y=[0, rot_vecs[0, 1]],
        z=[0, rot_vecs[0, 2]],
        line={'color': 'red', 'dash': 'solid', 'width': width_b},
        marker={'size': ms},
        name='x_body')
    y_b = go.Scatter3d(
        x=[0, rot_vecs[1, 0]],
        y=[0, rot_vecs[1, 1]],
        z=[0, rot_vecs[1, 2]],
        line={'color': 'green', 'dash': 'solid', 'width': width_b},
        marker={'size': ms},
        name='y_body')
    z_b = go.Scatter3d(
        x=[0, rot_vecs[2, 0]],
        y=[0, rot_vecs[2, 1]],
        z=[0, rot_vecs[2, 2]],
        line={'color': 'blue', 'dash': 'solid', 'width': width_b},
        marker={'size': ms},
        name='z_body')

    # Angular velocity:
    len_w = 100
    w = state[4:]
    # w_mag = np.linalg.norm(w)
    # if w_mag < 1e-4:  # Avoid division by zero
    #     w_mag = 1
    w_i = rotation.apply(w)  # Angular velocity expressed in the inertial frame (rad/s)
    vel = go.Scatter3d(
        x=[0, w_i[0] * len_w],
        y=[0, w_i[1] * len_w],
        z=[0, w_i[2] * len_w],
        line={'color': 'black', 'dash': 'solid', 'width': 10},
        marker={'size': 1},
        name='angular velocity')
    data_list = [x_i, y_i, z_i, x_b, y_b, z_b, vel]
    return data_list


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


def plot2d(x, a):
    """
    Plot the attitude, velocity, and actions over time.
    :param x: Array of states
    :param a: Array of actions
    :return:
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
    t = range(len(x[0]))

    alpha = np.zeros((len(t),))*np.nan
    beta = np.zeros((len(t),))*np.nan
    gamma = np.zeros((len(t),))*np.nan
    for i in t:
        q = x[0:4, i]
        rot = R.from_quat(q)
        alpha[i], beta[i], gamma[i] = rot.as_euler('ZYX', degrees=True)

    ax1.plot(t, alpha, label=r'$\alpha$')
    ax1.plot(t, beta, label=r'$\beta$')
    ax1.plot(t, gamma, label=r'$\gamma$')
    ax1.set_xlabel('Time (s)'), ax1.set_xlim([0, t[-1]])
    ax1.set_ylabel('Attitude (deg)')
    ax1.set_title('Attitude')
    ax1.legend()
    ax1.grid()

    ax2.plot(t, x[4]*180/pi, label='$w_x$')
    ax2.plot(t, x[5]*180/pi, label='$w_y$')
    ax2.plot(t, x[6]*180/pi, label='$w_z$')
    ax2.set_xlabel('Time (s)'), ax2.set_xlim([0, t[-1]])
    ax2.set_ylabel('Angular velocity (deg/s)')
    ax2.set_title('Velocity')
    ax2.legend()
    ax2.grid()

    action_x = a[0]
    action_y = a[1]
    action_z = a[2]
    # t = range(len(action_x))
    ax3.plot(t, action_x, label='x')
    ax3.plot(t, action_y, label='y')
    ax3.plot(t, action_z, label='z')
    ax3.set_xlabel('Time (s)'), ax3.set_xlim([0, t[-1]])
    ax3.set_ylabel('Torque (N.m)')
    ax3.set_title('Action')
    ax3.legend()
    ax3.grid()

    plt.tight_layout()
    plt.show()
    plt.close()

    return


def get_args():
    """
    Get the arguments when the script is executed from the command line.
    :return: Namespace containing the directory and name of the pickle file
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', dest='dir', type=str, default='logs/')  # Directory of the data file
    parser.add_argument('--file', dest='file', type=str, default='att_trajectory_0.pickle')  # Name of the data file
    parser.add_argument('--type', dest='type', type=str, default='3d')  # Type of plot (2d or 3d)
    args = parser.parse_args()

    return args


def get_data(args):
    """
    Retrieve the data from the pickle file
    :param args: Namespace containing the directory and name of the pickle file.
    :return: Arrays of states and actions
    """
    path = args.dir + args.file
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    states = data['state'][:, 1:]
    actions = data['action'][:, 1:]

    return states, actions


if __name__ == '__main__':
    arguments = get_args()
    sta, act = get_data(arguments)  # Get states and actions

    if arguments.type == '3d':
        plot3d_animation(sta)
    elif arguments.type == '2d':
        plot2d(sta, act)
    else:
        plot3d_animation(sta)
        plot2d(sta, act)

print('Finished.')
