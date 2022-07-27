"""
This script plots the rendezvous trajectory.
"""

import numpy as np
import time
import os
import argparse
import re
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as scipyRot
from utils.animations import define_camera, generate_slider, load_data


def plot_animation(args, data):
    """
    Make a 3d plotly animation showing the trajectory of the chaser around the target.\n
    :param args: Namespace containing the arguments from the command line.
    :param data: Dictionary containing the trajectory data.
    :return: None
    """

    # Chaser position data:
    rc = data['rc']

    # Chaser attitude data:
    qc = data['qc']

    # Chaser rotation rate data: (expressed in chaser body components)
    wc = data['wc']

    # Check that the required info about the target is available:
    # lim, target_radius, cone_half_angle, w_norm, w_mag = get_trajectory_data(data)

    # Create traces:
    lvlh_length, lvlh_width = (15, 10)
    chaser_length, chaser_width = (10, 5)
    w_scale, w_width = (100, 1)
    lx, ly, lz = draw_frame('LVLH', length=lvlh_length, width=lvlh_width, colors=['darkred', 'darkgreen', 'darkblue'])
    cx, cy, cz = draw_frame('chaser', pos=rc[:, 0], quat=qc[:, 0], length=chaser_length, width=chaser_width)
    chaser_w_line, chaser_w_tip = draw_vector(wc[:, 0], pos=rc[:, 0], quat=qc[:, 0],
                                              scale=w_scale, width=w_width, label='rotation rate')

    # Make the figure:
    lim = 50
    fig_dict = {
        'data': [lx, ly, lz, cx, cy, cz, chaser_w_line, chaser_w_tip],
        'layout': {
            'scene': dict(
                xaxis=dict(range=[-lim, lim]), xaxis_showspikes=False,  # zerolinecolor="black"
                yaxis=dict(range=[-lim, lim]), yaxis_showspikes=False,
                zaxis=dict(range=[-lim, lim]), zaxis_showspikes=False),
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
    # dt = data['dt']
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 0  # dt * 200
    # fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 30
    # fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 5

    # Create the frames:
    frames = []
    for i in range(0, len(rc[0])):
        # Update the chaser:
        cx, cy, cz = draw_frame('chaser', pos=rc[:, i], quat=qc[:, i], length=chaser_length, width=chaser_width)

        # Update the rotation rate:
        chaser_w_line, chaser_w_tip = draw_vector(wc[:, i], pos=rc[:, i], quat=qc[:, i],
                                                  scale=w_scale, width=w_width, label='rotation rate')

        # Define new frame:
        frame = {
            "data": [cx, cy, cz, chaser_w_line, chaser_w_tip],
            "name": str(i),
            "traces": [3, 4, 5, 6, 7],  # 'traces' indicates which trace we are updating.
        }
        frames.append(frame)

    fig.frames = tuple(frames)
    sliders = generate_slider(fig)
    fig.update_layout(sliders=sliders)

    if args.save:  # Save the animation as an html file
        num = re.findall('[0-9]+', args.path)
        if len(num) == 0:
            output_path = os.path.join('plots', 'animation.html')
        else:
            output_path = os.path.join('plots', 'animation' + num[0] + '.html')
        fig.write_html(output_path)
        print(f'Animation saved in: {output_path}')

    if args.show:  # Display the animation
        fig.show()

    return


def draw_frame(label: str, pos=None, quat=None, length=1, width=1, colors=None):
    """
    Draw the three axes (x, y, z) of a reference frame.
    :param label: name of the reference frame.
    :param pos: position of the reference frame.
    :param quat: quaternion (scalar first) describing the orientation of the reference frame.
    :param length: length of the axes.
    :param width: width of the plotted lines.
    :param colors: list containing a color for axes x, y, and z, respectively.
    :return: three Scatter3D graph objects.
    """

    xyz = np.array([
        [length, 0, 0],
        [0, length, 0],
        [0, 0, length],
    ])

    if quat is not None:
        assert quat.shape == (4,), f'Unexpected quaternion shape: {quat.shape}'
        rotation = scipyRot.from_quat(np.append(quat[1:], quat[0]))  # scipy uses scalar-last quaternions
        xyz = rotation.apply(xyz)  # Rotated vectors

    if pos is not None:
        assert pos.shape == (3,), f'Unexpected position shape: {pos.shape}'
        xyz += pos
    else:
        pos = np.array([0, 0, 0])

    if colors is None:
        colors = ['red', 'green', 'blue']
    else:
        assert isinstance(colors, list), 'Argument `colors` must be a list of colors.'
        if len(colors) == 1:
            colors = colors * 3

    x = go.Scatter3d(
        x=[pos[0], xyz[0, 0]],
        y=[pos[1], xyz[0, 1]],
        z=[pos[2], xyz[0, 2]],
        mode='lines',
        line={'color': colors[0], 'dash': 'solid', 'width': width},
        name=label + '_x')
    y = go.Scatter3d(
        x=[pos[0], xyz[1, 0]],
        y=[pos[1], xyz[1, 1]],
        z=[pos[2], xyz[1, 2]],
        mode='lines',
        line={'color': colors[1], 'dash': 'solid', 'width': width},
        name=label + '_y')
    z = go.Scatter3d(
        x=[pos[0], xyz[2, 0]],
        y=[pos[1], xyz[2, 1]],
        z=[pos[2], xyz[2, 2]],
        mode='lines',
        line={'color': colors[2], 'dash': 'solid', 'width': width},
        name=label + '_z')

    return x, y, z


def draw_vector(vec, pos=None, quat=None, scale=1, width=1, color=None, label=None, opacity=None):
    """
    Draw a 3D vector.
    :param vec: vector
    :param pos: origin of the vector
    :param quat: quaternion to rotate the vector by (scalar first)
    :param scale: scaling factor for the vector
    :param width: width of the plotted line
    :param color: color of the plotted vector
    :param label: label shown on the plot
    :param opacity: opacity of the vector
    :return: Two graph objects (Scatter3D, Cone)
    """
    # TODO: set min and max for vector length
    assert vec.shape == (3,)

    # Rotate the vector:
    if quat is not None:
        assert quat.shape == (4,), f'Unexpected quaternion shape: {quat.shape}'
        rotation = scipyRot.from_quat(np.append(quat[1:], quat[0]))  # scipy uses scalar-last quaternions
        vec = rotation.apply(vec)  # Rotated w to match the body frame

    # Scale and shift the vector:
    if pos is not None:
        assert pos.shape == (3,)
    else:
        pos = np.array([0, 0, 0])
    vec = vec * scale + pos

    if color is None:
        color = 'black'

    if opacity is None:
        opacity = 1

    line = go.Scatter3d(
        x=[pos[0], vec[0]],
        y=[pos[1], vec[1]],
        z=[pos[2], vec[2]],
        mode='lines',
        line={'color': color, 'dash': 'solid', 'width': width},
        opacity=opacity,
        name=label,
    )

    tip = go.Cone(
        x=[vec[0]],
        y=[vec[1]],
        z=[vec[2]],
        u=[0.3 * (vec[0] - pos[0])],
        v=[0.3 * (vec[1] - pos[1])],
        w=[0.3 * (vec[2] - pos[2])],
        colorscale=[[0, color], [1, color]],
        anchor='tip',
        opacity=opacity,
        hoverinfo='none',
        showscale=False,
    )

    return line, tip


def get_args():
    """
    Parses the arguments from the command line.
    :return: Namespace containing the arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--path',
        dest='path',
        type=str,
        default='',
        help='Provide the path to a pickle file containing trajectory data.'
    )
    parser.add_argument(
        '--save',
        dest='save',
        type=bool,
        nargs='?',
        const=True,     # this is the value that it takes if we call the argument
        default=False,  # this is the value that it takes by default
        help='Use this flag to save the results.'
    )
    parser.add_argument(
        '--show',
        dest='show',
        type=bool,
        nargs='?',
        const=True,  # this is the value that it takes if we call the argument
        default=False,  # this is the value that it takes by default
        help='Use this flag to display the plot.'
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    start = time.perf_counter()
    arguments = get_args()
    arguments.path = os.path.join('data', 'rdv_data1.pickle')
    arguments.show = True
    trajectory_data = load_data(arguments)
    plot_animation(arguments, trajectory_data)

    if not arguments.show:
        print('Use --show to display the plot(s).')
    if not arguments.save:
        print('Use --save to save the plot(s).')

    print(f'Finished on {time.ctime()}. ({round(time.perf_counter()-start, 2)} seconds)')
