"""
Plot a vpython animation of a given rendezvous trajectory. The user needs to specify the path to the file containing
the trajectory data.

Written by C. F. De Inza Niemeijer.
"""

import os
import sys
import argparse
import numpy as np
from functools import partial
from vpython import *
from utils.general import load_data
from utils.quaternions import quat_error, quat2rot
from utils.vpython_utils import create_scene, numpy2vec, create_chaser, create_target, create_koz, create_frame


class RunState:
    """
    Object to keep track of the state of the animation.\n
    """
    def __init__(self, running: bool):
        self.state = running    # whether or not the animation is running
        self.reset = False      # whether or not the animation has to be reset
        return


def make_animation(args):

    data = load_data(args.path)
    rc = data['rc']  # chaser position [m]
    qc = data['qc']  # chaser attitude
    qt = data['qt']  # target attitude
    assert rc.shape[0] == 3

    # Time data:
    if 't' in data:
        t = data['t'][0]  # make time one-dimensional
    else:
        if 't_max' in data:
            t = np.linspace(0, data['t_max'], num=len(rc[0]))
        else:
            print('Time data not found.')
            t = np.linspace(0, 1, num=len(rc[0]))

    # Keep-out zone radius:
    if 'koz_radius' in data:
        koz_radius = data['koz_radius']
    else:
        koz_radius = 5
        print(f'Radius of keep-out zone is undefined. Using default ({koz_radius} m)')

    print('Computing rotations...', end=' ')

    c_axis0, c_angle0 = quat2rot(qc[:, 0])  # converts quaternion to axis of rotation & degree of rotation.
    qc_axes = [numpy2vec(c_axis0)]          # axis around which the rotation occurs
    qc_angles = [c_angle0]                  # degree of the rotation [rad]
    t_axis0, t_angle0 = quat2rot(qt[:, 0])
    qt_axes = [numpy2vec(t_axis0)]
    qt_angles = [t_angle0]

    for i in range(1, len(t)):
        qc_error = quat_error(qc[:, i-1], qc[:, i])     # compute the error between the current and previous quaternion
        axis, angle = quat2rot(qc_error)                # convert quaterion to axis & angle
        qc_axes.append(numpy2vec(axis))                 # append result to list of axes
        qc_angles.append(angle)                         # append result to list of angles
        qt_error = quat_error(qt[:, i-1], qt[:, i])     # same method for target
        axis, angle = quat2rot(qt_error)
        qt_axes.append(numpy2vec(axis))
        qt_angles.append(angle)
    print('Done')

    # Create a scene and the objects:
    myscene = create_scene(title=f'Rendezvous: {os.path.split(args.path)[1]}\n', caption='')
    create_frame(np.array([0, 0, 0]))
    chaser = create_chaser(rc0=rc[:, 0])
    target = create_target(koz_radius)
    create_koz(koz_radius)

    # Rotate bodies to their initial orientation:
    chaser.rotate(qc_angles[0], axis=qc_axes[0])
    target.rotate(qt_angles[0], axis=qt_axes[0])

    # myscene.camera.follow(chaser)  # set the camera to follow the chaser

    # Graph:
    graph(title='Test', xmin=0, xmax=t[-1] * 1.05, ymin=rc.min(), ymax=rc.max(),
          align='left', ytitle='Position [m]', xtitle='Time [s]')
    f1 = gcurve(color=color.red, label='x')
    f2 = gcurve(color=color.green, label='y')
    f3 = gcurve(color=color.blue, label='z')
    f1.plot([0, chaser.pos.x])
    f2.plot([0, chaser.pos.y])
    f3.plot([0, chaser.pos.z])

    # Play-pause:
    running = RunState(False)

    b1 = button(text="Pause" if running.state else " Play ",
                pos=myscene.title_anchor, bind=partial(play_pause, run=running))

    # Main loop:
    k = 1
    # fps = 1 / data['dt']  # 30
    fps = 30
    while True:
        if running.state:
            rate(fps)
            if args.save:
                myscene.capture(f'img{k-1}')  # images will be saved in the 'Download' folder.
                if k < 3:
                    sleep(3)  # time delay to accept multiple downloads on Chrome pop-up
                else:
                    sleep(0.5)  # time delay to allow the images to be saved in correct order

            # Update chaser position and attitude:
            chaser.pos = vector(rc[0, k], rc[1, k], rc[2, k])
            chaser.rotate(angle=qc_angles[k], axis=qc_axes[k], origin=chaser.pos)
            target.rotate(angle=qt_angles[k], axis=qt_axes[k], origin=target.pos)
            # Plot position on the graph:
            f1.plot([t[k], chaser.pos.x])
            f2.plot([t[k], chaser.pos.y])
            f3.plot([t[k], chaser.pos.z])
            k += 1
            # Stop at last time-step:
            if k >= len(t):
                running.state = False
                b1.text = "Reset"
        else:
            # Reset:
            if running.reset:
                running.reset = False
                # clear graph:
                f1.delete()
                f2.delete()
                f3.delete()
                # Reset position and attitude of the bodies:
                chaser.axis = vector(1, 0, 0)  # default 'axis' direction
                chaser.up = vector(0, 1, 0)  # default 'up' direction
                chaser.pos = numpy2vec(rc[:, 0])
                chaser.rotate(angle=qc_angles[0], axis=qc_axes[0], origin=chaser.pos)
                target.axis = vector(1, 0, 0)
                target.up = vector(0, 1, 0)
                target.rotate(angle=qt_angles[0], axis=qt_axes[0], origin=target.pos)
                k = 1
    pass


def play_pause(b: button, run: RunState):
    """
    Callback function to play and pause the animation when pressing a button.\n
    :param b: vpython button object
    :param run: custom object with a 'state' attribute that shows if the animation is currently running.
    :return: None
    """

    if b.text == "Reset":
        run.reset = True            # reset the animation
    else:
        run.state = not run.state   # pause or un-pause the animation

    if run.state:
        b.text = "Pause"
    else:
        b.text = " Play "

    return


def get_args():
    """
    Get the arguments when running the script from the command line.
    The `path` argument specifies the pickle file containing the trajectory data. It is always required.
    :return: Namespace containing the arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', type=str, default='',
                        help='Path to the pickle file containing the trajectory data.')
    parser.add_argument('--save', dest='save', type=bool, nargs='?',
                        const=True,  # this is the value that it takes if we call the argument
                        default=False,  # this is the value that it takes by default
                        help='Use this flag to save images of the animation.'
                        )
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    arguments = get_args()
    # arguments.path = os.path.join('data', 'rdv_data00.pickle')
    # arguments.save = True

    # Check that the path argument exists and is not a directory:
    if os.path.isdir(arguments.path) or not os.path.exists(arguments.path):
        print(f'Incorrect --path argument "{arguments.path}"')
        print('Path must be an existing file.\nExiting')
        sys.exit()

    # Check if the images will be saved or not:
    if arguments.save:
        print('Images of the animation will be saved in the "Download" folder.')
    else:
        print('Images will not be saved. Call the --save argument to save them.')

    make_animation(arguments)

    print('Finished')
