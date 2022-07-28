"""
Plot a vpython animation of a given rendezvous trajectory.
"""

import os
import sys
import argparse
import numpy as np
from functools import partial
from vpython import *
from utils.general import load_data
from utils.quaternions import quat_error, quat2rot
from utils.vpython_utils import create_scene, numpy2vec, create_chaser, create_target, create_koz  # , play_pause_button


class RunState:
    """
    Object to keep track of the state of the animation (whether it is playing or not).\n
    """
    def __init__(self, running):
        self.state = running
        return


def make_animation(args):

    data = load_data(args.path)
    t = data['t'][0]  # make time one-dimensional
    rc = data['rc']  # chaser position [m]
    qc = data['qc']  # chaser attitude
    assert rc.shape[0] == 3
    # print(t[-1])

    print('Computing rotations...', end=' ')

    rot0 = quat2rot(qc[:, 0])
    qc_axes = [numpy2vec(rot0[0])]
    qc_angles = [rot0[1]]

    for i in range(1, qc.shape[1]):
        qc_error = quat_error(qc[:, i-1], qc[:, i])
        axis, angle = quat2rot(qc_error)
        qc_axes.append(numpy2vec(axis))
        qc_angles.append(angle)
    print('Done')

    # Create a scene and the objects:
    myscene = create_scene(title='Rendezvous\n', caption='')
    chaser = create_chaser(rc0=rc[:, 0])
    target = create_target()
    koz = create_koz()
    myscene.camera.follow(chaser)  # set the camera to follow the chaser

    # Graph:
    # k_max = len(rc[0])
    gd = graph(title='Test', xmin=0, xmax=t[-1] + 5, ymin=-50, ymax=10,
               align='left', ytitle='Position [m]', xtitle='Time [s]')
    f1 = gcurve(color=color.red, label='x')
    f2 = gcurve(color=color.green, label='y')
    f3 = gcurve(color=color.blue, label='z')
    f1.plot([0, chaser.pos.x])
    f2.plot([0, chaser.pos.y])
    f3.plot([0, chaser.pos.z])

    # Play-pause:
    running = RunState(False)

    button(text="Pause" if running.state else " Play ", pos=myscene.title_anchor, bind=partial(play_pause, run=running))

    # Main loop:
    k = 1
    while k < len(t):
        if running.state:
            rate(30)
            if args.save:
                myscene.capture(f'img{k-1}')  # images will be saved in the 'Download' folder.
                if k < 3:
                    sleep(3)  # time delay to accept multiple downloads on Chrome pop-up
                else:
                    sleep(0.5)  # time delay to allow the images to be saved in correct order

            # Update chaser position and attitude:
            chaser.pos = vector(rc[0, k], rc[1, k], rc[2, k])
            chaser.rotate(angle=qc_angles[k], axis=qc_axes[k], origin=chaser.pos)
            # Plot position on the graph:
            f1.plot([t[k], chaser.pos.x])
            f2.plot([t[k], chaser.pos.y])
            f3.plot([t[k], chaser.pos.z])
            k += 1
        # if k < len(rc[0])-1:
        #     k += 1
        # else:
        #     k = 0  # TODO: reset the rotation!
    return


def play_pause(b: button, run: RunState):
    """
    Callback function to play and pause the animation when pressing a button.\n
    :param b: vpython button object
    :param run: custom object with a 'state' attribute that shows if the animation is currently running.
    :return: None
    """
    run.state = not run.state
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
    arguments.path = os.path.join('data', 'rdv_data6.pickle')
    arguments.save = False

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
