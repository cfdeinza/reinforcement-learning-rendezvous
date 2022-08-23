"""
Script to plot charts of trajectory data.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.general import load_data
from utils.animations import plot_2dcomponents


def plot2d(args):
    data = load_data(args.path)
    rc = data['rc']  # chaser position [m]
    vc = data['vc']
    qc = data['qc']  # chaser attitude
    wc = data['wc']
    qt = data['qt']  # target attitude
    wt = data['wt']
    actions = data['a']
    rewards = data['rew'][0]
    t = data['t'][0]
    assert rc.shape[0] == 3
    # print(actions)

    # Splicing:
    start_index = 0
    end_index = -1
    if end_index != -1:
        rc = rc[:, 0:end_index]
        vc = vc[:, 0:end_index]
        qc = qc[:, 0:end_index]
        wc = wc[:, 0:end_index]
        qt = qt[:, 0:end_index]
        wt = wt[:, 0:end_index]
        actions = actions[:, 0:end_index]
        rewards = rewards[0:end_index]
        t = t[0:end_index]
    if start_index != 0:
        rc = rc[:, start_index:]
        vc = vc[:, start_index:]
        qc = qc[:, start_index:]
        wc = wc[:, start_index:]
        qt = qt[:, :, start_index:]
        wt = wt[:, start_index:]
        actions = actions[:, start_index:]
        rewards = rewards[start_index:]
        t = t[start_index:]

    # Create a matplotlib figure with two subplots:
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    plot_2dcomponents(ax[0, 0], t, rc[0], rc[1], rc[2], labels=['x', 'y', 'z'],
                      xlabel='Time (s)', ylabel='Distance (m)', title='Position')
    # Plot the overall distance from the target:
    dist = np.linalg.norm(rc, axis=0)
    ax[0, 0].plot(t, dist, 'k--', label='Overall'), ax[0, 0].legend()

    # Plot the velocity of the chaser over time:
    plot_2dcomponents(ax[0, 1], t, vc[0], vc[1], vc[2], labels=[r'$v_x$', r'$v_y$', r'$v_z$'],
                      xlabel='Time (s)', ylabel='Velocity (m/s)', title='Velocity')
    # Plot the overall speed of the chaser:
    speed = np.linalg.norm(vc, axis=0)
    ax[0, 1].plot(t, speed, 'k--', label='Overall'), ax[0, 1].legend()

    action_x, action_y, action_z = actions[0], actions[1], actions[2]
    sum_of_actions = np.abs(action_x) + np.abs(action_y) + np.abs(action_z)
    plot_2dcomponents(ax[0, 2], t, action_x, action_y, action_z,
                      labels=['x', 'y', 'z'], xlabel='Time (s)', ylabel='Delta V (m)',
                      title='Actions. Total ' + r'$\Delta V = $' + str(round(np.nansum(sum_of_actions), 2)))
    # Plot the overall control effort:
    ax[0, 2].plot(t, sum_of_actions, 'k--', label='Overall'), ax[0, 2].legend()

    # Plot the reward obtained by the chaser:
    ax[1, 0].plot(t, rewards)
    ax[1, 0].set_title(f'Rewards. Total = {round(np.nansum(rewards), 2)}')
    ax[1, 0].set_xlim([t[0], t[-1]])
    ax[1, 0].set_ylim([min(np.nanmin(rewards), 0), np.nanmax(rewards) + 1])
    ax[1, 0].grid()

    # Plot the rotational velocity of the chaser over time:
    plot_2dcomponents(ax[1, 1], t, wc[0], wc[1], wc[2], labels=[r'$\omega_x$', r'$\omega_y$', r'$\omega_z$'],
                      xlabel='Time (s)', ylabel='Rotation rate (rad/s)', title='Rotation rate')
    # Plot the overall rotation rate of the chaser:
    wc_norm = np.linalg.norm(wc, axis=0)
    ax[1, 1].plot(t, wc_norm, 'k--', label='Overall'), ax[1, 1].legend()

    action_u, action_v, action_w = actions[3], actions[4], actions[5]
    sum_of_attitude_actions = np.abs(action_u) + np.abs(action_v) + np.abs(action_w)
    plot_2dcomponents(ax[1, 2], t, action_u, action_v, action_w,
                      labels=['x', 'y', 'z'], xlabel='Time (s)', ylabel='Delta V (rad/s)',
                      title='Actions. Total ' + r'$\Delta V = $' + str(round(np.nansum(sum_of_attitude_actions), 2)))
    # Plot the overall control effort:
    ax[1, 2].plot(t, sum_of_attitude_actions, 'k--', label='Overall'), ax[1, 2].legend()

    plt.tight_layout()

    plt.show()
    plt.close()
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
    # parser.add_argument('--save', dest='save', type=bool, nargs='?',
    #                     const=True,  # this is the value that it takes if we call the argument
    #                     default=False,  # this is the value that it takes by default
    #                     help='Use this flag to save images of the animation.'
    #                     )
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    arguments = get_args()
    # arguments.path = os.path.join('data', 'rdv_data9.pickle')
    # arguments.save = True

    # Check that the path argument exists and is not a directory:
    if os.path.isdir(arguments.path) or not os.path.exists(arguments.path):
        print(f'Incorrect --path argument "{arguments.path}"')
        print('Path must be an existing file.\nExiting')
        sys.exit()

    # Check if the images will be saved or not:
    # if arguments.save:
    #     print('Images of the animation will be saved in the "Download" folder.')
    # else:
    #     print('Images will not be saved. Call the --save argument to save them.')

    plot2d(arguments)

    print('Finished')
