"""
Script to plot charts of trajectory data.

Written by C. F. De Inza Niemeijer.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.general import load_data
from utils.animations import plot_2dcomponents


def plot2d_response(args, data=None):
    if data is None:
        data = load_data(args.path)
    else:
        assert isinstance(data, dict), "parameter 'data' must be a dictionary"
    rc = data['rc']  # chaser position [m]
    vc = data['vc']
    # qc = data['qc']  # chaser attitude
    wc = data['wc']
    # qt = data['qt']  # target attitude
    # wt = data['wt']
    actions = data['a']
    actions[0:3] = actions[0:3] * data['max_delta_v']  # convert actions to delta_v and delta_w (in chaser body frame)
    actions[3:] = actions[3:] * data['max_delta_w']
    rewards = data['rew'][0]
    t = data['t'][0]
    t_max = data["t_max"]
    assert rc.shape[0] == 3
    # print(actions)

    # Create a matplotlib figure with 6 subplots:
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))

    # Chaser position:
    plot_2dcomponents(ax[0, 0], t, rc[0], rc[1], rc[2], labels=['x', 'y', 'z'],
                      xlabel='Time (s)', ylabel='Distance (m)', title='Position')
    dist = np.linalg.norm(rc, axis=0)
    ax[0, 0].plot(t, dist, 'k--', label='Overall'), ax[0, 0].legend()  # Overall distance
    ax[0, 0].set_xlim([t[0], t_max])

    # Chaser velocity:
    plot_2dcomponents(ax[0, 1], t, vc[0], vc[1], vc[2], labels=[r'$v_x$', r'$v_y$', r'$v_z$'],
                      xlabel='Time (s)', ylabel='Velocity (m/s)', title='Velocity')
    speed = np.linalg.norm(vc, axis=0)
    ax[0, 1].plot(t, speed, 'k--', label='Overall'), ax[0, 1].legend()  # Overall speed
    ax[0, 1].set_xlim([t[0], t_max])

    # Delta V:
    action_x, action_y, action_z = actions[0], actions[1], actions[2]
    sum_of_actions = np.abs(action_x) + np.abs(action_y) + np.abs(action_z)
    plot_2dcomponents(ax[0, 2], t, action_x, action_y, action_z, labels=['x', 'y', 'z'],
                      xlabel='Time (s)', ylabel='$\Delta V$ (m)', title='Actions. Total ' + r'$\Delta V = $' +
                                                                        str(round(np.nansum(sum_of_actions), 2)))
    ax[0, 2].plot(t, sum_of_actions, 'k--', label='Overall'), ax[0, 2].legend()  # Overall Delta V
    ax[0, 2].set_xlim([t[0], t_max])

    # Reward:
    ax[1, 0].plot(t, rewards)
    ax[1, 0].set_title(f'Rewards. Total = {round(np.nansum(rewards), 2)}')
    ax[1, 0].set_xlim([t[0], t_max])
    ax[1, 0].set_ylim([min(float(np.nanmin(rewards)), 0), np.nanmax(rewards) + 1])
    ax[1, 0].grid()

    # Chaser rotation rate:
    plot_2dcomponents(ax[1, 1], t, wc[0], wc[1], wc[2], labels=[r'$\omega_x$', r'$\omega_y$', r'$\omega_z$'],
                      xlabel='Time (s)', ylabel='Rotation rate (rad/s)', title='Rotation rate')
    wc_norm = np.linalg.norm(wc, axis=0)
    ax[1, 1].plot(t, wc_norm, 'k--', label='Overall'), ax[1, 1].legend()  # Overall rotation rate
    ax[1, 1].set_xlim([t[0], t_max])

    # Delta Omega:
    action_u, action_v, action_w = actions[3], actions[4], actions[5]
    sum_of_attitude_actions = np.abs(action_u) + np.abs(action_v) + np.abs(action_w)
    plot_2dcomponents(ax[1, 2], t, action_u, action_v, action_w, labels=['x', 'y', 'z'],
                      xlabel='Time (s)', ylabel='$\Delta \omega $ (rad/s)',
                      title='Actions. Total ' + r'$\Delta \omega = $' +
                            str(round(np.nansum(sum_of_attitude_actions), 2)))
    ax[1, 2].plot(t, sum_of_attitude_actions, 'k--', label='Overall'), ax[1, 2].legend()  # Overall Delta Omega
    ax[1, 2].set_xlim([t[0], t_max])

    plt.tight_layout()
    plt.show()
    plt.close()
    return


def plot2d_error(args, data=None):
    if data is None:
        data = load_data(args.path)
    else:
        assert isinstance(data, dict), "parameter `data` must be a dictionary"

    t = data["t"][0]
    t_max = data["t_max"]
    errors = data["errors"]
    assert errors.shape[0] == 4, f"array of errors has an unexpected shape: {errors.shape}"
    pos_error = errors[0, :]
    vel_error = errors[1, :]
    att_error = np.degrees(errors[2, :])
    rot_error = np.degrees(errors[3, :])

    # Create a matplotlib figure with 4 subplots:
    n_rows, n_cols = 2, 2
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(12, 6))
    plot_error_component(ax[0, 0], t, pos_error, data["max_rd_error"],
                         xlim=[0, t_max], ylim=[0, max(20, np.max(pos_error))],
                         xlabel="Time", ylabel="Position [m]", title="Position",
                         )
    plot_error_component(ax[0, 1], t, vel_error, data["max_vd_error"],
                         xlim=[0, t_max], ylim=[0, max(2, np.max(vel_error))],
                         xlabel="Time [s]", ylabel="Velocity [m/s]", title="Velocity",
                         )
    plot_error_component(ax[1, 0], t, att_error, np.degrees(data["max_qd_error"]),
                         xlim=[0, t_max], ylim=[0, np.degrees(data["max_attitude_error"])],
                         xlabel="Time [s]", ylabel="Attitude [deg]", title="Attitude"
                         )
    plot_error_component(ax[1, 1], t, rot_error, np.degrees(data["max_wd_error"]),
                         xlim=[0, t_max], ylim=[0, max(10, np.max(rot_error))],
                         xlabel="Time [s]", ylabel="Rotation rate [deg/s]", title="Rotation rate"
                         )
    plt.tight_layout()
    plt.show()
    plt.close()

    return


def plot_error_component(ax, x, y, constraint, xlim, ylim=None, xlabel=None, ylabel=None, title=None):
    ax.plot(x, y, label="Error")
    ax.plot(xlim, [constraint] * 2, "r--", label="Constraint")
    ax.set_xlim(xlim), ax.set_ylim(ylim)
    ax.set_xlabel(xlabel), ax.set_ylabel(ylabel)
    ax.grid(), ax.legend()
    ax.set_title(title)
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

    plot2d_response(arguments)
    plot2d_error(arguments)

    print('Finished')

"""
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
"""
