"""
This script generates trajectories using an existing model with varying initial conditions,
and then records the results on Weights & Biases.
"""

import os
import wandb
import numpy as np
from arguments import get_monte_carlo_args
from functools import partial
from utils.general import load_model
from utils.environment_utils import make_env


def main(args):

    # Load an existing model:
    saved_model = load_model(path=args.model, env=None)

    # Create an environment:
    reward_kwargs = None  # Does not affect performance during evaluation
    config = {}
    stochastic = True
    eval_env = make_env(reward_kwargs=reward_kwargs, quiet=False, config=config, stochastic=stochastic)
    n_evals = 100

    # Set-up the sweep:
    wandb.login(key="e9d6f3f54d82d87f667aa6b5681dd5810d8a8663")
    sweep_configuration = configure_sweep(n_evals)
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="evaluate")

    # Run the sweep:
    wandb.agent(sweep_id, function=partial(evaluate, saved_model, eval_env))

    return


def configure_sweep(n_evals):
    """
    Create the dictionary used to configure the sweep.\n
    :param n_evals: the number of evaluations to perform
    :return: configuration dictionary
    """

    sweep_config = {
        "name": "eval_sweep",   # name of the sweep (not the project)
        "metric": {             # metric to optimize, has to be logged with `wandb.log()`
            "name": "successes",
            "goal": "maximize",
        },
        "method": "grid",       # search method ("grid", "random", or "bayes")
        "parameters": {         # parameters to sweep through
            "eval": {
                "values": [i + 1 for i in range(n_evals)]
            },
            # "seed": {
            #     "values": [0],
            # },
        },
    }
    return sweep_config


def evaluate(model, env):

    with wandb.init() as _:
        print(f"Eval: {wandb.config.get('eval')}")
        # Create empty arrays to store results:
        expected_timesteps = int(env.t_max / env.dt) + 1        # size of the empty arrays
        errors = np.full((4, expected_timesteps), np.nan)       # array where errors will be saved
        t = np.full((1, expected_timesteps), np.nan)            # array where times will be saved
        in_koz = np.full((1, expected_timesteps), np.nan)       # array to indicate when the chaser is in the KOZ
        num_collisions = 0                                      # counts the number of collisions
        total_reward = 0

        # Reset environment:
        obs = env.reset()
        lstm_states = None
        ep_start = np.ones(shape=(1,), dtype=bool)
        done = False
        errors[:, 0] = env.get_errors()
        t[0, 0] = env.t
        in_koz[0, 0] = int(env.check_collision())
        num_collisions += int(env.check_collision())
        min_dist_from_koz = env.dist_from_koz()

        # Record the initial state:
        rc0 = env.rc
        vc0 = env.vc
        qc0 = env.qc
        wc0 = env.wc
        qt0 = env.qt
        wt0 = env.wt
        # Also record the rotation rates in the LVLH frame:
        wc0_lvlh = env.chaser2lvlh(vec=wc0)
        wt0_lvlh = env.target2lvlh(vec=wt0)

        k = 1
        while not done:
            # Select action:
            action, lstm_states = model.predict(
                observation=obs,  # input to the policy network
                state=lstm_states,  # last hidden state (used for recurrent policies)
                episode_start=ep_start,  # the last mask? (used for recurrent policies)
                deterministic=True  # whether or not to return deterministic actions (default is False)
            )

            # Step forward in time:
            obs, reward, done, info = env.step(action)
            ep_start[0] = done

            # Update data:
            errors[:, k] = env.get_errors()
            t[0, k] = env.t
            collision = env.check_collision()
            in_koz[0, k] = int(collision)
            num_collisions += int(collision)
            dist_from_koz = env.dist_from_koz()
            if dist_from_koz < min_dist_from_koz:
                min_dist_from_koz = dist_from_koz
            total_reward += reward
            k += 1

        # Remove nans if the episode was terminated before t_max:
        if env.t < env.t_max:
            t = np.array([t[~np.isnan(t)]])
            errors = errors[:, 0:t.size]
            in_koz = in_koz[:, 0:t.size]

        # Compute terminal errors:
        pos_error = errors[0]
        vel_error = errors[1]
        max_rd_error = env.max_rd_error
        max_vd_error = env.max_vd_error
        terminal_achieved = np.logical_and(pos_error < max_rd_error, vel_error < max_vd_error)
        if np.any(terminal_achieved):
            i_terminal = np.argmax(terminal_achieved)  # First index where pos error & vel error < constraint
            terminal_pos_error = errors[0, i_terminal:-1].mean()
            terminal_vel_error = errors[1, i_terminal:-1].mean()
            terminal_att_error = np.degrees(errors[2, i_terminal:-1].mean())
            terminal_rot_error = np.degrees(errors[3, i_terminal:-1].mean())
        else:
            # If terminal pos & vel conditions were not achieved during this run, set errors to nan
            # TODO: calculate terminal error some other way, so we have a value at least.
            terminal_pos_error = None
            terminal_vel_error = None
            terminal_att_error = None
            terminal_rot_error = None

        # Log the episode initial conditions and results:
        log_dict = dict(
            rc0=np.linalg.norm(rc0),                # magnitude of initial chaser position [m]
            rc0x=rc0[0],                            # x-component of the initial chaser position [m]
            rc0y=rc0[1],                            # y-component of the initial chaser position [m]
            rc0z=rc0[2],                            # z-component of the initial chaser position [m]
            vc0=np.linalg.norm(vc0),                # magnitude of initial chaser velocity [m/s]
            vc0x=vc0[0],                            # x-component of the initial chaser velocity [m/s]
            vc0y=vc0[1],                            # y-component of the initial chaser velocity [m/s]
            vc0z=vc0[2],                            # z-component of the initial chaser velocity [m/s]
            qc0=np.degrees(2*np.arccos(qc0[0])),    # magnitude of initial chaser attitude [deg]
            wc0=np.degrees(np.linalg.norm(wc0)),    # magnitude of initial chaser rot rate [deg/s]
            wc0x=np.degrees(wc0[0]),                # x-component of the initial chaser rot rate [deg/s] (in CB frame)
            wc0y=np.degrees(wc0[1]),                # y-component of the initial chaser rot rate [deg/s] (in CB frame)
            wc0z=np.degrees(wc0[2]),                # z-component of the initial chaser rot rate [deg/s] (in CB frame)
            wc0x_lvlh=np.degrees(wc0_lvlh[0]),      # x-component of the initial chaser rot rate [deg/s] (in LVLH)
            wc0y_lvlh=np.degrees(wc0_lvlh[1]),      # y-component of the initial chaser rot rate [deg/s] (in LVLH)
            wc0z_lvlh=np.degrees(wc0_lvlh[2]),      # z-component of the initial chaser rot rate [deg/s] (in LVLH)
            qt0=np.degrees(2 * np.arccos(qt0[0])),  # magnitude of initial target attitude [deg]
            wt0=np.degrees(np.linalg.norm(wt0)),    # magnitude of initial target rot rate [deg/s]
            wt0x=np.degrees(wt0[0]),                # x-component of the initial chaser rot rate [deg/s] (in TB frame)
            wt0y=np.degrees(wt0[1]),                # y-component of the initial chaser rot rate [deg/s] (in TB frame)
            wt0z=np.degrees(wt0[2]),                # z-component of the initial chaser rot rate [deg/s] (in TB frame)
            wt0x_lvlh=np.degrees(wt0_lvlh[0]),      # x-component of the initial chaser rot rate [deg/s] (in LVLH)
            wt0y_lvlh=np.degrees(wt0_lvlh[1]),      # y-component of the initial chaser rot rate [deg/s] (in LVLH)
            wt0z_lvlh=np.degrees(wt0_lvlh[2]),      # z-component of the initial chaser rot rate [deg/s] (in LVLH)
            ep_len=t[0, -1],                        # length of the episode [s]
            num_collisions=num_collisions,          # number of collisions during episode
            collided=int(num_collisions > 0),       # 1 if the chaser entered the KOZ, 0 otherwise
            total_reward=total_reward,              # total reward achieved during episode
            total_delta_v=env.total_delta_v,        # total Delta V used during the trajectory
            num_successes=env.success,              # number of successes during episode
            succeeded=int(env.success > 0),         # 1 if the chaser achieved at least 1 success, 0 otherwise
            min_dist_from_koz=min_dist_from_koz,    # the min distance of the chaser to the KOZ (negative means inside)
            pos_error=terminal_pos_error,           # average terminal position error [m]
            vel_error=terminal_vel_error,           # average terminal velocity error [m/s]
            att_error=terminal_att_error,           # average terminal attitude error [deg]
            rot_error=terminal_rot_error,           # average terminal rot rate error [deg/s]
        )

        wandb.log(log_dict)

    return


if __name__ == "__main__":

    arguments = get_monte_carlo_args()
    arguments.model = r"C:\Users\charl\Downloads\rnn_model_decent5.zip"
    main(arguments)
