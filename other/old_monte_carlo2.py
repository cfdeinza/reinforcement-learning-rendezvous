"""
This script evaluated trajectories using an existing model with a given set of initial conditions.
The initial conditions are taken from a csv file.
The results are saved in a csv file.
"""

import os
import numpy as np
import pandas as pd
from arguments import get_monte_carlo_args
from utils.general import load_model
from utils.environment_utils import make_env


def main(args):

    if not args.save:
        print("Results will NOT be saved. Use the '--save' argument to save the results as a csv file.")

    # Load an existing model:
    saved_model = load_model(path=args.model, env=None)

    # Create an environment to evaluate the model:
    reward_kwargs = None  # Does not affect performance during evaluation
    config = dict(dt=1, t_max=60)
    eval_env = make_env(reward_kwargs=reward_kwargs, quiet=False, config=config, stochastic=False)

    # Get the initial conditions:
    try:
        print(f"Loading csv file containing initial conditions...", end=" ")
        df0 = pd.read_csv(args.input_file)
        print("Done")
    except FileNotFoundError:
        df0 = None
        print(f"\nCould not find the requested file named '{args.input_file}'\nExiting")
        exit()

    # Create a dictionary to store the results:
    results = dict(
        ep_len=[],
        num_collisions=[],
        collided=[],
        total_reward=[],
        total_delta_v=[],
        num_successes=[],
        succeeded=[],
        min_dist_from_koz=[],
        pos_error=[],
        vel_error=[],
        att_error=[],
        rot_error=[],
    )

    # Run the evaluations:
    for i in range(len(df0)):
        print(f"Eval: {i}")
        # Get initial states from dataframe:
        rc = df0.iloc[i][['rcx', 'rcy', 'rcz']].to_numpy()
        vc = df0.iloc[i][['vcx', 'vcy', 'vcz']].to_numpy()
        qc = df0.iloc[i][['qcw', 'qcx', 'qcy', 'qcz']].to_numpy()
        wc = df0.iloc[i][['wcx', 'wcy', 'wcz']].to_numpy()
        qt = df0.iloc[i][['qtw', 'qtx', 'qty', 'qtz']].to_numpy()
        wt = df0.iloc[i][['wtx', 'wty', 'wtz']].to_numpy()
        # Normalize quaternions:
        qc = qc / np.linalg.norm(qc)
        qt = qt / np.linalg.norm(qt)
        initial_state = dict(rc=rc, vc=vc, qc=qc, wc=wc, qt=qt, wt=wt)
        # Run the trajectory:
        out = evaluate(saved_model, eval_env, initial_state)
        # Store trajectory outputs in dictionary:
        for key in results.keys():
            results[key].append(out[key])

    succeeded_eps = np.array(results["succeeded"]).sum()
    collided_eps = np.array(results["collided"]).sum()
    print(f"Success %: {succeeded_eps/len(df0)*100}")
    print(f"Collision%: {collided_eps/len(df0)*100}")

    # Save results in csv file:
    if args.save:
        df_results = pd.DataFrame(results)
        num = 0
        output_file_name = f"monte_carlo_results{str(num).zfill(2)}.csv"
        while os.path.exists(output_file_name):
            num += 1
            output_file_name = f"monte_carlo_results{str(num).zfill(2)}.csv"
        print(f"Saving results to '{output_file_name}'...", end=" ")
        df_results.to_csv(output_file_name)
        print("Done")
    return


def evaluate(model, env, initial_state):

    # Create empty arrays to store results:
    expected_timesteps = int(env.t_max / env.dt) + 1        # size of the empty arrays
    errors = np.full((4, expected_timesteps), np.nan)       # array where errors will be saved
    t = np.full((1, expected_timesteps), np.nan)            # array where times will be saved
    in_koz = np.full((1, expected_timesteps), np.nan)       # array to indicate when the chaser is in the KOZ
    num_collisions = 0                                      # counts the number of collisions
    num_successes = 0
    total_reward = 0

    # Reset the environment with the given initial conditions:
    env.reset()
    env.rc = initial_state['rc']
    env.vc = initial_state['vc']
    env.qc = initial_state['qc']
    env.wc = initial_state['wc']
    env.qt = initial_state['qt']
    env.wt = initial_state['wt']
    obs = env.get_observation()
    lstm_states = None
    ep_start = np.ones(shape=(1,), dtype=bool)
    done = False
    errors[:, 0] = env.get_errors()
    t[0, 0] = env.t
    in_koz[0, 0] = int(env.check_collision())
    num_collisions += int(env.check_collision())
    if not env.collided:
        num_successes += int(env.check_success())
    min_dist_from_koz = env.dist_from_koz()

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
        if not env.collided:
            num_successes += int(env.check_success())
        dist_from_koz = env.dist_from_koz()
        if dist_from_koz < min_dist_from_koz:
            min_dist_from_koz = dist_from_koz
        total_reward += reward
        k += 1

    # Remove nans if the episode was terminated before t_max:
    if env.t < env.t_max:
        t = np.array([t[~np.isnan(t)]])
        errors = errors[:, 0:t.size]
        # in_koz = in_koz[:, 0:t.size]

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
        terminal_pos_error = None
        terminal_vel_error = None
        terminal_att_error = None
        terminal_rot_error = None

    # Log the results:
    out = dict(
        ep_len=t[0, -1],                        # length of the episode [s]
        num_collisions=num_collisions,          # number of collisions during episode
        collided=int(num_collisions > 0),       # 1 if the chaser entered the KOZ, 0 otherwise
        total_reward=total_reward,              # total reward achieved during episode
        total_delta_v=env.total_delta_v,        # total Delta V used during the trajectory
        num_successes=num_successes,            # number of successes during episode
        succeeded=int(num_successes > 0),       # 1 if the chaser achieved at least 1 success, 0 otherwise
        min_dist_from_koz=min_dist_from_koz,    # the min distance of the chaser to the KOZ (negative means inside)
        pos_error=terminal_pos_error,           # average terminal position error [m]
        vel_error=terminal_vel_error,           # average terminal velocity error [m/s]
        att_error=terminal_att_error,           # average terminal attitude error [deg]
        rot_error=terminal_rot_error,           # average terminal rot rate error [deg/s]
    )

    return out


if __name__ == "__main__":

    arguments = get_monte_carlo_args()
    # arguments.model = r"C:\Users\charl\Downloads\rnn_model_decent5.zip"
    # arguments.model = r"C:\Users\charl\Downloads\mlp_model_final_03.zip"
    # arguments.model = r"C:\Users\charl\Downloads\rnn_model_final_06.zip"
    # arguments.model = r"C:\Users\charl\Downloads\rnn_model_box_04_08.zip"
    # arguments.input_file = os.path.join("initial1000.csv")
    # arguments.input_file = os.path.join("initial50.csv")
    # arguments.save = False
    main(arguments)
