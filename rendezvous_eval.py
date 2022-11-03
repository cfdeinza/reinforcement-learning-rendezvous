"""
This script generates a rendevous trajectory. The user needs to specify a controller by using the argument --model.

Written by C. F. De Inza Niemeijer.
"""

from rendezvous_env import RendezvousEnv
from utils.general import load_model  # , load_env
import argparse
import numpy as np
import pickle
import os

"""
Evaluate a model over a single episode.
"""


def get_action(model, obs):
    action, state = model.predict(
            observation=obs,        # input to the policy network
            state=None,             # last hidden state (used for recurrent policies)
            episode_start=None,     # the last mask? (used for recurrent policies)
            deterministic=True      # whether or not to return deterministic actions (default is False)
        )
    """
    The outputs of `model.predict()` are the action and the next hidden state.
    for more information on the inputs and outpus of the predict function, 
    see https://github.com/DLR-RM/stable-baselines3/blob/ef10189d80dbb2efb3b5391cba4eb3c2ab5c7aae/
    stable_baselines3/common/policies.py#L307
    """
    return action


def evaluate(model, env, args):
    """
    Evaluate one episode.\n
    :param model: model that picks the actions
    :param env: environment
    :param args: command-line arguments
    :return: dictionary with episode data
    """

    env.reset()

    print(f"Initial state:")  # {np.hstack((env.rc, env.vc, env.qc, env.wc, env.qt))}")
    print(f"rc: {env.rc} m, magnitude: {round(np.linalg.norm(env.rc), 3)} m")
    print(f"vc: {env.vc} m/s, magnitude: {round(np.linalg.norm(env.vc), 3)} m/s")
    print(f"qc: {env.qc}")
    print(f"wc: {np.degrees(env.wc)} deg/s, magnitude: {round(np.linalg.norm(np.degrees(env.wc)), 3)} deg/s")
    print(f"qt: {env.qt}")
    print(f"wt: {np.degrees(env.wt)} deg/s, magnitude: {round(np.linalg.norm(np.degrees(env.wt)), 3)} deg/s")
    # print(f'Starting rotation rate: {env.state[4:]}')

    obs = env.get_observation()
    total_reward = 0
    lstm_states = None
    ep_start = np.ones(shape=(1,), dtype=bool)
    done = False
    # info = {}

    # Create empty arrays to save values:
    expected_timesteps = int(env.t_max / env.dt) + 1    # size of the empty arrays
    # State:
    rc = np.full((3, expected_timesteps), np.nan)       # chaser position
    vc = np.full((3, expected_timesteps), np.nan)       # chaser velocity
    qc = np.full((4, expected_timesteps), np.nan)       # chaser attitude
    wc = np.full((3, expected_timesteps), np.nan)       # chaser rotation rate
    qt = np.full((4, expected_timesteps), np.nan)       # target attitude
    wt = np.full((3, expected_timesteps), np.nan)       # target rotation rate
    # Action and reward:
    a = np.full((6, expected_timesteps), np.nan)        # actions (last obs has no action)
    rew = np.full((1, expected_timesteps), np.nan)      # rewards (first state has no reward)
    # Errors:
    # array of errors expressed in magnitudes (pos [m], vel [m/s], att [rad], rotation rate [rad/s])
    errors = np.full((4, expected_timesteps), np.nan)
    # Time:
    t = np.full((1, expected_timesteps), np.nan)        # times

    # Save the initial values:
    rc[:, 0] = env.rc
    vc[:, 0] = env.vc
    qc[:, 0] = env.qc
    wc[:, 0] = env.wc
    qt[:, 0] = env.qt
    wt[:, 0] = env.wt
    errors[:, 0] = env.get_errors()
    t[0, 0] = env.t

    # torque = 3e-2
    # force = 0.25
    k = 1
    while not done:

        # Select action:
        # action = get_action(model, obs)
        action, lstm_states = model.predict(
            observation=obs,            # input to the policy network
            state=lstm_states,          # last hidden state (used for recurrent policies)
            episode_start=ep_start,     # the last mask? (used for recurrent policies)
            deterministic=True          # whether or not to return deterministic actions (default is False)
        )
        """
        The outputs of `model.predict()` are the action and the next hidden state.
        for more information on the inputs and outpus of the predict function, 
        see https://github.com/DLR-RM/stable-baselines3/blob/ef10189d80dbb2efb3b5391cba4eb3c2ab5c7aae/
        stable_baselines3/common/policies.py#L307
        """

        # Step forward in time:
        obs, reward, done, info = env.step(action)
        ep_start[0] = done

        # Save current values:
        total_reward += reward
        rc[:, k] = env.rc
        vc[:, k] = env.vc
        qc[:, k] = env.qc
        wc[:, k] = env.wc
        qt[:, k] = env.qt
        wt[:, k] = env.wt
        # delta_v, delta_w = env.process_action(action)
        # a[:, k-1] = np.append(delta_v, delta_w)  # processed action
        processed_action = env.process_action(action)
        a[:, k-1] = processed_action
        rew[0, k] = reward
        errors[:, k] = env.get_errors()
        t[0, k] = env.t
        print(f'Step: {env.t}', end='\r')
        k += 1

    # trajectory = info['trajectory']
    # actions = info['actions']

    # Remove nans if the episode was terminated before t_max:
    if env.t < env.t_max:
        t = np.array([t[~np.isnan(t)]])
        rc = rc[:, 0:t.size]
        vc = vc[:, 0:t.size]
        qc = qc[:, 0:t.size]
        wc = wc[:, 0:t.size]
        qt = qt[:, 0:t.size]
        wt = wt[:, 0:t.size]
        a = a[:, 0:t.size]
        rew = rew[:, 0:t.size]
        errors = errors[:, 0:t.size]

    print(f'\nTotal reward: {round(total_reward, 2)}')

    # Create dictionary of env attributes:
    data = vars(env)

    # Add new pairs to dictionary:
    new_pairs = [
        ('rc', rc), ('vc', vc), ('qc', qc), ('wc', wc), ('qt', qt), ('wt', wt),
        ('a', a), ('rew', rew), ('errors', errors), ('t', t),
        ('process_action', None),  # pickle cannot handle lambda functions
    ]
    for key, val in new_pairs:
        data[key] = val

    # Remove unwanted attributes:
    unwanted_attributes = ['viewer']
    for key in unwanted_attributes:
        data.pop(key)  # Will raise a KeyError if key does not exist
        # data.pop(key, None)  # Will not raise an error if key does not exist

    if args.save:
        # Name of the output file:
        file_num = 0
        name = os.path.join('data', 'rdv_data' + str(file_num).zfill(2) + '.pickle')
        while os.path.exists(name):
            file_num += 1
            name = os.path.join('data', 'rdv_data' + str(file_num).zfill(2) + '.pickle')

        # Save the data file:
        with open(name, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'Saved trajectory data: {name}')

    return data


def get_args():
    """
    Parses the arguments from the command line.
    :return: Namespace containing the arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        dest='model',
        type=str,
        default='',
        help='Provide the name of an existing model to evaluate or to continue training.'
    )
    parser.add_argument(
        '--save',
        dest='save',
        type=bool,
        nargs='?',
        default=False,  # default value
        const=True,     # value when called
        help='Use this flag to save the results.'
    )
    parser.add_argument(
        '--render',
        dest='render',
        type=bool,
        nargs='?',
        const=True,
        default=False,
        help='Use this flag to render the episode.'
    )
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    arguments = get_args()
    # arguments.model = os.path.join('models', 'model_01.zip')
    # arguments.save = True
    # arguments.render = True

    reward_kwargs = None
    environment = RendezvousEnv(reward_kwargs=reward_kwargs)
    if reward_kwargs is None:
        print("Note: reward_kwargs have not been defined. Using default values.")

    # environment.dt = 0.1  # you can set a new time interval here
    # environment = load_env(arguments)
    saved_model = load_model(path=arguments.model, env=environment)

    if not arguments.save:
        print('Trajectory data will not be saved. To save, use --save argument')

    evaluate(saved_model, environment, arguments)

    print('Finished')
