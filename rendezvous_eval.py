"""
This script generates a rendevous trajectory.
"""

from rendezvous_env import RendezvousEnv
from main import load_model  #, load_env
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
    see https://github.com/DLR-RM/stable-baselines3/blob/ef10189d80dbb2efb3b5391cba4eb3c2ab5c7aae/stable_baselines3/common/policies.py#L307
    """
    return action


def evaluate(model, env, args):

    env.reset()

    print(f'Initial state: {np.hstack((env.rc, env.vc, env.qc, env.wc))}')
    # print(f'Starting rotation rate: {env.state[4:]}')

    obs = env.get_observation()
    total_reward = 0
    done = False
    info = {}
    expected_timesteps = int(env.t_max / env.dt) + 1
    rc = np.full((3, expected_timesteps), np.nan)
    vc = np.full((3, expected_timesteps), np.nan)
    qc = np.full((4, expected_timesteps), np.nan)
    wc = np.full((3, expected_timesteps), np.nan)
    a = np.full((6, expected_timesteps), np.nan)    # actions (last obs has no action)
    rew = np.full((1, expected_timesteps), np.nan)  # rewards (first state has no reward)
    t = np.full((1, expected_timesteps), np.nan)    # times
    rc[:, 0] = env.rc
    vc[:, 0] = env.vc
    qc[:, 0] = env.qc
    wc[:, 0] = env.wc
    t[0, 0] = env.t

    torque = 3e-2
    force = 0.25
    k = 1
    while not done:
        # action = get_action(model, obs)
        if k == 10:
            action = np.array([0, force, 0, torque, torque, torque])
        # elif k == 92:
        #     action = np.array([0, 0, 0, torque, 0, -torque])
        else:
            action = np.array([0, 0, 0, 0, 0, 0])
        obs, reward, done, info = env.step(action)
        total_reward += reward
        rc[:, k] = env.rc
        vc[:, k] = env.vc
        qc[:, k] = env.qc
        wc[:, k] = env.wc
        a[:, k-1] = action
        rew[0, k] = reward
        t[0, k] = env.t
        print(f'Step: {k}', end='\r')
        k += 1

    # trajectory = info['trajectory']
    # actions = info['actions']

    print(f'\nTotal reward: {round(total_reward, 2)}')

    if args.save:
        data = {
            'rc': rc,
            'vc': vc,
            'qc': qc,
            'wc': wc,
        }
        file_num = 0
        name = os.path.join('data', 'rdv_data' + str(file_num) + '.pickle')
        while os.path.exists(name):
            file_num += 1
            name = os.path.join('data', 'rdv_data' + str(file_num) + '.pickle')

        with open(name, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'Saved trajectory data: {name}')

    return


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
        const=True,     # this is the value that it takes if we call the argument
        default=False,  # this is the value that it takes by default
        help='Use this flag to save the results.'
    )
    parser.add_argument(
        '--render',
        dest='render',
        type=bool,
        nargs='?',
        const=True,
        default=False,
        help='Use this flag to render the episodes.'
    )
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    arguments = get_args()
    # arguments.model = ''
    arguments.save = True
    # arguments.render = True

    environment = RendezvousEnv()
    # environment = load_env(arguments)
    if len(arguments.model) > 0:
        saved_model = load_model(arguments, environment)
    else:
        saved_model = None

    evaluate(saved_model, environment, arguments)

    print('Finished')
