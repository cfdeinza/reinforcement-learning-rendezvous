"""
Evaluate a model over a single episode.

Written by C. F. De Inza Niemeijer.
"""

from custom.custom_att_env import Attitude
from main import load_model  # , load_env
from arguments import get_args
import numpy as np


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


def evaluate(model, env):

    env.reset()

    print(f'Starting attitude: {env.state[0:4]}')
    print(f'Starting rotation rate: {env.state[4:]}')

    obs = env.state
    total_reward = 0
    done = False
    info = {}
    k = 0
    while not done:
        # action = get_action(model, obs)
        action = np.array([1 * np.cos(k), 0, 0])
        obs, reward, done, info = env.step(action)
        total_reward += reward
        k += 0.1

    trajectory = info['trajectory']
    actions = info['actions']

    print(f'Total reward: {round(total_reward, 2)}')

    return


if __name__ == '__main__':
    arguments = get_args()
    environment = Attitude()
    # environment = load_env(arguments)
    saved_model = load_model(arguments, environment)

    evaluate(saved_model, environment)
