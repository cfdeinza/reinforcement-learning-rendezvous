"""
Evaluate a model over a single episode.

Written by C. F. De Inza Niemeijer.
"""

from custom.custom_rdv_env import Rendezvous3DOF
from main import load_model  # , load_env
from arguments import get_main_args


def evaluate(model, env):

    env.reset()

    print(f'Starting chaser position: {env.pos}')
    print(f'Starting chaser velocity: {env.vel}')
    print(f'Starting corridor axis: {env.corridor_axis}')

    obs = env.get_current_observation()
    total_reward = 0
    done = False
    info = {}

    while not done:
        action, state = model.predict(
            observation=obs,        # input to the policy network
            state=None,             # last hidden state (used for recurrent policies)
            episode_start=None,     # the last mask? (used for recurrent policies)
            deterministic=True      # whether or not to return deterministic actions (default is False)
        )
        """
        The outputs are the action and the next hidden state.
        for more information on the inputs and outpus of the predict function, 
        see https://github.com/DLR-RM/stable-baselines3/blob/ef10189d80dbb2efb3b5391cba4eb3c2ab5c7aae/stable_baselines3/common/policies.py#L307
        """

        obs, reward, done, info = env.step(action)
        total_reward += reward

    trajectory = info['trajectory']
    actions = info['actions']

    print(f'Total reward: {round(total_reward, 2)}')

    return


if __name__ == '__main__':
    arguments = get_main_args()
    environment = Rendezvous3DOF()
    # environment = load_env(arguments)
    saved_model = load_model(arguments, environment)

    evaluate(saved_model, environment)