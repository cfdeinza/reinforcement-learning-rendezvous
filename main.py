"""
Main script used for training a PPO model.

Written by C. F. De Inza Niemeijer.
"""

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from custom.custom_callbacks import CustomWandbCallback, CustomCallback
from rendezvous_env import RendezvousEnv
from arguments import get_args
# from custom.custom_model import CustomPPO


def load_model(args, env):
    """
    Load an existing PPO model if a valid file name is given. Otherwise, create a new PPO model.\n
    :param args: command-line arguments.
    :param env: gym environment.
    :return: trained model.
    """

    model_path = args.model
    model = None

    if model_path == '':
        print('No model provided for training. Making new model...')
        n_steps = 640*3  # 3648  # num of steps to run between each model update  # each env does this amount of steps
        # model = CustomPPO(MlpPolicy, env, n_steps=n_steps, verbose=1)
        model = PPO(MlpPolicy, env, n_steps=n_steps, verbose=1)
    else:
        print(f'Loading saved model "{model_path}"...', end=' ')
        try:
            model = PPO.load(model_path, env=env)
            print('Done')
        except FileNotFoundError:
            print(f'No such file "{model_path}".\nExiting')
            exit()

    return model


def train(args, model):
    """
    Train the model for a given number of time steps.
    Uses the EvalCallback to periodically evaluate and save the model.
    By default, it runs 5 episodes on each evaluation.\n
    :param args: command-line arguments.
    :param model: PPO model.
    :return:
    """

    # model.wandb_start()

    steps = args.steps
    save = not args.nosave

    if save:
        if args.wandb:
            callback = CustomWandbCallback(RendezvousEnv)   # Custom callback to track experiment with Weights & Biases
        else:
            callback = CustomCallback(RendezvousEnv)        # Custom callback to save the best model
        print(f'The best model will be saved in {callback.best_model_save_path}')
    else:
        callback = None
        print(f'Note: The model will NOT be saved.')

    print('Training...')
    model.learn(total_timesteps=steps, callback=callback)

    # model.wandb_end()

    return


def main(args):
    """
    Main function to run.
    The arguments ("mode" and "model") are parsed from the command line.\n
    :param args: command-line arguments.
    :return: None
    """

    # Create an instance of the environment:
    env = RendezvousEnv()

    # Wrap the environment in a Monitor and a DummyVecEnv wrappers:
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # Load/create the model:
    model = load_model(args, env)

    # Train the model:
    train(args, model)

    pass


if __name__ == '__main__':
    arguments = get_args()
    # arguments.n_envs = 4
    # arguments.model = r'models\best_model.zip'
    # arguments.nosave = True
    main(arguments)
