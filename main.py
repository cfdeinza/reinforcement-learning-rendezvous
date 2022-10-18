"""
Main script used for training a PPO model.

Written by C. F. De Inza Niemeijer.
"""

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.nn import Tanh
from custom.custom_callbacks import CustomWandbCallback, CustomCallback
from rendezvous_env import RendezvousEnv
from arguments import get_main_args
# from custom.custom_model import CustomPPO


def load_model(args, env):
    """
    Load an existing PPO model if a valid path is given. Otherwise, create a new PPO model.\n
    :param args: command-line arguments.
    :param env: gym environment.
    :return: model.
    """

    model_path = args.model
    model = None

    if model_path == '':
        print('No model provided for training. Making new model...')
        # model = CustomPPO(MlpPolicy, env, n_steps=n_steps, verbose=1)
        model = PPO(
            policy=MlpPolicy,   # Network used for the Policy and Value function
            env=env,            # environment where data is collected
            n_steps=2048,       # num of steps per rollout  # each env does this amount of steps
            n_epochs=40,        # number of gradient descent steps per iteration
            gamma=1,            # discount factor
            policy_kwargs={"activation_fn": Tanh},
            verbose=1,
        )
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
    :param model: PPO model to be trained.
    :return: None
    """

    # model.wandb_start()

    steps = args.steps
    save = not args.nosave

    if save:
        if args.wandb:
            callback = CustomWandbCallback(
                env=RendezvousEnv,
                reward_kwargs=None,
                wandb_run=None,
                save_name="mlp_model",
                n_evals=5,
                verbose=0,
            )   # Custom callback to track experiment with Weights & Biases
        else:
            callback = CustomCallback(
                env=RendezvousEnv,
                reward_kwargs=None,
                save_name="mlp_model",
                n_evals=5,
                verbose=0,
            )   # Custom callback to save the best model
        print(f'The best model will be saved in {callback.save_path}')
    else:
        callback = None
        print(f'Note: The model will NOT be saved.')

    print('Training...')
    model.learn(total_timesteps=steps, callback=callback)

    # model.wandb_end()

    return


def main(args):
    """
    Main function to run when this file is executed.
    The arguments ("mode" and "model") are parsed from the command line.\n
    :param args: command-line arguments.
    :return: None
    """

    # Create an instance of the environment:
    env = RendezvousEnv(reward_kwargs=None)

    # Wrap the environment in a Monitor and a DummyVecEnv wrappers:
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # Load/create the model:
    model = load_model(args, env)

    # Train the model:
    train(args, model)

    pass


if __name__ == '__main__':
    arguments = get_main_args()
    # arguments.n_envs = 4
    # arguments.model = r'models\best_model.zip'
    # arguments.nosave = True
    main(arguments)
