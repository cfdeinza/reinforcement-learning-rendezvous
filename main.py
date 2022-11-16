"""
Main script used for training a PPO model.

Written by C. F. De Inza Niemeijer.
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.nn import Tanh
# from rendezvous_env import RendezvousEnv
from utils.environment_utils import make_env, copy_env
from custom.custom_callbacks import CustomWandbCallback, CustomCallback
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

    # Wrap the environment in a Monitor and a DummyVecEnv wrappers:
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    if model_path == '':
        print('No model provided for training. Making new model...')
        # model = CustomPPO(MlpPolicy, env, n_steps=n_steps, verbose=1)
        model = PPO(
            policy=MlpPolicy,       # Network used for the Policy and Value function
            env=env,                # environment where data is collected
            learning_rate=2e-3,     # Default: 3e-4
            n_steps=2048,           # num of steps per rollout  # each env does this amount of steps
            batch_size=128,         # Default: 64 for MLP, 128 for RNN
            n_epochs=40,            # number of gradient descent steps per iteration
            clip_range=0.25,        # Default: 0.2
            gamma=0.99,             # discount factor
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


def main(args):
    """
    Main function to run when this script is executed.\n
    - Creates the environments for training and evaluation\n
    - Creates/loads an MLP model\n
    - Trains the model for a given number of time steps.\n
    - Uses a custom callback to periodically evaluate and save the model.\n
    :param args: command-line arguments.
    :return: None
    """

    steps = args.steps
    save = not args.nosave

    # Make envs:
    reward_kwargs = None
    train_env = make_env(reward_kwargs, quiet=False, config=None, stochastic=True)
    eval_env = copy_env(train_env)

    # Load/create model:
    model = load_model(args, env=train_env)

    # Set-up the callback function:
    if save:
        if args.wandb:
            callback = CustomWandbCallback(
                env=eval_env,
                wandb_run=None,
                save_name="mlp_model",
                n_evals=10,
                project="train",
                run_id=None,  # use this to resume a paused/crashed run
                verbose=0,
            )   # Custom callback to track experiment with Weights & Biases
        else:
            callback = CustomCallback(
                env=eval_env,
                save_name="mlp_model",
                n_evals=10,
                verbose=0,
            )   # Custom callback to save the best model
        print(f'The best model will be saved in {callback.save_path}')
    else:
        callback = None
        print(f'Note: The model will NOT be saved.')

    # Train the model:
    if args.start > 0:  # Set the current number of timesteps (if resuming a previous W&B run)
        model.num_timesteps = args.start
        print(f"Starting at num_timesteps = {model.num_timesteps}")  # Only relevant for logging
    print(f"Training the model for {steps} steps...")
    model.learn(
        total_timesteps=steps,
        callback=callback,
        reset_num_timesteps=True if args.start == 0 else False,
    )

    return


if __name__ == '__main__':
    arguments = get_main_args()
    # arguments.n_envs = 4
    # arguments.nosave = True
    arguments.model = os.path.join("models", "mlp_model.zip")
    # arguments.wandb = True
    # arguments.start = 0  # Define the starting training step
    # arguments.steps = 20_000_000
    main(arguments)
