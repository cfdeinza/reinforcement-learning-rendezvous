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
from utils.general import print_model, schedule_fn
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
            batch_size=128,         # Default: 64 for MLP, 128 for RNN
            n_epochs=40,            # number of gradient descent steps per iteration
            clip_range=0.25,        # Default: 0.2
            policy_kwargs={"activation_fn": Tanh},
            verbose=1,
        )
    else:
        print(f'Loading saved model "{model_path}"...', end=' ')
        try:
            custom_objects = None
            model = PPO.load(model_path, env=env, custom_objects=custom_objects)
            print('Done')
        except FileNotFoundError:
            print(f'No such file "{model_path}".\nExiting')
            exit()

    print_model(model)

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
    env_config = {}
    stochastic = True
    train_env = make_env(reward_kwargs, quiet=False, config=env_config, stochastic=stochastic)
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
                n_evals=50 if stochastic is True else 1,
                project="train",
                run_id=None,  # use this to resume a paused/crashed run
                verbose=0,
            )   # Custom callback to track experiment with Weights & Biases
        else:
            callback = CustomCallback(
                env=eval_env,
                save_name="mlp_model",
                n_evals=50,
                verbose=0,
            )   # Custom callback to save the best model
        print(f'The best model will be saved in {callback.save_path}')
    else:
        callback = None
        print(f'Note: The model will NOT be saved.')

    # Train the model:
    print(f"Training the model for {steps} steps...")
    model.learn(
        total_timesteps=steps,
        callback=callback,
        reset_num_timesteps=False,
    )

    return


if __name__ == '__main__':
    arguments = get_main_args()
    # arguments.nosave = True
    # arguments.model = os.path.join("models", "mlp_model.zip")
    # arguments.wandb = True
    # arguments.steps = 8_000_000
    main(arguments)
