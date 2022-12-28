"""
Main script used for training a PPO model with a recurrent neural network.

Written by C. F. De Inza Niemeijer.
"""

import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
from torch.nn import Tanh
from utils.environment_utils import make_env, copy_env
from utils.general import print_model, schedule_fn
from custom.custom_callbacks import CustomWandbCallback, CustomCallback
from arguments import get_main_args


def load_model(args, env):
    """
    Loads an existing Recurrent PPO model (a valid path must be specified), otherwise creates a new model.\n
    :param args: arguments from the command-line.
    :param env: Instance of the environment
    :return: model
    """

    model_path = args.model
    model = None

    # Wrap the environment in a Monitor and a DummyVecEnv wrappers:
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    # env = DummyVecEnv([make_vec_env] * 4)

    if model_path == '':
        print('No model provided for training. Making new model...')
        policy_kwargs = {
            "net_arch": None,
            "activation_fn": Tanh,
            "lstm_hidden_size": 256,        # number of hidden units for each LSTM layer. Default is 256
            "n_lstm_layers": 1,             # number of LSTM layers. Default is 1
            "shared_lstm": False,           # whether the LSTM is shared by the actor and the critic. Default is False
            "enable_critic_lstm": True,     # must be set to True if `shared_lstm` is False, and vice-versa
            "lstm_kwargs": None,            # additional kwargs for LSTM constructor
            # https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html#recurrentppo-policies
            # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        }
        model = RecurrentPPO(
            policy="MlpLstmPolicy",     # Network used for the policy and value function
            env=env,                    # environment where data is collected
            n_steps=4096,                # default is 128 (for RNN)
            batch_size=128,             # Default is 128 (for RNN)
            learning_rate=3e-4,         # Default is 3e-4
            clip_range=0.2,             # Default is 0.2
            n_epochs=35,                # Default is 10 (number of gradient descent steps per iteration)
            gamma=0.99,                 # discount factor
            policy_kwargs=policy_kwargs,
            verbose=1,
        )
        """
        MlpPolicy is a policy object that implements actor critic, using an MLP (2 layers of 64, with tanh func).
        In SB3, the term "policy" refers to the class that handles all the networks used for training (not only
        the network used to predict actions).
        Note that the PPO class automatically wraps the Gym environment into a Monitor and a DummyVecEnv.
        """
    else:
        print(f'Loading saved model "{model_path}"...')
        try:
            custom_objects = None
            # custom_objects = {
            #     "lr_schedule": schedule_fn(3e-4),
            #     "learning_rate": schedule_fn(3e-4),
            #     "clip_range": schedule_fn(0.2),
            # }
            model = RecurrentPPO.load(model_path, env=env, custom_objects=custom_objects)
            print('Successfully loaded model')
        except FileNotFoundError:
            print(f'No such file "{model_path}".\nExiting')
            exit()

    print_model(model)

    return model


def main(args):
    """
    Main function to run when this script is executed.\n
    - Creates the environments for training and evaluation\n
    - Creates/loads an RNN model\n
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
    train_env = make_env(reward_kwargs, quiet=True, config=env_config, stochastic=stochastic)
    eval_env = copy_env(train_env)

    # Load/create model:
    model = load_model(args, env=train_env)

    # Set-up the callback function:
    if save:
        if args.wandb:
            callback = CustomWandbCallback(
                env=eval_env,
                wandb_run=None,
                save_name="rnn_model",
                n_evals=50 if stochastic is True else 1,
                project="train",
                run_id=None,
                verbose=0,
            )
        else:
            callback = CustomCallback(
                env=eval_env,
                save_name="rnn_model",
                n_evals=50,
                verbose=0,
            )
        print(f"The best model will be saved in {callback.save_path}")
    else:
        callback = None
        print(f"Note. The model will NOT be saved.")

    # Train the model:
    if args.start > 0:  # Set the current number of timesteps (if resuming a previous run)
        model.num_timesteps = args.start
        print(f"Starting at num_timesteps = {model.num_timesteps}")  # Only relevant for logging
    print(f"Training the model for {steps} steps...")
    model.learn(
        total_timesteps=steps,
        callback=callback,
        reset_num_timesteps=False,  # True if args.start == 0 else False,
    )

    return


if __name__ == '__main__':
    arguments = get_main_args()
    # arguments.nosave = True
    # arguments.model = os.path.join("models", "rnn_model.zip")
    # arguments.wandb = True
    # arguments.start = 0  # Define the starting training step
    # arguments.steps = 2_000_000
    main(arguments)
