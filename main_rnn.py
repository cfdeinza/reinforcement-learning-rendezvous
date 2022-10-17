"""
Main script used for training a PPO model with a recurrent neural network.

Written by C. F. De Inza Niemeijer.
"""

# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.callbacks import EvalCallback
# import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from rendezvous_env import RendezvousEnv
from custom.custom_att_env import Attitude
from custom.custom_callbacks import CustomWandbCallback, CustomCallback
from arguments import get_main_args
from sb3_contrib import RecurrentPPO
# from sb3_contrib.ppo_recurrent import MlpLstmPolicy


def load_env(args):
    """
    Create an instance of the environment and wrap it in the required Gym wrappers.\n
    :param args: Namespace containing arguments.
    :return: Wrapped environment.
    """

    env = None

    if args.env == 'rdv':
        env = RendezvousEnv()
        # env = Rendezvous3DOF()  # this works
        # env = Rendezvous3DOF(config=None)  # this was briefly used for ray rllib
        # env = PendulumEnv()  # this has no 'done' condition
        # env = gym.make('Pendulum-v1')  # this works
    elif args.env == 'att':
        env = Attitude()
    elif args.env == '':
        print('Need to specify an environment.\nExiting')
        exit()
    else:
        print(f'Environment "{args.env}" not recognized.\nExiting')
        exit()

    # Wrap the environment in a Monitor and a DummyVecEnv wrappers:
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    return env


def load_model(args, env):
    """
    Load an existing PPO model if a valid file name is given. Otherwise, create a new PPO model.\n
    """

    model_path = args.model
    model = None

    if model_path == '':
        print('No model provided for training. Making new model...')
        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=RendezvousEnv(),
            verbose=1,
            n_steps=2048,  # default is 2048 (multiple of batch_size, which is 64 by default)
            gamma=1,
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
            model = RecurrentPPO.load(model_path, env=env)
            print('Successfully loaded model')
        except FileNotFoundError:
            print(f'No such file "{model_path}".\nExiting')
            exit()

    return model


def lr_schedule(initial_value: float):
    """
    Learning rate scheduler. Default learning rate for PPO is 3e-4.\n
    :param initial_value: Initial learning rate.
    :return: Callback function that computes current learning rate
    """

    def func(progress_remaining: float) -> float:
        """
        Computes current learning rate based on progress remaining.\n
        :param progress_remaining: value that will decrease from 1 to 0.
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def train(args, model):
    """
    Train the model for a given number of time steps.
    Uses the EvalCallback to periodically evaluate and save the model.
    By default, it runs 5 episodes on each evaluation.\n
    """

    steps = args.steps
    save = not args.nosave

    if save:
        if args.wandb:
            callback = CustomWandbCallback(
                env=RendezvousEnv,
                reward_kwargs=None,
                wandb_run=None,
                save_name="rnn_model",
                verbose=0,
            )
        else:
            callback = CustomCallback(
                env=RendezvousEnv,
                reward_kwargs=None,
                save_name="rnn_model",
                verbose=0,
            )
        print(f"The best model will be saved in {callback.save_path}")
    else:
        callback = None
        print(f"Note. The model will NOT be saved.")
        # eval_env = model.env
        # n_envs = eval_env.num_envs
        #
        # callback = EvalCallback(
        #     eval_env,  # Environment used for evaluation (must be identical to the training environment)
        #     best_model_save_path='./models/',  # Path to the folder where the best model is saved (best_model.zip)
        #     log_path='./logs/',         # Path to the folder where the the evaluations info is saved (evaluations.npz)
        #     n_eval_episodes=1,          # Number of episodes tested in each evaluation
        #     eval_freq=10000/n_envs,     # Time steps between evaluations
        #     deterministic=True,         # Stochastic or deterministic actions used for evaluations
        #     render=args.render          # Render the evaluations
        # )
        # print(f'The best model will be saved in {callback.save_path}')
    print('Training...')
    model.learn(total_timesteps=steps, callback=callback)

    if save:  # Save the model when training is complete
        last_model_path = './models/last_model.zip'
        model.save(last_model_path)
        print(f'Saved the last model to "{last_model_path}"')

    return


def main(args):
    """
    Main function to run.
    The arguments ("mode" and "model") are parsed from the command line.\n
    """

    # Create an instance of the environment:
    env = RendezvousEnv(reward_kwargs=None)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    # env = Rendezvous3DOF(config=None)  # this was briefly used for ray rllib
    # env = gym.make('Pendulum-v1')  # simply using PendulumEnv() yields no `done` condition.
    # env = Rendezvous3DOF()  # this works
    # env = Rendezvous3DOF(config=None)  # this was briefly used for ray rllib

    # args.model = 'PPO_model'
    model = load_model(args, env)  # Load/create the model

    train(args, model)

    pass


if __name__ == '__main__':
    arguments = get_main_args()
    # arguments.nosave = False
    # arguments.model = os.path.join("models", "best_model.zip")
    # arguments.wandb = True
    main(arguments)
