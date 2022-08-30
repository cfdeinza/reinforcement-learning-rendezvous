"""
Main script used for training a PPO model.

Written by C. F. De Inza Niemeijer.
"""

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from rendezvous_env import RendezvousEnv
from arguments import get_args


def load_model(args, env):
    """
    Load an existing PPO model if a valid file name is given. Otherwise, create a new PPO model.\n
    :param args: command-line arguments.
    :param env: gym environment.
    :return: trained model.
    """

    model_name = args.model
    model = None

    if model_name == '':
        print('No model provided for training. Making new model...')
        n_steps = 3648  # number of steps to run between each model update
        model = PPO(MlpPolicy, env, n_steps=n_steps, verbose=1)
    else:
        print(f'Loading saved model "{model_name}"...')
        try:
            model = PPO.load(model_name, env=env)
            print('Successfully loaded model')
        except FileNotFoundError:
            print(f'No such file "{model_name}".\nExiting')
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

    steps = args.steps
    save = not args.nosave

    if save:
        eval_env = model.env

        callback = EvalCallback(
            eval_env,  # Environment used for evaluation (must be identical to the training environment)
            best_model_save_path='./models/',  # Path to the folder where the best model is saved (best_model.zip)
            log_path='./logs/',     # Path to the folder where the the evaluations info is saved (evaluations.npz)
            n_eval_episodes=1,      # Number of episodes tested in each evaluation
            eval_freq=10000,        # Time steps between evaluations
            deterministic=True,     # Stochastic or deterministic actions used for evaluations
            render=args.render      # Render the evaluations
        )
        print(f'The best model will be saved in {callback.best_model_save_path}')

    else:
        callback = None
        print(f'Note: The model will NOT be saved.')

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
    :param args: command-line arguments.
    :return: None
    """

    # Create an instance of the environment:
    env = RendezvousEnv()
    # env = Rendezvous3DOF()  # old 3DOF environment
    # env = Rendezvous3DOF(config=None)  # this was briefly used for ray rllib
    # env = gym.make('Pendulum-v1')  # simply using PendulumEnv() yields no `done` condition.

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
    # arguments.model = r'models\best_model.zip'
    main(arguments)
