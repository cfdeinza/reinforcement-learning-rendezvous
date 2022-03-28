from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from custom_env import Rendezvous3DOF, Attitude
from custom_callbacks import my_eval_callback
from arguments import get_args

"""
Main script used for training or evaluating a PPO model.

Written by C. F. De Inza Niemeijer.
"""


def load_env(args):
    """
    Create an instance of the environment and wrap it in the required Gym wrappers.
    :param args: Namespace containing arguments.
    :return: Wrapped environment.
    """

    env = None

    if args.env == 'rdv':
        env = Rendezvous3DOF()
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
    Load an existing PPO model if a valid file name is given. Otherwise, create a new PPO model.
    """

    mode = args.mode
    model_name = args.model
    model = None

    if model_name == '':
        if mode == 'train':
            print('No model provided for training. Making new model...')
            # model = PPO(MlpPolicy, env, verbose=1)
            model = PPO(MlpPolicy, env, verbose=1, use_sde=args.sde, sde_sample_freq=10)
            # MlpPolicy is a policy object that implements actor critic, using an MLP (2 layers of 64?). In SB3, the
            # term "policy" refers to the class that handles all the networks used for training (not only the network
            # used to predict actions).
            # Note that the PPO class automatically wraps the Gym environment into a Monitor and a DummyVecEnv.
        elif mode == 'eval':
            print('Need to specify a model for testing.\nExiting')
            exit()
        else:
            print(f'Mode "{mode}" not recognized.\nExiting')
            exit()
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
    By default, it runs 5 episodes on each evaluation.
    """

    steps = args.steps
    save = not args.nosave

    if save:
        eval_env = model.env

        callback = EvalCallback(
            eval_env,  # Environment used for evaluation (must be identical to the training environment)
            best_model_save_path='./models/',  # Path to the folder where the best model is saved (best_model.zip)
            log_path='./logs/',     # Path to the folder where the the evaluations info is saved (evaluations.npz)
            n_eval_episodes=5,      # Number of episodes tested in each evaluation
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


def evaluate(args, model):
    """
    Test the performance of the model for a given number of episodes.
    Renders each episode and then prints the average reward (and its standard dev).
    """

    save = not args.nosave

    if save:
        callback = my_eval_callback
    else:
        callback = None
        print(f'Note: The trajectory will NOT be saved.')

    print('Evaluating...')
    episodes = 1

    mean_reward, std_reward = evaluate_policy(
        model,
        model.env,
        n_eval_episodes=episodes,
        render=args.render,
        callback=callback)

    print(f'Mean reward over {episodes} episodes: {mean_reward:.2f} +/- {std_reward:.2f}')

    return


def main(args):
    """
    Main function to run.
    The arguments ("mode" and "model") are parsed from the command line.
    """

    # Create an instance of the environment: Rendezvous3DOF() or Attitude()
    env = load_env(args)

    # args.model = 'PPO_model'
    model = load_model(args, env)  # Load/create the model

    if args.mode == 'train':

        train(args, model)

    elif args.mode == 'eval':

        evaluate(args, model)

    else:

        print(f'Mode "{args.mode}" not recognized.\nExiting')
        exit()

    pass


if __name__ == '__main__':
    arguments = get_args()
    # arguments.mode = 'eval'
    # arguments.nosave = False
    # arguments.model = r'models\best_model.zip'
    # arguments.env = 'rdv'
    main(arguments)
