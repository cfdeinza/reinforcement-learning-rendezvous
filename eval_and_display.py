"""
This script generates a trajectory using an existing model, and then plots the trajectory,
either as an animation or in a 2d plot.
"""

import os
import argparse
# import numpy as np
from rendezvous_env import RendezvousEnv
from utils.general import load_model
# from utils.quaternions import quat2mat
from rendezvous_eval import evaluate
from rendezvous_plot_2d import plot2d
from rendezvous_plot_vpython import make_animation


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        dest="model",
        type=str,
        default="",
        help="Provide the name of an existing model",
    )
    parser.add_argument(
        "--save",
        dest="save",
        type=bool,
        nargs="?",
        default=False,  # default value
        const=True,  # value when we call the argument
        help="Use this flag to save the results."
    )
    args = parser.parse_args()
    return args


def make_env(reward_kwargs, quiet=True) -> RendezvousEnv:
    """
    Creates an instance of the Rendezvous environment.\n
    :param reward_kwargs: dictionary containing keyword arguments for the reward function
    :param quiet: `True` to supress printed outputs, `False` to print outputs
    :return:
    """

    env = RendezvousEnv(reward_kwargs=reward_kwargs, quiet=quiet)

    return env


def main(args):

    # Make an instance of the environment:
    reward_kwargs = None
    eval_env = make_env(reward_kwargs, quiet=False)
    if reward_kwargs is None:
        print("Note: reward_kwargs have not been defined. Using default values.")

    # Load an existing model:
    saved_model = load_model(path=args.model, env=eval_env)
    # Evaluate the model:
    data = evaluate(saved_model, eval_env, args)
    # make_animation(args, data=data)
    plot2d(args, data=data)

    # # Iterate over multiple models:
    # data = {}
    # for i in range(1, 7):
    #     args.model = os.path.join("models_fuel_coef_sweep_02", "model0" + str(i) + ".zip")
    #     saved_model = load_model(path=args.model, env=eval_env)
    #     data[i] = evaluate(saved_model, eval_env, args)
    # for i in range(1, 7):
    #     plot2d(args, data=data[i])
    #     # make_animation(args, data=data)


if __name__ == "__main__":
    # qt0 = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0])
    # wt0 = np.matmul(quat2mat(qt0).T, np.array([0, 0, np.radians(3)]))
    # environment = RendezvousEnv(qt0=qt0, wt0=wt0)

    arguments = get_args()
    arguments.path = ""     # prevent make_animation from crashing
    arguments.save = False  # make sure we don't save a gif of the animation
    # arguments.model = os.path.join("models_fuel_coef_sweep_02", "model01.zip")
    arguments.model = os.path.join("models", "rnn_model_01.zip")
    main(arguments)
