"""
This script generates a trajectory using an existing model, and then plots the trajectory,
either as an animation or in a 2d plot.
"""

import os
import argparse
from rendezvous_env import RendezvousEnv
from utils.general import load_model
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


if __name__ == "__main__":
    environment = RendezvousEnv()
    arguments = get_args()
    arguments.path = ""  # prevent make_animation from crashing
    arguments.save = False  # make sure we don't save a gif of the animation
    arguments.model = os.path.join("models_fuel_coef_sweep_02", "model06.zip")
    saved_model = load_model(path=arguments.model, env=environment)

    # Plot a 3D animation:
    data = evaluate(saved_model, environment, arguments)
    make_animation(arguments, data=data)

    # # Iterate over multiple models:
    # data = {}
    # for i in range(1, 7):
    #     arguments.model = os.path.join("models_fuel_coef_sweep_02", "model0" + str(i) + ".zip")
    #     saved_model = load_model(path=arguments.model, env=RendezvousEnv())
    #     data[i] = evaluate(saved_model, RendezvousEnv(), arguments)
    # for i in range(1, 7):
    #     plot2d(arguments, data=data[i])
    #     # make_animation(arguments, data=data)
