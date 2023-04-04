"""
This script generates a trajectory using an existing model, and then plots the trajectory,
either as an animation or in a 2d plot.
"""

import os
import argparse
import numpy as np
# from rendezvous_env import RendezvousEnv
from utils.environment_utils import make_env
from utils.general import load_model
# from utils.quaternions import quat2mat
from rendezvous_eval import evaluate
from rendezvous_plot_2d import plot2d_response, plot2d_error
from rendezvous_plot_vpython import make_animation
from time import perf_counter


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


def main(args):

    # Make an instance of the environment:
    reward_kwargs = None
    # config = None
    config = dict(
        dt=1,
        t_max=60,
        koz_radius=5,
        # rc0_range=0,
        rc0=np.array([0, -10, 0]),
        qc0=np.array([np.cos(np.radians(0)/2), 0, 0, 0]),
        # qt0=np.array([np.cos(np.pi/4)-0.09, 0, 0, np.sin(np.pi/4)-0.09]),
        # qt0_range=0,
        wt0=np.radians(np.array([0, 0, 0])),
        # wt0_range=0,
    )
    stochastic = False
    eval_env = make_env(reward_kwargs, quiet=False, config=config, stochastic=stochastic)

    # Load an existing model:
    saved_model = load_model(path=args.model, env=eval_env)
    # Evaluate the model:
    data = evaluate(saved_model, eval_env, args)
    plot2d_response(args, data=data)
    plot2d_error(args, data=data)
    make_animation(args, data=data)

    # # Iterate over multiple models:
    # datas = []
    # for i in range(1, 6):
    #     # path = os.path.join("models_fuel_coef_sweep_02", "model0" + str(i) + ".zip")
    #     path = os.path.join("models_att_coef_sweep_01", "rew_tune_model_0" + str(i) + ".zip")
    #     saved_model = load_model(path=path, env=eval_env)
    #     # datas[i] = evaluate(saved_model, eval_env, args)
    #     d = evaluate(saved_model, env=eval_env, args=args)
    #     datas.append(d.copy())
    # for data in datas:
    #     plot2d_response(args=None, data=data)
    #     plot2d_error(args=None, data=data)
    #     # make_animation(args, data=data)


if __name__ == "__main__":
    # qt0 = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0])
    # wt0 = np.matmul(quat2mat(qt0).T, np.array([0, 0, np.radians(3)]))
    # environment = RendezvousEnv(qt0=qt0, wt0=wt0)

    arguments = get_args()
    arguments.path = ""     # prevent make_animation from crashing
    arguments.save = False  # make sure we don't save a gif of the animation
    # arguments.model = os.path.join("models_fuel_coef_sweep_02", "model01.zip")
    # arguments.model = os.path.join("models", "mlp_model_att_01.zip")
    # arguments.model = os.path.join("models_att_coef_sweep_01", "rew_tune_model_01.zip")
    # arguments.model = os.path.join("models", "rnn_model_2_02.zip")
    # arguments.model = r"C:\Users\charl\Downloads\rnn_model_decent5.zip"
    # arguments.model = r"C:\Users\charl\Downloads\rnn_model_box_04_08.zip"
    # arguments.model = r"C:\Users\charl\Downloads\rnn_model_final2_01.zip"
    main(arguments)
