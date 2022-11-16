"""
    This file contains the arguments to parse at the command line.
    File `main.py` will call get_main_args(), which then returns the arguments.
    The arguments are:
        - mode: "train" or "eval" (indicates whether to train or evaluate the model).
        - model: Name of an existing file (to load a previously saved model).
        - steps: Number of training steps.
        - env: Select the environment ("rdv" for Rendezvous3DOF, or "att" for Attitude).
        - nosave: Use this flag to NOT save the results.
        - render: Use this flag to render the episodes.
        - render: Use this flag to enable gSDE (Generalized State Dependent Exploration).

    Written by C. F. De Inza Niemeijer.
"""

import argparse


def get_main_args():
    """
    Parses the arguments from the command line when calling `main.py`.\n
    :return: Namespace containing the arguments.
    """
    parser = argparse.ArgumentParser()

    # default_mode = 'train'
    # parser.add_argument(
    #     '--mode',
    #     dest='mode',
    #     type=str,
    #     default=default_mode,
    #     help=f'Use \'train\' to train the model, \'eval\' to evaluate the model. Default is \'{default_mode}\'.'
    # )
    parser.add_argument(
        '--model',
        dest='model',
        type=str,
        default='',
        help='Provide the name of an existing model to evaluate or to continue training.'
    )
    default_steps = 200_000
    parser.add_argument(
        '--steps',
        dest='steps',
        type=int,
        default=default_steps,
        help=f'Select the number of training steps. Default is {default_steps}.'
    )
    parser.add_argument(
        '--start',
        dest='start',
        type=int,
        default=0,
        help=f'Define the starting timestep (Does not affect training, only used for logging purposes). Default is 0.'
    )
    parser.add_argument(
        '--nosave',
        dest='nosave',
        type=bool,
        nargs='?',
        const=True,
        default=False,
        help='Use this flag to avoid saving the results.'
    )
    parser.add_argument(
        '--wandb',
        dest='wandb',
        type=bool,
        nargs='?',
        const=True,
        default=False,
        help='Use this flag to track the experiment using Weights and Biases.'
    )
    parser.add_argument(
        '--render',
        dest='render',
        type=bool,
        nargs='?',
        const=True,
        default=False,
        help='Use this flag to render the episodes.'
    )
    # parser.add_argument(
    #     '--sde',
    #     dest='sde',
    #     type=bool,
    #     nargs='?',
    #     const=True,
    #     default=False,
    #     help='Use this flag to enable gSDE (Generalized State Dependent Exploration).'
    # )

    args = parser.parse_args()

    return args


def get_tune_args():
    """
    Parses the arguments from the command line when calling the tuning scripts.\n
    :return: Namespace containing the arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project",
        dest="project",
        type=str,
        default="",
        help="Set the name of the Weights & Biases project where this run(s) will be saved."
    )

    default_iterations = 25
    parser.add_argument(
        "--iterations",
        dest="iterations",
        type=int,
        default=default_iterations,
        help=f"Set the number of iterations (rollout-optimization loops). Default is {default_iterations}"
    )

    args = parser.parse_args()

    return args


def get_monte_carlo_args():
    """
    Parses the argumennts when calling the monte_carlo script from the command line.\n
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        dest="model",
        type=str,
        default="",
        help="Path to the model. Default is ''"
    )
    parser.add_argument(
        "--save",
        dest="save",
        type=bool,
        nargs="?",
        default=False,  # default value
        const=True,     # value when we call the argument
        help="Use this flag to save the results."
    )

    args = parser.parse_args()

    return args
