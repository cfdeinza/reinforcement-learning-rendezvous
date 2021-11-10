import argparse

"""
    This file contains the arguments to parse at the command line.
    File main.py will call get_args, which then returns the arguments.
    The arguments are:
        mode: "train" or "eval" (indicates whether to train or evaluate the model).
        model: Name of an existing file (to load a previously saved model).
        steps: Number of training steps.
        env: Select the environment ("rdv" for Rendezvous3DOF, or "att" for Attitude).
        nosave: Use this flag to not save the results.
"""


def get_args():
    """
    Parses the arguments from the command line.
    :return: Namespace containing the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        dest='mode',
        type=str,
        default='train',
        help='Use \'train\' to train the model, \'eval\' to evaluate the model'
    )
    parser.add_argument(
        '--model',
        dest='model',
        type=str,
        default='',
        help='Provide the name of an existing model to evaluate or continue training'
    )
    parser.add_argument(
        '--steps',
        dest='steps',
        type=int,
        default=200000,
        help='Select the number of training steps'
    )
    # HACK: The following arguments are not intuitive. Try something better.
    parser.add_argument(
        '--env',
        dest='env',
        type=str,
        default='',
        help='Use \'rdv\' to use the Rendezvous3DOF environment, or \'att\' to use the Attitude environment'
    )
    # Save the results: 1 to save, otherwise does not save.
    parser.add_argument(
        '--nosave',
        dest='nosave',
        type=bool,
        nargs='?',
        const=True,
        default=False,
        help='Use this flag to avoid saving the results.'
    )

    args = parser.parse_args()

    return args
