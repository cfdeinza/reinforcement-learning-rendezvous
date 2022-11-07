from rendezvous_env import RendezvousEnv
from copy import deepcopy


def make_env(reward_kwargs, quiet=True, config=None, stochastic=True) -> RendezvousEnv:
    """
    Creates an instance of the Rendezvous environment.\n
    :param reward_kwargs: dictionary containing keyword arguments for the reward function
    :param quiet: `True` to supress printed outputs, `False` to print outputs
    :param config: dictionary with additional configuration parameters
    :param stochastic: whether or not to randomize the initial state of the system
    :return: instance of the environment
    """

    if reward_kwargs is None:
        print("Note: reward_kwargs have not been defined. Using default values.")

    if config is None:
        config = {}

    if stochastic is False:
        # Set range of initial state to zero:
        for i in ["rc0_range", "vc0_range", "qc0_range", "wc0_range", "qt0_range", "wt0_range"]:
            config[i] = 0
        print("Note: `stochastic` was set to False. the initial state of the environment will NOT be randomized.")

    env = RendezvousEnv(
        # rc0=None if config.get("rc0") is None else np.array([0, -config.get("rc0"), 0]),
        rc0=config.get("rc0"),
        vc0=config.get("vc0"),
        qc0=config.get("qc0"),
        wc0=config.get("wc0"),
        qt0=config.get("qt0"),
        # wt0=None if config.get("wt0") is None else np.array([0, 0, config.get("wt0")]),
        wt0=config.get("wt0"),
        rc0_range=config.get("rc0_range"),
        vc0_range=config.get("vc0_range"),
        qc0_range=config.get("qc0_range"),
        wc0_range=config.get("wc0_range"),
        qt0_range=config.get("qt0_range"),
        wt0_range=config.get("wt0_range"),
        koz_radius=config.get("koz_radius"),
        corridor_half_angle=config.get("corridor_half_angle"),
        h=config.get("h"),
        dt=config.get("dt"),
        reward_kwargs=reward_kwargs,
        quiet=quiet
    )

    print(f"Made new environment:")
    for key, val in vars(env).items():
        print(f"{key}: {val}")

    return env


def copy_env(env):
    """
    Create a deep copy of the given environment.\n
    :param env: instance of an environment.
    :return: identical instance of the given environment.
    """

    return deepcopy(env)
