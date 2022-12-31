import numpy as np
from rendezvous_env import RendezvousEnv
from new_env import NewEnv
from copy import deepcopy
from stable_baselines3.common.monitor import Monitor
from utils.quaternions import quat2rot


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

    rc0 = config.get("rc0", None)
    if rc0 is not None and not isinstance(rc0, np.ndarray):
        rc0 = np.array([0., -rc0, 0.])  # Convert to array if necessary

    wt0 = config.get("wt0", None)
    if wt0 is not None and not isinstance(wt0, np.ndarray):
        wt0 = np.array([0., 0., wt0])  # Convert to array if necessary

    env = RendezvousEnv(
        rc0=rc0,  # config.get("rc0"),
        vc0=config.get("vc0"),
        qc0=config.get("qc0"),
        wc0=config.get("wc0"),
        qt0=config.get("qt0"),
        wt0=wt0,  # config.get("wt0"),
        rc0_range=config.get("rc0_range"),
        vc0_range=config.get("vc0_range"),
        qc0_range=config.get("qc0_range"),
        wc0_range=config.get("wc0_range"),
        qt0_range=config.get("qt0_range"),
        wt0_range=config.get("wt0_range"),
        reward_kwargs=reward_kwargs,
        koz_radius=config.get("koz_radius"),
        corridor_half_angle=config.get("corridor_half_angle"),
        h=config.get("h"),
        dt=config.get("dt"),
        t_max=config.get("t_max"),
        quiet=quiet
    )

    print_env(env)

    return env


def copy_env(env):
    """
    Create a deep copy of the given environment.\n
    :param env: instance of an environment.
    :return: identical instance of the given environment.
    """

    return deepcopy(env)


def print_env(env):
    """
    Print the attributes of the given environment.\n
    :param env: object
    :return: None
    """

    print(f"\nEnvironment:")
    for key, val in vars(env).items():
        if val is not None:
            print(f"{key}: {val}")

    return None


def make_new_env(config: dict):

    env = NewEnv(config)

    print_env(env)

    return env


def print_state(env):
    print(f"Environment state at t = {env.t} s:")
    print(f"rc: {env.rc}, mag: {round(np.linalg.norm(env.rc), 3)} m")
    print(f"vc: {env.vc}, mag: {round(np.linalg.norm(env.vc), 3)} m/s")
    print(f"wc: {np.degrees(env.wc)}, mag: {round(np.linalg.norm(np.degrees(env.wc)), 3)} deg/s")
    print(f"wc: {env.chaser2lvlh(np.degrees(env.wc))} (expressed in LVLH)")
    print(f"wt: {np.degrees(env.wt)}, mag: {round(np.linalg.norm(np.degrees(env.wt)), 3)} deg/s")
    print(f"wt: {env.target2lvlh(np.degrees(env.wt))} (expressed in LVLH)")
    axis, angle = quat2rot(env.qt)
    print(f"qt: {env.qt}, e = {axis}, theta = {np.degrees(angle)} deg")
    # print(f": {}, mag: {round(np.linalg.norm(), 3)} ")
    return


def make_vec_env():
    """
    In case I am using multiple environments.\n
    :return:
    """
    env = RendezvousEnv()
    return Monitor(env)
