"""
Script to run a sensitivity analysis on the learning algorithm.
The parameters of the environment will be slightly modified to see how they affect they learning process.
(Similar to the tuning scripts, but varying the parameters of the environment instead)
"""

import wandb
import numpy as np
from math import radians
from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.nn import Tanh
from rendezvous_env import RendezvousEnv
from custom.custom_callbacks import CustomWandbCallback
from arguments import get_tune_args
from functools import partial


def make_model(policy, env, config=None):
    """
    Create a model using the hyperparameters given by the sweep.
    :param policy: policy for the model to use
    :param env: environment for the model to learn on
    :param config: configuration dictionary passed by the sweep
    :return: model
    """

    if config is None:
        config = {}

    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    model = PPO(
        policy,
        env,
        learning_rate=2e-3,     # Default: 3e-4
        n_steps=2048,           # Default: 2048 for MLP, 128 for RNN
        batch_size=128,         # Default: 64 for MLP, 128 for RNN
        n_epochs=40,            # Default: 10
        clip_range=0.25,        # Default: 2
        policy_kwargs={"activation_fn": Tanh},
        seed=0,
        verbose=1,
    )

    return model


def make_env(reward_kwargs, quiet=True, config=None) -> RendezvousEnv:
    """
    Creates an instance of the Rendezvous environment.\n
    :param reward_kwargs: dictionary containing keyword arguments for the reward function
    :param quiet: `True` to supress printed outputs, `False` to print outputs
    :param config: dictionary with additional configuration parameters
    :return: instance of the environment
    """

    if config is None:
        config = {}

    env = RendezvousEnv(
        rc0=None if config.get("rc0") is None else np.array([0, -config.get("rc0"), 0]),
        vc0=config.get("vc0"),
        qc0=config.get("qc0"),
        wc0=config.get("wc0"),
        qt0=config.get("qt0"),
        wt0=None if config.get("wt0") is None else np.array([0, 0, config.get("wt0")]),
        rc0_range=0,  # config.get("rc0_range"),
        vc0_range=0,  # config.get("vc0_range"),
        qc0_range=0,  # config.get("qc0_range"),
        wc0_range=0,  # config.get("wc0_range"),
        qt0_range=0,  # config.get("qt0_range"),
        wt0_range=0,  # config.get("wt0_range"),
        koz_radius=config.get("koz_radius"),
        corridor_half_angle=config.get("corridor_half_angle"),
        h=config.get("h"),
        dt=config.get("dt"),
        reward_kwargs=reward_kwargs,
        quiet=quiet
    )

    return env


def train_function(iterations):
    """
    Function that will be executed on each run of the sweep.
    :return: None
    """

    with wandb.init() as run:

        # print(f"Starting Weights & Biases run. ID: {run.id}")

        # Make environments:
        reward_kwargs = None
        train_env = make_env(reward_kwargs, quiet=True, config=wandb.config)
        eval_env = make_env(reward_kwargs, quiet=False, config=wandb.config)
        if reward_kwargs is None:
            print("Note: reward_kwargs have not been defined. Using default values")

        print("check: envs created correctly.")

        model = make_model(MlpPolicy, train_env, config=wandb.config)
        print("check: model created correctly")
        # Check that the hyperparameters have been updated:
        # print(f"learning_rate: {model.learning_rate}")
        # print(f"clip_range: {model.clip_range(1)}")

        # every run will have the same number of rollout-optimization loops
        # total_timesteps = wandb.config["n_steps"] * iterations
        total_timesteps = model.n_steps * iterations

        model.learn(
            total_timesteps=total_timesteps,
            callback=CustomWandbCallback(
                env=eval_env,
                wandb_run=run,
                n_evals=1,
                save_name="sensitivity_model",
            )
        )

    return


def configure_sweep(params: list):
    """
    Create the dictionary used to configure the sweep.\n
    :param params: list of parameter names to include in the sweep
    :return: configuration dictionary
    """

    all_params = {
        "rc0": {
            "distribution": "categorical",  # "uniform",
            "values": [10, 15, 20, 25, 30],  # magnitude of r_x [m]
            # "min": 10,
            # "max": 30,
        },
        "wt0": {
            "distribution": "categorical",
            "values": [radians(i) for i in [0, 2, 3, 4, 6]],  # magnitude of w_z [rad/s]
        },
        "koz_radius": {
            "distribution": "categorical",  # "uniform",
            "values": [2.5, 4, 5, 6, 7.5],
            # "min": 3,
            # "max": 7,
        },
        "corridor_half_angle": {
            "distribution": "categorical",  # "uniform",
            "values": [radians(i) for i in [15, 25, 30, 35, 45]],
            # "min": 15,
            # "max": 45,
        },
        "h": {  # orbit altitude [m]
            "distribution": "categorical",  # "uniform",
            "values": [400e3, 600e3, 800e3, 1000e3, 2000e3],
            # "min": 400e3,
            # "max": 2000e3,
        },
        "dt": {
            "distribution": "categorical",
            "values": [0.25, 0.5, 1, 2, 4],
        },
    }

    active_params = {}
    for i in params:
        active_params[i] = all_params.get(i)

    sweep_config = {
        "name": "sensitivity_sweep" + "_" + "-".join(params),    # name of the sweep (not the project)
        "metric": {                     # metric to optimize, has to be logged with `wandb.log()`
            "name": "best_rew",
            "goal": "maximize",
        },
        "method": "grid",     # search method ("grid", "random", or "bayes")
        "parameters": active_params  # parameters to sweep through
    }
    return sweep_config


if __name__ == "__main__":

    # Get arguments:
    arguments = get_tune_args()
    arguments.iterations = 489  # 1M timesteps  # 50
    project_name = arguments.project
    if project_name == "":
        project_name = "sensitivity"

    # Set-up the sweep:
    wandb.login(key="e9d6f3f54d82d87f667aa6b5681dd5810d8a8663")

    # Select the parameters to include in the sweep:
    parameter_names = []  # "rc0", "wt0", "koz_radius", "corridor_half_angle", "h", "dt"
    sweep_configuration = configure_sweep(parameter_names)
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)

    # Run the sweep:
    wandb.agent(sweep_id, function=partial(train_function, arguments.iterations))

    # # To continue an unfinished sweep: use the old sweep ID and project name (can be found on the W&B platform)
    # old_sweep_id = ""
    # old_project_name = ""
    # wandb.agent(old_sweep_id, project=old_project_name, function=partial(train_function, arguments.iterations))