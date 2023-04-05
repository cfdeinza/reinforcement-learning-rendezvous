"""
Script to tune the number of steps (n_steps).
"""

import wandb
import numpy as np
# from stable_baselines3.ppo import PPO
# from stable_baselines3.ppo.policies import MlpPolicy
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.nn import Tanh
# from rendezvous_env import RendezvousEnv
from utils.environment_utils import make_env, copy_env
from custom.custom_callbacks import CustomWandbCallback
from arguments import get_tune_args
from functools import partial


def make_model(policy, env, config):
    """
    Create a model using the hyperparameters given by the sweep.
    :param policy: policy for the model to use
    :param env: environment for the model to learn on
    :param config: configuration dictionary passed by the sweep
    :return: model
    """

    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    policy_kwargs = {
        "activation_fn": Tanh,          # Default: Tanh
        "lstm_hidden_size": 256,        # Default: 256
        "n_lstm_layers": 1,             # Default: 1
        "shared_lstm": False,           # Default: False
        "enable_critic_lstm": True,     # Default: True (must set to False if `shared_lstm` is True)
        }
    model = RecurrentPPO(
        policy,
        env,
        learning_rate=config["learning_rate"],  # Default: 3e-4 for MLP and RNN
        n_steps=config["n_steps"],  # Default: 2048 for MLP, 128 for RNN
        batch_size=config["batch_size"],  # Default: 64 for MLP, 128 for RNN
        n_epochs=config["n_epochs"],  # Default: 10 for MLP and RNN
        clip_range=config["clip_range"],  # Default: 0.2 for MLP and RNN
        gamma=0.99,  # Default: 0.99 for MLP and RNN
        policy_kwargs=policy_kwargs,
        # IMPORTANT: remember to include as arguments every hyperparameter that is part of the sweep.
        seed=config.get("seed", 0),
    )

    return model


def train_function(iterations):
    """
    Function that will be executed on each run of the sweep.
    :return: None
    """

    with wandb.init() as run:

        # print(f"Starting Weights & Biases run. ID: {run.id}")

        # Make environments:
        reward_kwargs = None
        env_config = {
            "wt0": np.array([0., 0., np.radians(1.5)]),
        }
        stochastic = False
        train_env = make_env(reward_kwargs, quiet=True, config=env_config, stochastic=stochastic)
        eval_env = copy_env(train_env)

        model = make_model("MlpLstmPolicy", train_env, config=wandb.config)

        total_timesteps = 500_000  # each run will train for this number of timesteps (different number of iters)

        model.learn(
            total_timesteps=total_timesteps,
            callback=CustomWandbCallback(
                env=eval_env,
                wandb_run=run,
                save_name="steps_tune_model",  # name of the file where the best model will be saved
                n_evals=1 if stochastic is False else 10,
                verbose=0,
            )
        )

    return


def configure_sweep():
    """
    Create the dictionary used to configure the sweep.\n
    :return: configuration dictionary
    """

    sweep_config = {
        "name": "steps_sweep",   # name of the sweep (not the project)
        "metric": {             # metric to optimize, has to be logged with `wandb.log()`
            "name": "best_rew",
            "goal": "maximize",
        },
        "method": "grid",     # search method ("grid", "random", or "bayes")
        "parameters": {         # parameters to sweep through
            "learning_rate": {
                "values": [2e-3],
            },
            "batch_size": {
                "values": [128],
            },
            "n_epochs": {
                "values": [40],
            },
            "clip_range": {
                "values": [0.25],
            },
            "n_steps": {
                "values": [128, 256, 512, 1024, 2048, 4096],
            },
            "seed": {
                "values": [0],
            },
        },
    }
    return sweep_config


if __name__ == "__main__":

    # Get arguments:
    arguments = get_tune_args()
    arguments.iterations = 50
    project_name = arguments.project
    if project_name == "":
        project_name = "alg_sweep"

    # Set-up the sweep:
    wandb.login(key="e9d6f3f54d82d87f667aa6b5681dd5810d8a8663")
    sweep_configuration = configure_sweep()
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)

    # Run the sweep:
    wandb.agent(sweep_id, function=partial(train_function, arguments.iterations))

    # # To continue an unfinished sweep: use the old sweep ID and project name (can be found on the W&B platform)
    # old_sweep_id = ""
    # old_project_name = ""
    # wandb.agent(old_sweep_id, project=old_project_name, function=partial(train_function, arguments.iterations))
