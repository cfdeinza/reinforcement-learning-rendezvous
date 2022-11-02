"""
A script to tune the policy (actor and critic networks) using a Weights&Biases sweep.
"""

import wandb
# from stable_baselines3.ppo import PPO
# from stable_baselines3.ppo.policies import MlpPolicy
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.nn import ReLU, Sigmoid, Tanh
from rendezvous_env import RendezvousEnv
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
    net = [config["n_neurons"]] * config["n_layers"]
    activations = {"ReLU": ReLU, "Sigmoid": Sigmoid, "Tanh": Tanh}
    policy_kwargs = {
        "net_arch": [dict(vf=net, pi=net)],
        "activation_fn": activations[config["activation_fn"]],  # Default: Tanh
        "lstm_hidden_size": config["lstm_hidden_size"],         # Default: 256
        "n_lstm_layers": config["n_lstm_layers"],               # Default: 1
        "shared_lstm": bool(config["shared_lstm"]),             # Default: False
        "enable_critic_lstm": not bool(config["shared_lstm"]),  # Default: True
    }
    model = RecurrentPPO(
        policy,
        env,
        # learning_rate=config["learning_rate"],
        # n_steps=config["n_steps"],
        # batch_size=config["batch_size"],
        # n_epochs=config["n_epochs"],
        # clip_range=config["clip_range"],
        learning_rate=3e-4,     # Default: 3e-4 for MLP and RNN
        n_steps=2048,           # Default: 2048 for MLP, 128 for RNN
        batch_size=128,         # Default: 64 for MLP, 128 for RNN
        n_epochs=10,            # Default: 10 for MLP and RNN
        clip_range=0.2,         # Default: 0.2 for MLP and RNN
        gamma=0.99,             # Default: 0.99 for MLP and RNN
        policy_kwargs=policy_kwargs,
        # IMPORTANT: remember to include as arguments every hyperparameter that is part of the sweep.
        seed=0,
        verbose=1,
    )

    return model


def make_env(reward_kwargs, quiet=True) -> RendezvousEnv:
    """
    Creates an instance of the Rendezvous environment.\n
    :param reward_kwargs: dictionary containing keyword arguments for the reward function
    :param quiet: `True` to supress printed outputs, `False` to print outputs
    :return: instance of the environment
    """

    env = RendezvousEnv(reward_kwargs=reward_kwargs, quiet=quiet)

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
        train_env = make_env(reward_kwargs, quiet=True)
        eval_env = make_env(reward_kwargs, quiet=False)
        if reward_kwargs is None:
            print("Note: reward_kwargs have not been defined. Using default values")

        model = make_model("MlpLstmPolicy", train_env, config=wandb.config)
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
                save_name="net_tune_model",  # name of the file where the best model will be saved
            )
        )

    return


def configure_sweep():
    """
    Create the dictionary used to configure the sweep.\n
    :return: configuration dictionary
    """

    sweep_config = {
        "name": "network_sweep",  # name of the sweep (not the project)
        "metric": {  # metric to optimize, has to be logged with `wandb.log()`
            "name": "best_rew",
            "goal": "maximize",
        },
        "method": "grid",  # search method ("grid", "random", or "bayes")
        "parameters": {  # parameters to sweep through
            # "learning_rate": {"values": [3e-4]},
            # "n_steps": {"values": [2048]},
            # "batch_size": {"values": [64]},
            # "n_epochs": {"values": [10]},
            # "clip_range": {"values": [0.2]},
            # "net_arch": {               # network architecture (see `ideas\ppo_implementation.md` for more information
            #     "values": [
            #         [dict(vf=[16, 16], pi=[16, 16])],
            #         [dict(vf=[64, 64], pi=[64, 64])],  # default
            #         [dict(vf=[128, 128], pi=[128, 128])],
            #         [dict(vf=[32, 32, 32], pi=[32, 32, 32])],
            #         [128, dict(vf=[64], pi=[64])]
            #     ],
            # },
            "n_layers": {
                "values": [2, 3, 4],            # Default is 2
            },
            "n_neurons": {
                "values": [16, 32, 64],         # Default is 64
            },
            "activation_fn": {
                "values": ["ReLU", "Sigmoid", "Tanh"],  # Default is Tanh
            },
            "lstm_hidden_size": {
                "values": [32, 64, 128, 256],   # Default is 256
            },
            "n_lstm_layers": {
                "values": [1, 2, 3],            # Default is 1
            },
            "shared_lstm": {
                "values": [0, 1]                # Default is False
            }
        },
    }

    return sweep_config


if __name__ == "__main__":

    arguments = get_tune_args()
    arguments.iterations = 250  # 100
    project_name = arguments.project
    if project_name == "":
        project_name = "net_sweep"

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
