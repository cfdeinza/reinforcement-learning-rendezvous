import wandb
import numpy as np
from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import MlpPolicy
# from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.nn import Tanh  # ReLU, Sigmoid, Tanh
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
    policy_kwargs = None
    # policy_kwargs = {
    #     "activation_fn": Tanh,          # Default: Tanh
    #     "lstm_hidden_size": 256,        # Default: 256
    #     "n_lstm_layers": 1,             # Default: 1
    #     "shared_lstm": False,           # Default: False
    #     "enable_critic_lstm": True,     # Default: True (must set to False if `shared_lstm` is True)
    # }
    model = PPO(  # RecurrentPPO or PPO
        policy,
        env,
        learning_rate=2e-3,     # Default: 3e-4 for MLP and RNN
        n_steps=2048,           # Default: 2048 for MLP, 128 for RNN
        batch_size=128,         # Default: 64 for MLP, 128 for RNN
        n_epochs=40,            # Default: 10 for MLP and RNN
        clip_range=0.25,        # Default: 0.2 for MLP and RNN
        gamma=0.99,             # Default: 0.99 for MLP and RNN
        policy_kwargs=policy_kwargs,
        seed=config.get("seed", 0),
        verbose=1,
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
        reward_kwargs = {
            "collision_coef": wandb.config["collision_coef"],
            "bonus_coef": wandb.config["bonus_coef"],
            "fuel_coef": wandb.config["fuel_coef"],
            "att_coef": wandb.config["att_coef"],
        }
        env_config = {"wt0": np.array([0, 0, np.radians(1.5)])}
        train_env = make_env(reward_kwargs, quiet=True, config=env_config, stochastic=False)
        eval_env = copy_env(train_env)

        model = make_model(MlpPolicy, train_env, config=wandb.config)
        # model = make_model("MlpLstmPolicy", train_env, config=wandb.config)

        # every run will have the same number of rollout-optimization loops
        # total_timesteps = wandb.config["n_steps"] * iterations
        total_timesteps = model.n_steps * iterations

        model.learn(
            total_timesteps=total_timesteps,
            callback=CustomWandbCallback(
                env=eval_env,
                wandb_run=run,
                save_name="rew_tune_model",  # name of the file where the best model will be saved
                n_evals=1,
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
        "name": "reward_sweep",  # name of the sweep (not the project)
        "metric": {  # metric to optimize, has to be logged with `wandb.log()`
            "name": "best_rew",
            "goal": "maximize",
        },
        "method": "grid",  # search method ("grid", "random", or "bayes")
        "parameters": {  # parameters to sweep through
            # "gamma": {                  # discount factor
            #     "distribution": "categorical",
            #     "values": [0.8, 0.9, 0.99, 1]
            # },
            "collision_coef": {
                # "distribution": "uniform",
                # "min": 0,
                # "max": 1,
                "values": [0.5],  # [0, 0.1, 0.25, 0.5, 1]  # Default: 0.5
            },
            "bonus_coef": {
                # "distribution": "uniform",
                # "min": 0,
                # "max": 10,
                "values": [10],  # [0, 2, 4, 8], # Default: 10
            },
            "fuel_coef": {
                # "distribution": "uniform",
                # "min": 0,
                # "max": 1,
                "values": [0],  # [0, 0.05, 0.1, 0.25, 0.5, 1],  # Default: 0
            },
            "att_coef": {
                "values": [0, 1, 2, 4],  # [0, 0.05, 0.1, 0.25, 0.5],  # Default: 0.25
            },
            "seed": {
                "values": [0, 1, 2],
            },
        },
    }

    return sweep_config


if __name__ == "__main__":

    # Get arguments:
    arguments = get_tune_args()
    arguments.iterations = 100  # 293 for 600k steps  # 200 for 400k steps  # 486  # 250
    project_name = arguments.project
    if project_name == "":
        project_name = "rew_sweep"

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
