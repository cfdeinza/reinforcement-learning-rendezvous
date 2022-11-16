import wandb
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
        n_steps=2048,                           # Default: 2048 for MLP, 128 for RNN
        batch_size=config["batch_size"],        # Default: 64 for MLP, 128 for RNN
        n_epochs=config["n_epochs"],            # Default: 10 for MLP and RNN
        clip_range=config["clip_range"],        # Default: 0.2 for MLP and RNN
        gamma=0.99,                             # Default: 0.99 for MLP and RNN
        policy_kwargs=policy_kwargs,
        # IMPORTANT: remember to include as arguments every hyperparameter that is part of the sweep.
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
        reward_kwargs = None
        env_config = {}
        stochastic = True
        train_env = make_env(reward_kwargs, quiet=True, config=env_config, stochastic=stochastic)
        eval_env = copy_env(train_env)

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
                save_name="alg_tune_model",  # name of the file where the best model will be saved
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
        "name": "algorithm_sweep",   # name of the sweep (not the project)
        "metric": {             # metric to optimize, has to be logged with `wandb.log()`
            "name": "best_rew",
            "goal": "maximize",
        },
        "method": "random",     # search method ("grid", "random", or "bayes")
        "parameters": {         # parameters to sweep through
            "learning_rate": {
                # "distribution": "uniform",
                "distribution": "log_uniform_values",
                "min": 1e-6,
                "max": 1e-1,
                # "values": [1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 1e-2]
                # "values": [1e-5, 1e-4],
            },
            # "n_steps": {
            #     "distribution": "categorical",  # uniform, categorical?
            #     # "min": 512,
            #     # "max": 8192,
            #     "values": [512, 1024, 2048, 4096],
            # },
            "batch_size": {
                "distribution": "categorical",
                # "min": 8,
                # "max": 512,
                "values": [32, 64, 128, 512],  # all factors of n_steps to avoid batch truncation
            },
            "n_epochs": {
                "distribution": "int_uniform",
                "min": 1,
                "max": 100,
                # "values": [1, 4, 8, 16, 32, 64, 100]
            },
            "clip_range": {
                "distribution": "categorical",
                # "min": 0.02,
                # "max": 0.8,
                "values": [0.02, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1, 2],
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
