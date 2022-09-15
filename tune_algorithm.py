import wandb
from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from rendezvous_env import RendezvousEnv
from custom.custom_callbacks import CustomWandbCallback
from arguments import get_tune_args


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
    model = PPO(
        policy,
        env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        clip_range=config["clip_range"],
        # IMPORTANT: remember to include as arguments every hyperparameter that is part of the sweep.
        seed=0,
        verbose=1,
    )

    return model


def train_function():
    """
    Function that will be executed on each run of the sweep.
    :return: None
    """

    with wandb.init() as run:

        # print(f"Starting Weights & Biases run. ID: {run.id}")

        model = make_model(MlpPolicy, RendezvousEnv(quiet=True), config=wandb.config)
        # Check that the hyperparameters have been updated:
        # print(f"learning_rate: {model.learning_rate}")
        # print(f"clip_range: {model.clip_range(1)}")

        total_timesteps = wandb.config["n_steps"] * 10  # every run will have 10 rollout-optimization loops

        model.learn(total_timesteps=20_000, callback=CustomWandbCallback(RendezvousEnv, wandb_run=run))

    return


def configure_sweep():
    """
    Create the dictionary used to configure the sweep.\n
    :return: configuration dictionary
    """

    sweep_config = {
        "name": "test_sweep",   # name of the sweep (not the project)
        "metric": {             # metric to optimize, has to be logged with `wandb.log()`
            "name": "max_rew",
            "goal": "maximize",
        },
        "method": "random",     # search method ("grid", "random", or "bayes")
        "parameters": {         # parameters to sweep through
            "learning_rate": {
                "distribution": "uniform",
                "min": 1e-6,
                "max": 1e-2,
                # "values": [1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 1e-2]
                # "values": [1e-5, 1e-4],
            },
            "n_steps": {
                "distribution": "categorical",  # uniform, categorical?
                # "min": 512,
                # "max": 8320,
                "values": [512, 1024, 2048, 4096, 8192],
            },
            "batch_size": {
                "distribution": "categorical",
                # "min": 8,
                # "max": 512,
                "values": [8, 16, 32, 64, 128, 256, 512],  # all factors of n_steps to avoid batch truncation
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
                "values": [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 1, 2],
            },
        },
    }
    return sweep_config


if __name__ == "__main__":

    # Get arguments:
    arguments = get_tune_args()
    # project_name = arguments.project
    project_name = "scrap_sweep"

    # Set-up the sweep:
    sweep_configuration = configure_sweep()
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)

    # Run the sweep:
    wandb.agent(sweep_id, function=train_function)