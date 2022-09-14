import wandb
from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.nn import ReLU, Sigmoid, Tanh
from rendezvous_env import RendezvousEnv
from custom.custom_callbacks import CustomWandbCallback


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
        "net_arch": config["net_arch"],
        "activation_fn": config["activation_fn"],
    }
    model = PPO(
        policy,
        env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        clip_range=config["clip_range"],
        policy_kwargs=policy_kwargs,
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

        model.learn(total_timesteps=10_000, callback=CustomWandbCallback(RendezvousEnv, wandb_run=run))

    return


# hyperparameter_defaults = {
#     "learning_rate": 3e-4,
#     "clip_range": 0.2,
# }


if __name__ == "__main__":

    # Set-up the sweep:
    sweep_configuration = {
        "name": "test_sweep",           # name of the sweep (not the project)
        "metric": {                     # metric to optimize, has to be logged with `wandb.log()`
            "name": "max_rew",
            "goal": "maximize",
        },
        "method": "random",             # search method ("grid", "random", or "bayes")
        "parameters": {                 # parameters to sweep through
            "learning_rate": {"values": [3e-4]},
            "n_steps": {"values": [2048]},
            "batch_size": {"values": [64]},
            "n_epochs": {"values": [10]},
            "clip_range": {"values": [0.2]},
            "net_arch": {
                "values": {
                    [[], [], []],
                }
            },
            "activation_fn": {
                "values": {
                    [ReLU, Sigmoid, Tanh]
                }
            }
        },
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="test_sweep1")

    # Run the sweep:
    wandb.agent(sweep_id, function=train_function)
