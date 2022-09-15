import wandb
from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
# from torch.nn import ReLU, Sigmoid, Tanh
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
    model = PPO(
        policy,
        env,
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

        reward_kwargs = {
            "collision_coef": wandb.config["collision_coef"],
            "bonus_coef": wandb.config["bonus_coeff"],
            "fuel_coef": wandb.config["fuel_coef"],
        }  # TODO: include these arguments in the environment

        model = make_model(MlpPolicy, RendezvousEnv(quiet=True), config=wandb.config)
        # Check that the hyperparameters have been updated:
        # print(f"learning_rate: {model.learning_rate}")
        # print(f"clip_range: {model.clip_range(1)}")

        total_timesteps = wandb.config["n_steps"] * 10  # every run will have 10 rollout-optimization loops

        model.learn(total_timesteps=total_timesteps, callback=CustomWandbCallback(RendezvousEnv, wandb_run=run))

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
            "collision_coef": {
                "distribution": "uniform",
                "min": 0,
                "max": 10,
            },
            "bonus_coef": {
                "distribution": "uniform",
                "min": 0,
                "max": 10,
            },
            "fuel_coef": {
                "distribution": "uniform",
                "min": 0,
                "max": 10,
            },
        },
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="test_sweep1")

    # Run the sweep:
    wandb.agent(sweep_id, function=train_function)
