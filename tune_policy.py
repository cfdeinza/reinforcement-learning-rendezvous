import wandb
from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import MlpPolicy
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
        # "net_arch": config["net_arch"],
        "net_arch": [dict(vf=net, pi=net)],
        "activation_fn": activations[config["activation_fn"]],
    }
    model = PPO(
        policy,
        env,
        # learning_rate=config["learning_rate"],
        # n_steps=config["n_steps"],
        # batch_size=config["batch_size"],
        # n_epochs=config["n_epochs"],
        # clip_range=config["clip_range"],
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        # IMPORTANT: remember to include as arguments every hyperparameter that is part of the sweep.
        seed=0,
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

        model = make_model(MlpPolicy, RendezvousEnv(quiet=True), config=wandb.config)
        # Check that the hyperparameters have been updated:
        # print(f"learning_rate: {model.learning_rate}")
        # print(f"clip_range: {model.clip_range(1)}")

        # every run will have the same number of rollout-optimization loops
        # total_timesteps = wandb.config["n_steps"] * iterations
        total_timesteps = model.n_steps * iterations

        model.learn(total_timesteps=total_timesteps, callback=CustomWandbCallback(RendezvousEnv, wandb_run=run))

    return


# hyperparameter_defaults = {
#     "learning_rate": 3e-4,
#     "clip_range": 0.2,
# }


if __name__ == "__main__":

    arguments = get_tune_args()
    arguments.iterations = 100
    project_name = arguments.project
    if project_name == "":
        project_name = "net_sweep"

    # Set-up the sweep:
    wandb.login(key="e9d6f3f54d82d87f667aa6b5681dd5810d8a8663")
    sweep_configuration = {
        "name": "test_sweep",           # name of the sweep (not the project)
        "metric": {                     # metric to optimize, has to be logged with `wandb.log()`
            "name": "best_rew",
            "goal": "maximize",
        },
        "method": "grid",               # search method ("grid", "random", or "bayes")
        "parameters": {                 # parameters to sweep through
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
                "values": [2, 3, 4],
            },
            "n_neurons": {
                "values": [16, 32, 64],
            },
            "activation_fn": {
                "values": ["ReLU", "Sigmoid", "Tanh"],
            }
        },
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)

    # Run the sweep:
    wandb.agent(sweep_id, function=partial(train_function, arguments.iterations))

    # To continue an unfinished run: use the old sweep ID and project name.
    # old_sweep_id = ""  # look at the the ID of the sweep on W&B
    # wandb.agent(old_sweep_id, project=project_name, function=partial(train_function, arguments.iterations))
