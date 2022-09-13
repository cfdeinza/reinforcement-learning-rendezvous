"""
This file contains custom callbacks. For more info on how to make custom callbacks see:
    - The documentation page: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
    - The source code: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py

Written by C. F. De Inza Niemeijer.
"""

from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
import os
# import warnings
# from typing import Any, Dict, Optional  # , Callable, List, Union
import pickle
import wandb
import numpy as np
from rendezvous_env import RendezvousEnv


class CustomWandbCallback(BaseCallback):
    """
    Custom callback to track the experiment using Weights & Biases (W&B).
    Upon training start, the callback initiates a W&B run, and logs some of the model's hyperparameters.
    At the start of every rollout, the callback evaluates an episode using the current policy, and it
    saves the model if the current reward exceeds the previous highest reward.
    """

    def __init__(self, verbose=0, prev_best_rew=None, wandb_run=None):
        """
        Instantiate callback object.
        Attributes inherited from BaseCallback:
        - self.model: the RL model
        - self.training_env: the environment used for training
        - self.n_calls: number of times the callback was called
        - self.num_timesteps: number of times that step() has been called?
        - self.locals: local variables
        - self.globals: global variables
        - self.logger: sb3 common logger object (used to report things in the terminal)
        - self.parent: parent object (BaseCallback)\n
        :param verbose: 0 for no output, 1 for info, 2 for debug
        :param prev_best_rew: previous best reward
        """
        super(CustomWandbCallback, self).__init__(verbose)

        self.prev_best_rew = prev_best_rew
        self.best_model_save_path = os.path.join("models", "best_model")
        self.wandb_run = wandb_run

        # Attribute to track if the W&B run was initiated externally or by the callback:
        if wandb_run is not None:
            self.wandb_external_start = True
        else:
            self.wandb_external_start = False

        pass

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        It initializes a W&B run if no run has yet been initiated.\n
        :return: None
        """

        if self.wandb_run is None:
            # Save the model hyperparameters:
            wandb_config = {
                "policy": self.model.policy,
                "learning_rate": self.model.learning_rate,
                "n_steps": self.model.n_steps,
                "batch_size": self.model.batch_size,
                "n_epochs": self.model.n_epochs,
                "gamma": self.model.gamma,
                "gae_lambda": self.model.gae_lambda,
                "clip_range": self.model.clip_range(1),
                "clip_range_vf": self.model.clip_range_vf,
                "normalize_advantage": self.model.normalize_advantage,
                "ent_coef": self.model.ent_coefficient,
                "vf_coef": self.model.vf_coef,
                "max_grad_norm": self.model.max_grad_norm,
                "target_kl": self.model.target_kl,
                "seed": self.model.seed,
            }

            # Initialize W&B run:
            self.wandb_run = wandb.init(
                project=None,
                entity=None,
                config=wandb_config,
                job_type="Train",
                name=None,
                mode="online",
                id=None,
            )

        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interactions using the current policy.
        This event is triggered before collecting new samples.
        It evaluates an episode of the environment using the current policy.
        The model is saved if the reward is better than the previous best.\n
        :return: None
        """

        # Evaluate the current policy, and save the total reward achieved during the episode.
        result = self.evaluate_policy()

        # Log results to W&B:
        result["total_timesteps"] = self.num_timesteps
        # data = {
        #     "ep_rew": rew,                      # total reward achieved during the eval episode
        #     "ep_len": t_end,                    # duration of the eval episode
        #     "total_steps": self.num_timesteps,  # total training steps executed
        # }
        wandb.log(result)

        # Save the model if the reward is better than last time:
        self.check_and_save(result["ep_rew"])

        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to env.step()\n
        :return: boolean (if the callback returns False, training is aborted early)
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.\n
        :return: None
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the learn() method.
        Perform one last evaluation and then finish the W&B run.\n
        :return: None
        """

        # Evaluate the current policy, and save the total reward achieved during the episode.
        result = self.evaluate_policy()

        # Log results to W&B:
        result["total_timesteps"] = self.num_timesteps  # total training steps executed
        wandb.log(result)

        # Save the model if the reward is better than last time:
        self.check_and_save(result["ep_rew"])
        # self.model.save(os.path.join(wandb.run.dir, "last_model"))  # Save an extra copy of the model on W&B
        wandb.log({"max_rew": wandb.run.summary["max_rew"]})        # Log best reward for the sweep

        # Finish W&B run (only if it was started by the Callback):
        if not self.wandb_external_start:
            self.wandb_run.finish()

        pass

    def evaluate_policy(self):
        """
        Run one episode with the current model.
        :return: total reward, episode_duration
        """

        model = self.model
        env = RendezvousEnv()  # HACK: hard-coded environment
        # env = self.training_env.envs[0].env  # I'm worried this might affect the environment used for training
        obs = env.reset()
        total_reward = 0
        done = False

        while not done:

            # Get action from the model:
            action, state = model.predict(
                observation=obs,        # input to the policy network
                state=None,             # last hidden state (used for recurrent policies)
                episode_start=None,     # the last mask? (used for recurrent policies)
                deterministic=True      # whether or not to return deterministic actions (default is False)
            )

            # Step forward in time:
            obs, reward, done, info = env.step(action)

            # Add current reward to the total:
            total_reward += reward

        end_time = env.t
        final_dist = np.linalg.norm(env.rc)
        print(f'Evaluation complete. Total reward = {round(total_reward, 2)} at {end_time}')

        output = {
            "ep_rew": total_reward,  # Total reward achieved during the episode
            "ep_len": end_time,      # Length of the episode (seconds)
            "ep_dist": final_dist,   # Final distance of the chaser to the target (meters)
        }

        return output

    def check_and_save(self, rew) -> None:
        """
        Check if the current reward is better than the previous best. If so, save the current model.\n
        :param rew: reward obtained from the most recent evaluation.
        :return: None
        """

        if self.prev_best_rew is None or rew > self.prev_best_rew:

            # Update the best reward:
            self.prev_best_rew = rew

            # Update the summary of the W&B run:
            wandb.run.summary["max_rew"] = rew
            wandb.run.summary["timstep_of_max_rew"] = self.num_timesteps
            # wandb.run.summary["best_reward"] = f"{round(rew, 2)} at {self.num_timesteps} steps."

            # Save the current model:
            print(f'New best reward. Saving model on {self.best_model_save_path}')
            self.model.save(self.best_model_save_path)  # Save a local copy
            self.model.save(os.path.join(wandb.run.dir, "best_model"))  # Save an extra copy of the model on W&B
            # wandb.save(self.best_model_save_path + ".zip")  # save a copy on W&B (FAILS: PERMISSION REQUIRED)

        pass


class CustomCallback(BaseCallback):
    """
    Same as CustomWandbCallback, but without using Weights and Biases. Maybe it would be best to put them both together.
    Custom callback to save the model that yields the highest reward. Before every rollout, the callback evaluates the
    current policy over a single episode. If the total reward during the episode is higher than the previous best, the
    current model is saved.\n
    """

    def __init__(self, verbose=0, prev_best_rew=None):
        """
        Instantiate callback object.
        :param verbose: 0 for no output, 1 for info, 2 for debug
        :param prev_best_rew: previous best reward
        """
        super(CustomCallback, self).__init__(verbose)

        self.prev_best_rew = prev_best_rew
        self.best_model_save_dir = os.path.join(".", "models")  # Directory where the best model will be saved
        self.best_model_save_path = os.path.join("models", "best_model")

        pass

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts. It initializes the W&B run.\n
        :return: None
        """

        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interactions using the current policy.
        This event is triggered before collecting new samples.
        It evaluates an episode of the environment using the current policy.
        The model is saved if the reward is better than the previous best.\n
        :return: None
        """

        # Evaluate the current policy, and save the total reward achieved during the episode.
        rew, t_end = self.evaluate_policy()

        # Save the model if the reward is better than last time:
        self.check_and_save(rew)

        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to env.step()\n
        :return: boolean (if the callback returns False, training is aborted early)
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.\n
        :return: None
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the learn() method.
        Perform one last evaluation and then finish the W&B run.\n
        :return: None
        """

        # Evaluate the current policy, and save the total reward achieved during the episode.
        rew, t_end = self.evaluate_policy()

        # Save the model if the reward is better than last time:
        self.check_and_save(rew)

        pass

    def evaluate_policy(self):
        """
        Run one episode with the current model.
        :return: total reward, episode_duration
        """

        model = self.model
        env = RendezvousEnv()  # HACK: hard-coded environment
        # env = self.training_env.envs[0].env  # I'm worried this might affect the environment used for training
        obs = env.reset()
        total_reward = 0
        done = False

        while not done:

            # Get action from the model:
            action, state = model.predict(
                observation=obs,        # input to the policy network
                state=None,             # last hidden state (used for recurrent policies)
                episode_start=None,     # the last mask? (used for recurrent policies)
                deterministic=True      # whether or not to return deterministic actions (default is False)
            )

            # Step forward in time:
            obs, reward, done, info = env.step(action)

            # Add current reward to the total:
            total_reward += reward

        end_time = env.t
        print(f'Evaluation complete. Total reward = {round(total_reward, 2)} at {end_time}')

        return total_reward, end_time

    def check_and_save(self, rew) -> None:
        """
        Check if the current reward is better than the previous best. If so, save the current model.\n
        :param rew: reward obtained from the most recent evaluation.
        :return: None
        """

        if self.prev_best_rew is None or rew > self.prev_best_rew:

            # Update the best reward:
            self.prev_best_rew = rew

            # Save the current model:
            print(f'New best reward. Saving model on {self.best_model_save_path}')
            self.model.save(self.best_model_save_path)  # save a local copy

        pass


def my_eval_callback(locals_, globals_) -> None:
    """
    Callback passed to the evaluate_policy function in order to save the evaluated trajectories (actions and states).
    If multiple trajectories are evaluated, only the first one is saved.
    I use the 'done' variable to check when a trajectory is finished. Then I save  the whole instance of the environment
    as a dictionary using the pickle module.
    Note that 'self.trajectory' and 'self.actions' must be stored in 'info', otherwise they get deleted by the reset()
    function before we can save them.
    """
    obj = locals_['env'].envs[0].env
    n = locals_['episode_counts'][0]  # The current evaluation episode
    if (n == 0) & locals_['done']:
        data = vars(obj)
        data['trajectory'] = locals_['info']['trajectory']
        data['actions'] = locals_['info']['actions']
        name = f'logs/eval_trajectory_{n}.pickle'
        with open(name, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'Saved trajectory data: {name}')
    return
