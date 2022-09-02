"""
This file contains custom callbacks. For more info on how to make custom callbacks see:
    - The documentation page: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
    - The source code: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py

Written by C. F. De Inza Niemeijer.
"""

from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Union
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

    def __init__(self, verbose=0, prev_best_rew=None):
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
        self.best_model_save_dir = os.path.join(".", "models")  # Directory where the best model will be saved
        self.best_model_save_path = os.path.join("models", "best_model")
        self.wandb_run = None  # Cannot initialize wandb run yet, because self.model is still None at this point.

        pass

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts. It initializes the W&B run.\n
        :return: None
        """

        # Save the model hyperparameters:
        wandb_config = {
            "batch_size": self.model.batch_size,
            "gamma": self.model.gamma,
            "learning_rate": self.model.learning_rate,
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
        rew, t_end = self.evaluate_policy()

        # Log results to W&B:
        data = {
            "ep_rew": rew,                      # total reward achieved during the eval episode
            "ep_len": t_end,                    # duration of the eval episode
            "total_steps": self.num_timesteps,  # total training steps executed
        }
        wandb.log(data)

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

        # Log results to W&B:
        data = {
            "ep_rew": rew,  # total reward achieved during the eval episode
            "ep_len": t_end,  # duration of the eval episode
            "total_steps": self.num_timesteps,  # total training steps executed
        }
        wandb.log(data)

        # Save the model if the reward is better than last time:
        self.check_and_save(rew)

        # Finish W&B run:
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

            # Update the summary of the W&B run:
            wandb.run.summary["best_reward"] = f"{round(rew, 2)} at {self.num_timesteps} steps."

            # Save the current model:
            print(f'New best reward. Saving model on {self.best_model_save_path}')
            self.model.save(self.best_model_save_path)

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
    # info = locals_['info']
    # print(locals_['i'])
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
    # if (obj.t == obj.t_max-1) and (n == 0):
    #     # data = {
    #     #     'state': obj.trajectory,
    #     #     'action': obj.actions,
    #     #     'target_radius': obj.target_radius,
    #     #     'cone_half_angle': obj.cone_half_angle
    #     # }
    #     data = vars(obj)


class MyCustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0):
        super(MyCustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> None:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        print('we are in _on_step()')
        return

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


class MyEventCallback(EventCallback):
    """
    A custom callback that saves the trajectory (states and actions).
    """

    def __init__(self, eval_env, log_path, verbose=1):

        super(MyEventCallback, self).__init__(verbose=verbose)

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env

        # Trajectory will be saved in "trajectory.npz"
        if log_path is not None:
            log_path = os.path.join(log_path, "trajectory")
        self.log_path = log_path

    def _init_callback(self) -> None:

        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folder if needed:
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_step(self):

        info = locals()
        if info['done']:
            print('done')

        return True
