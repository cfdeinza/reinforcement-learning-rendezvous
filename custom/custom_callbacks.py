"""
This file contains custom callbacks. For more info on how to make custom callbacks see:
    - The documentation page: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
    - The source code: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py

Written by C. F. De Inza Niemeijer.
"""

import os
from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
# import warnings
# from typing import Any, Dict, Optional  # , Callable, List, Union
# import pickle
import wandb
import numpy as np


class CustomWandbCallback(BaseCallback):
    """
    Custom callback to track the experiment using Weights & Biases (W&B).
    Upon training start, the callback initiates a W&B run, and logs some of the model's hyperparameters.
    At the start of every rollout, the callback evaluates an episode using the current policy, and it
    saves the model if the current reward exceeds the previous highest reward.
    """

    def __init__(
            self,
            env,
            wandb_run=None,
            prev_best_rew=None,
            save_name: str=None,
            n_evals: int=1,
            project: str=None,
            run_id: str=None,
            verbose: int=0,
    ):
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
        :param env: environment to use for evaluations
        :param wandb_run: current Weights & Biases run
        :param verbose: 0 for no output, 1 for info, 2 for debug
        :param prev_best_rew: previous best reward
        :param save_name: name of the zip file to save the best model in
        :param n_evals: number of episodes evaluated per rollout
        :param project: name of the project where the run will be saved
        :param run_id: id of the run to be resumed
        """
        super(CustomWandbCallback, self).__init__(verbose)

        self.env = env
        self.wandb_run = wandb_run
        self.prev_best_rew = prev_best_rew
        self.best_results = None  # Dictionary that records the best result of the current run.
        if save_name is None:
            save_name = "best_model"
        self.save_path = os.path.join("models", save_name)
        assert isinstance(n_evals, int), "The number of evaluations must be an integer"
        assert n_evals > 0, "The number of evaluations must be greater than zero."
        self.n_evals = n_evals
        self.project = project
        self.run_id = run_id

        # Attribute to track if the W&B run was initiated externally or by the callback:
        if wandb_run is not None:
            self.wandb_external_start = True
            self.project = None  # Project is overridden if the run was started externally (e.g. by the tuning script)
            self.run_id = None  # Do NOT try to resume a run if it was started externally (e.g. by the tuning script)
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
                "ent_coef": self.model.ent_coef,
                "vf_coef": self.model.vf_coef,
                "max_grad_norm": self.model.max_grad_norm,
                "target_kl": self.model.target_kl,
                "seed": self.model.seed,
            }

            # Initialize W&B run:
            self.wandb_run = wandb.init(
                project=self.project,
                entity=None,
                config=wandb_config,
                job_type="Train",
                name=None,
                mode="online",  # Use "disabled" for testing
                id=self.run_id,
                resume="must" if self.run_id is not None else "allow",  # raises an error if the run_id does not exist
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
        # self.check_and_save(result["ep_rew"], result["ep_success"], result["ep_delta_v"])
        self.check_and_save(result)

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
        # self.check_and_save(result["ep_rew"], result["ep_success"], result["ep_delta_v"])
        self.check_and_save(result)
        # self.model.save(os.path.join(wandb.run.dir, "last_model"))  # Save an extra copy of the model on W&B

        # Log the metrics of the run with the highest reward:
        # wandb.log({
        #     "max_rew": wandb.run.summary["max_rew"],            # best reward achieved during the run
        #     "max_success": wandb.run.summary["max_success"],    # success of best reward
        #     "max_delta_v": wandb.run.summary["max_delta_v"],    # delta V of best reward
        # })
        wandb.log(self.best_results)

        # Finish W&B run (only if it was started by the Callback):
        if not self.wandb_external_start:
            self.wandb_run.finish()

        pass

    def evaluate_policy(self):
        """
        Run one episode with the current model.
        :return: dictionary with outputs
        """

        model = self.model

        # Make arrays to record results of each evaluation:
        ep_rews = np.zeros(shape=(self.n_evals,)) * np.nan  # reward achieved throughout the episode
        ep_end_times = np.zeros(shape=(self.n_evals,)) * np.nan  # time at which the episode ended
        ep_dists = np.zeros(shape=(self.n_evals,)) * np.nan  # minimum distance of the chaser to the origin
        ep_delta_vs = np.zeros(shape=(self.n_evals,)) * np.nan  # Delta V used throughout the episode
        ep_delta_ws = np.zeros(shape=(self.n_evals,)) * np.nan  # Delta omega used throughout the episode
        ep_successes = np.zeros(shape=(self.n_evals,)) * np.nan  # Num of timesteps where chaser achieved final conds.
        ep_collision_percentages = np.zeros(shape=(self.n_evals,)) * np.nan  # Percentage of steps where chaser collided
        ep_times_of_first_collision = np.zeros(shape=(self.n_evals,)) * np.nan  # Time of first collision
        ep_min_pos_errors = np.zeros(shape=(self.n_evals,)) * np.nan  # Minimum chaser position error (w/o colliding)
        ep_avg_att_errors = np.zeros(shape=(self.n_evals,)) * np.nan  # Average attitude error during the episode

        episodes_with_collisions = 0  # Number of episodes that had collisions in them
        episodes_with_successes = 0  # Number of episodes that had successes in them
        complete_trajectories = 0  # Number of episodes with the required amount of consecutive successes

        for i in range(self.n_evals):
            obs = self.env.reset()
            total_reward = 0
            sum_of_att_errors = self.env.get_attitude_error()
            chaser_in_koz = self.env.check_collision()
            if chaser_in_koz:
                collisions = 1  # keeps track of the amount of collisions that occur during the episode
                time_of_first_collision = self.env.t  # records the time at which the first collision occurs
                # successes = 0
                min_pos_error = np.nan  # records the minimum pos error achieved by the chaser without entering the KOZ
            else:
                collisions = 0
                time_of_first_collision = np.nan  # Previously used -1 (so that W&B can record it properly)
                # successes = int(self.env.check_success())
                rd_lvlh = self.env.target2lvlh(self.env.rd)  # Goal position (expressed in LVLH)
                min_pos_error = self.env.get_pos_error(rd_lvlh)
            lstm_states = None
            ep_start = np.ones(shape=(1,), dtype=bool)
            done = False

            while not done:

                # Get action from the model:
                action, lstm_states = model.predict(
                    observation=obs,         # input to the policy network
                    state=lstm_states,       # last hidden state (used for recurrent policies)
                    episode_start=ep_start,  # the last mask? (used for recurrent policies)
                    deterministic=True       # whether or not to return deterministic actions (default is False)
                )

                # Step forward in time:
                obs, reward, done, info = self.env.step(action)
                ep_start[0] = done

                # Add current reward to the total:
                total_reward += reward
                sum_of_att_errors += self.env.get_attitude_error()
                chaser_in_koz = self.env.check_collision()
                if chaser_in_koz:
                    collisions += 1
                    if np.isnan(time_of_first_collision):
                        time_of_first_collision = self.env.t
                else:
                    if np.isnan(time_of_first_collision):
                        # successes += int(self.env.check_success())
                        rd_lvlh = self.env.target2lvlh(self.env.rd)  # Goal position (expressed in LVLH)
                        min_pos_error = min(min_pos_error, self.env.get_pos_error(rd_lvlh))

            end_time = self.env.t
            steps = end_time / self.env.dt
            successes = self.env.success
            print(f'Evaluation complete. Total reward = {round(total_reward, 2)} at {end_time}')
            ep_rews[i] = total_reward
            ep_end_times[i] = end_time
            ep_dists[i] = np.linalg.norm(self.env.rc)
            ep_delta_vs[i] = self.env.total_delta_v
            ep_delta_ws[i] = self.env.total_delta_w
            ep_successes[i] = successes
            ep_collision_percentages[i] = collisions / steps * 100
            ep_times_of_first_collision[i] = time_of_first_collision
            ep_min_pos_errors[i] = min_pos_error
            ep_avg_att_errors[i] = sum_of_att_errors / (steps + 1)
            episodes_with_collisions += int(collisions > 0)
            episodes_with_successes += int(successes > 0)
            complete_trajectories += int(self.env.successful_trajectory)

        print(f"Average reward over {self.n_evals} episode(s): {ep_rews.mean()}")

        # Compute the average of the min position errors and the collision times:
        if np.all(np.isnan(ep_min_pos_errors)):
            avg_min_pos_error = -1
        else:
            avg_min_pos_error = np.nanmean(ep_min_pos_errors)
        if np.all(np.isnan(ep_times_of_first_collision)):
            avg_collision_time = -1
        else:
            avg_collision_time = np.nanmean(ep_times_of_first_collision)

        # Create ouput dictionary:
        output = {
            "ep_rew": ep_rews.mean(),           # Total reward achieved during the episode
            "ep_len": ep_end_times.mean(),      # Length of the episode (seconds)
            "ep_dist": ep_dists.mean(),         # Final distance of the chaser to the target (meters)
            "ep_delta_v": ep_delta_vs.mean(),   # Total delta V used during the episode
            "ep_delta_w": ep_delta_ws.mean(),   # Total delta omega used during the episode
            "ep_success": ep_successes.mean(),  # Whether the trajectory was successful (achieved capture w/o collision)
            "ep_collision_percentage": ep_collision_percentages.mean(),  # % of timesteps where chaser was inside KOZ
            "ep_time_of_first_collision": avg_collision_time,   # time when chaser first entered the KOZ
            "ep_min_pos_error": avg_min_pos_error,              # minimum pos error of the chaser without entering KOZ
            "ep_avg_att_error": ep_avg_att_errors.mean(),       # average att error of the chaser (regardless of KOZ)
            "%_collided_episodes": episodes_with_collisions/self.n_evals*100,  # % of episodes that had > 0 collisions
            "%_successfull_episodes": episodes_with_successes/self.n_evals*100,  # % of episodes that had > 0 successes
            "%_complete_trajectories": complete_trajectories/self.n_evals*100
        }

        return output

    def check_and_save(self, results) -> None:
        """
        Check if the current reward is better than the previous best. If so, save the current model.\n
        :param results: dictionary containing information about the episode
        :return: None
        """

        if self.prev_best_rew is None or results["ep_rew"] > self.prev_best_rew:

            # Update the best reward:
            self.prev_best_rew = results["ep_rew"]

            # Update the summary of the W&B run:
            self.best_results = {
                "best_rew": results["ep_rew"],
                "timestep_of_best_rew": self.num_timesteps,
                "best_success": results["ep_success"],
                "best_delta_v": results["ep_delta_v"],
                "best_ep_len": results["ep_len"],
                "best_collision_percentage": results["ep_collision_percentage"],
                "best_time_of_first_collision": results["ep_time_of_first_collision"],
                "best_min_pos_error": results["ep_min_pos_error"],
                "best_avg_att_error": results["ep_avg_att_error"],
            }

            # Save the current model:
            print(f'New best reward. Saving model on {self.save_path}')
            self.model.save(self.save_path)  # Save a local copy
            # Save an extra copy of the model on W&B:
            self.model.save(os.path.join(wandb.run.dir, os.path.split(self.save_path)[1]))
            # wandb.save(self.save_path + ".zip")  # save a copy on W&B (FAILS: PERMISSION REQUIRED)

        pass


class CustomCallback(BaseCallback):
    """
    Same as CustomWandbCallback, but without using Weights and Biases. Maybe it would be best to put them both together.
    Custom callback to save the model that yields the highest reward. Before every rollout, the callback evaluates the
    current policy over a single episode. If the total reward during the episode is higher than the previous best, the
    current model is saved.\n
    """

    def __init__(
            self,
            env,
            prev_best_rew=None,
            # reward_kwargs=None,  # dict with keyword arguments for the reward function
            save_name=None,
            n_evals=1,
            verbose=0,
    ):
        """
        Instantiate callback object.
        :param env: environment to use for evaluations
        :param verbose: 0 for no output, 1 for info, 2 for debug
        :param prev_best_rew: previous best reward
        :param save_name: name of the file where the best model will be saved
        :param n_evals: number of episodes evaluated
        """
        super(CustomCallback, self).__init__(verbose)

        self.env = env
        self.prev_best_rew = prev_best_rew
        self.best_model_save_dir = os.path.join(".", "models")  # Directory where the best model will be saved
        # self. reward_kwargs = {} if reward_kwargs is None else reward_kwargs
        if save_name is None:
            save_name = "best_model"
        self.save_path = os.path.join("models", save_name)
        self.n_evals = n_evals

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
        results = self.evaluate_policy()

        # Save the model if the reward is better than last time:
        self.check_and_save(results)

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
        results = self.evaluate_policy()

        # Save the model if the reward is better than last time:
        self.check_and_save(results)

        pass

    def evaluate_policy(self):
        """
        Run one episode with the current model.
        :return: dictionary with results
        """

        model = self.model
        # env = self.env(reward_kwargs=self.reward_kwargs)
        # env = self.training_env.envs[0].env  # I'm worried this might affect the environment used for training

        print("Evaluating...")
        ep_rews = np.zeros(shape=(self.n_evals,)) * np.nan
        ep_end_times = np.zeros(shape=(self.n_evals,)) * np.nan
        for i in range(self.n_evals):
            obs = self.env.reset()
            total_reward = 0
            lstm_states = None
            ep_start = np.ones(shape=(1,), dtype=bool)
            done = False

            while not done:

                # Get action from the model:
                action, lstm_states = model.predict(
                    observation=obs,         # input to the policy network
                    state=lstm_states,       # last hidden state (used for recurrent policies)
                    episode_start=ep_start,  # the last mask? (used for recurrent policies)
                    deterministic=True       # whether or not to return deterministic actions (default is False)
                )

                # Step forward in time:
                obs, reward, done, info = self.env.step(action)
                ep_start[0] = done

                # Add current reward to the total:
                total_reward += reward

            end_time = self.env.t
            print(f'Episode complete. Total reward = {round(total_reward, 2)} at {end_time}')
            ep_rews[i] = total_reward
            ep_end_times[i] = end_time
            print(f"Average reward over {self.n_evals} episode(s): {ep_rews.mean()}")

        output = {
            "ep_rew": ep_rews.mean(),
            "ep_len": ep_end_times.mean(),
        }

        return output

    def check_and_save(self, results) -> None:
        """
        Check if the current reward is better than the previous best. If so, save the current model.\n
        :param results: dictionary containing results from the most recent evaluation.
        :return: None
        """

        if self.prev_best_rew is None or results["ep_rew"] > self.prev_best_rew:

            # Update the best reward:
            self.prev_best_rew = results["ep_rew"]

            # Save the current model:
            print(f'New best reward. Saving model on {self.save_path}')
            self.model.save(self.save_path)  # save a local copy

        pass


# # Old callback:
# def my_eval_callback(locals_, globals_) -> None:
#     """
#     Callback passed to the evaluate_policy function in order to save the evaluated trajectories (actions and states).
#     If multiple trajectories are evaluated, only the first one is saved.
#     I use the 'done' variable to check when a trajectory is finished. Then I save the whole instance of the env as a
#     dictionary using the pickle module.
#     Note that 'self.trajectory' and 'self.actions' must be stored in 'info', otherwise they get deleted by the reset()
#     function before we can save them.
#     """
#     obj = locals_['env'].envs[0].env
#     n = locals_['episode_counts'][0]  # The current evaluation episode
#     if (n == 0) & locals_['done']:
#         data = vars(obj)
#         data['trajectory'] = locals_['info']['trajectory']
#         data['actions'] = locals_['info']['actions']
#         name = f'logs/eval_trajectory_{n}.pickle'
#         with open(name, 'wb') as handle:
#             pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
#             print(f'Saved trajectory data: {name}')
#     return
