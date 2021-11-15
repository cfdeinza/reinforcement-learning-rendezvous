from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Union
import pickle

"""
This file contains custom callbacks. For more info on how to make custom callbacks see:
    - The documentation page: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
    - The source code: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py
"""


def my_eval_callback(locals_, globals_) -> None:
    """
    Callback passed to the evaluate_policy function in order to save the evaluated trajectories (actions and states).
    I use 't' to check when each trajectory is finished, and then save the data with pickle. I would prefer to use the
    'done' variable, but 'done' is True AFTER each trajectory is finished (and all the previous states and actions have
    been deleted). If I were to use 'done' I would have to save the states and actions on every step, instead of at the
    end.
    """
    # info = locals_['info']
    # print(locals_['i'])
    obj = locals_['env'].envs[0].env
    n = locals_['episode_counts'][0]  # The current evaluation episode
    if (obj.t == obj.t_max-1) and (n == 0):  # TODO:Use a different conditional. This won't work if t_max is not reached
        # data = {
        #     'state': obj.trajectory,
        #     'action': obj.actions,
        #     'target_radius': obj.target_radius,
        #     'cone_half_angle': obj.cone_half_angle
        # }
        data = vars(obj)
        name = f'logs/eval_trajectory_{n}.pickle'
        with open(name, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'Saved trajectory data: {name}')

    return


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
