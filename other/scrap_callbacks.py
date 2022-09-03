"""
Old class for an EventCallback. Currently not used. Not sure how it works anymore.
"""

from stable_baselines3.common.callbacks import EventCallback
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
import os
import warnings


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
