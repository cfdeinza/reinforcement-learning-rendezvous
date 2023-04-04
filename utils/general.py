"""
This module contains general utility functions.

Written by C. F. De Inza Niemeijer.
"""

import sys
import pickle5 as pickle
import numpy as np
from scipy.spatial.transform import Rotation as scipyRot
from scipy.interpolate import interp1d
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import get_schedule_fn


def load_data(path: str):
    """
    Load data from a pickle file.\n
    :param path: path to the file.
    :return: data (usually a dictionary)
    """

    if len(path) == 0:
        print('Please specify the path to the trajectory file.\nExiting')
        sys.exit()

    print(f'Loading file "{path}"...', end=' ')

    try:
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
        print('Done')
    except FileNotFoundError:
        print(f'Error. Could not find file or directory "{path}"\nExiting')
        sys.exit()

    return data


def load_env(env_class):
    """
    Create an instance of the environment and wrap it in the required Gym wrappers.\n
    :return: Wrapped environment.
    """

    env = env_class()
    # env = Rendezvous3DOF()  # old 3DOF environment
    # env = Rendezvous3DOF(config=None)  # this was briefly used for ray rllib
    # env = gym.make('Pendulum-v1')  # simply using PendulumEnv() yields no `done` condition.

    # Wrap the environment in a Monitor and a DummyVecEnv wrappers:
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    return env


def load_model(path, env):
    """
    Load a saved model.\n
    :param path: path to the file containing the trained model.
    :param env: environment corresponding to the model.
    :return: model
    """

    model = None
    if path == '':
        print('No model provided. Exiting')
        exit()
    else:
        print(f'Loading saved model "{path}"...')
        try:
            custom_objects = {
                "lr_schedule": get_schedule_fn(3e-4),
                "learning_rate": get_schedule_fn(3e-4),
                "clip_range": get_schedule_fn(0.2),
            }
            if "rnn" in path:  # HACK: find a safer way to identify if the model to be loaded is Recurrent.
                print("RNN model detected...")
                model = RecurrentPPO.load(path, env=env, custom_objects=custom_objects)
            else:
                print("MLP model detected...")
                model = PPO.load(path, env=env, custom_objects=custom_objects)
            print('Successfully loaded model')
        except FileNotFoundError:
            print(f'No such file "{path}".\nExiting')
            exit()
    return model


def schedule_fn(initial_value: float, const=True):
    """
    Returns a function that computes the current value of a model parameter, based on the progress remaining.
    Progress remaining is a value that decreases from 1 to 0 as the training progresses.
    It is used for model parameters such as learning_rate and clip_range.\n
    :param initial_value: Initial value of the parameter.
    :param const: whether or not the value should remain constant (otherwise decrease linearly)
    :return: function that computes the current value of the parameter
    """

    def constant(progress_remaining: float) -> float:
        current_value = (progress_remaining * 0) + initial_value  # Constant
        return current_value

    def linear(progress_remaining: float) -> float:
        current_value = max(progress_remaining * initial_value, initial_value/10_000)  # Linear decrease
        return current_value

    def logarithmic(progress_remaining: float) -> float:
        current_value = initial_value * 10000**(progress_remaining - 1)
        return current_value

    if const is True:
        func = constant
    else:
        # func = linear
        func = logarithmic

    return func


def interp(data: np.ndarray, time: np.ndarray, new_time: np.ndarray, kind=None):
    """
    Linearly interpolate the data from a numpy array.\n
    :param data: array containing data to be interpolated.
    :param time: time corresponding to the data.
    :param new_time: new time to sample data from.
    :param kind: type of interpolation to perform (default is "linear")
    :return: interpolated data.
    """
    if kind is None:
        kind = "linear"

    f = interp1d(x=time, y=data, kind=kind)

    new_data = f(new_time)

    return new_data


def print_model(model):
    """
    Print the attributes of the model.\n
    :param model: object
    :return: None
    """

    print("\nModel:")
    dict_form = vars(model)
    for key, val in dict_form.items():
        if str(key)[0] != "_":
            if (key == "learning_rate" or key == "clip_range") and callable(val):
                print(f"{key}: [{val(1)} - {val(0)}]")
            else:
                print(f"{key}: {val}")

    return


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.float:
    """
    Computes the angle between two 3D vectors, using the geometric definition of the dot product:
    cos(theta) = (v1 . v2) / (|v1|*|v2|) \n
    :param v1: numpy array containing a 3d vector
    :param v2: numpy array containing a 3d vector
    :return: angle (in radians) between v1 and v2 (range: 0 to pi)
    """

    # Check the size of the input vectors:
    assert v1.shape == (3,)
    assert v2.shape == (3,)
    assert np.linalg.norm(v1) != 0
    assert np.linalg.norm(v2) != 0

    # Apply the dot product formula: (rounded to 5 decimal places to avoid occasional Runtime Warnings)
    angle = np.arccos(round(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), 5))

    return angle


def perpendicular_vector(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Computes the unit vector perpendicular to v1 and v2 using the cross product: v3 = v1 x v2.\n
    :param v1: numpy array containing a 3d vector
    :param v2: numpy array containing a 3d vector
    :return: 3d vector perpendicular to v1 and v2
    """

    # Check the size of the input vectors:
    assert v1.shape == (3,)
    assert v2.shape == (3,)
    assert np.linalg.norm(v1) != 0
    assert np.linalg.norm(v2) != 0

    # Apply the cross product:
    v3 = np.cross(v1, v2)

    # Normalize the vector:
    v3 = v3 / np.linalg.norm(v3)

    return v3


def rotate_vector_about_axis(vec, axis, theta):
    """
    Rotate a 3D vector by [theta] radians about a given axis, using the quaternion formulation.\n
    :param vec: 3D vector to be rotated.
    :param axis: 3D axis for vec to rotate around.
    :param theta: angle (in radians) to rotate the vector.
    :return: rotated vector
    """

    # Check the size of the input vectors:
    assert vec.shape == (3,)
    assert axis.shape == (3,)
    assert np.linalg.norm(vec) != 0
    assert np.linalg.norm(axis) != 0

    euler_axis = axis / np.linalg.norm(axis)
    quaternion = np.append(euler_axis * np.sin(theta / 2), np.cos(theta / 2))  # Rotation quaternion
    rotation = scipyRot.from_quat(quaternion)
    new_vec = rotation.apply(vec)

    return new_vec


def normalize_value(val, low, high, custom_range=None):
    """
    Normalize a value to [-1, 1], or to some custom range.\n
    :param val: value(s) to normalize (can be an ndarray)
    :param low: minimum bound for the un-normalized value
    :param high: maximum bound for the un-normalized value
    :param custom_range: some custom range [a, b] to normalize the value to
    :return: normalized value
    """
    if custom_range is None:
        custom_range = [-1, 1]

    a, b = custom_range
    norm_val = (b - a) * (val - low) / (high - low) + a

    return norm_val


def random_unit_vector() -> np.ndarray:
    """
    Generate a 3D unit vector pointed in a random direction (sampled from a uniform distribution).\n
    :return: numpy array of shape (3,)
    """
    vec = np.random.uniform(low=-1, high=1, size=(3,))
    return vec / np.linalg.norm(vec)


# Tests:
if __name__ == '__main__':
    print(normalize_value(np.array([20, 100, -100]), -100, 100))
    print(__doc__)
