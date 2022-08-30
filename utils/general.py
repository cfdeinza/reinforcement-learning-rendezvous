"""
This module contains general utility functions.

Written by C. F. De Inza Niemeijer.
"""

import sys
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as scipyRot
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


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
        print('Successful.')
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

    if path == '':
        model = None
        print('No model provided. Exiting')
        exit()
    else:
        print(f'Loading saved model "{path}"...')
        try:
            model = PPO.load(path, env=env)
            print('Successfully loaded model')
        except FileNotFoundError:
            print(f'No such file "{path}".\nExiting')
            exit()
    return model


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

    # Apply the dot product formula:
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

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


def clohessy_wiltshire(r0: np.ndarray, v0: np.ndarray, n: float, t: float) -> tuple:
    """
    Use the Clohessy-Wiltshire model to compute the position and velocity of the chaser after time t.\n
    :param r0: initial position of the chaser
    :param v0: initial velocity of the chaser
    :param n: mean motion of the orbit [s^-1]
    :param t: time interval [s]
    :return: tuple containing the new position and velocity of the chaser
    """
    assert r0.shape == (3,)
    assert v0.shape == (3,)

    x0 = np.append(r0, v0)
    nt = n * t

    stm = np.array([
        [4 - 3*np.cos(nt),      0, 0,             1 / n*np.sin(nt),        2 / n*(1 - np.cos(nt)),                   0],
        [6*(np.sin(nt) - nt),   1, 0,             -2 / n*(1 - np.cos(nt)), 1 / n*(4*np.sin(nt) - 3*nt),              0],
        [0,                     0, np.cos(nt),    0,                       0,                           1/n*np.sin(nt)],
        [3*n*np.sin(nt),        0, 0,             np.cos(nt),              2*np.sin(nt),                             0],
        [-6*n*(1 - np.cos(nt)), 0, 0,             -2*np.sin(nt),           4*np.cos(nt) - 3,                         0],
        [0,                     0, -n*np.sin(nt), 0,                       0,                               np.cos(nt)]
    ])

    xt = np.matmul(stm, x0)

    rt = xt[0:3]

    vt = xt[3:]

    return rt, vt


def angular_acceleration(inertia: np.ndarray, inv_inertia: np.ndarray, w: np.ndarray, torque: np.ndarray):
    """
    Compute the angular acceleration of a rigid body using Euler's rotation equations.\n
    :param inertia: moment of inertia of the rigid body.
    :param inv_inertia: inverse of the moment of inertia.
    :param w: rotational rate of the rigid body.
    :param torque: torque acting on the rigid body.
    :return: angular acceleration vector
    """

    assert w.shape == (3,)
    assert torque.shape == (3,)

    angular_momentum = np.matmul(inertia, w)
    cross_product = np.cross(w, angular_momentum)
    w_dot = np.matmul(inv_inertia, torque - cross_product)

    assert w_dot.shape == (3,)

    # print(f'Iw = {angular_momentum}')
    # print(f'w x Iw = {cross_product}')
    # print(f'w\' = {w_dot}')
    return w_dot


def quat_derivative(q, w):
    """
    Compute the derivative of a quaternion using the definition on
    [AHRS](https://ahrs.readthedocs.io/en/latest/filters/angular.html?highlight=quaternion#quaternion-derivative).
    :param q: scalar-first quaternion
    :param w: angular velocity [rad/s]
    :return: derivative of the quaternion
    """

    assert q.shape == (4,)
    # assert np.linalg.norm(q) == 1
    assert w.shape == (3,)

    q = q / np.linalg.norm(q)

    w1, w2, w3 = w
    skew = np.array([
        [0,  -w1, -w2, -w3],
        [w1,   0,  w3, -w2],
        [w2, -w3,   0,  w1],
        [w3,  w2, -w1,   0]
    ])

    q_dot = 0.5 * np.matmul(skew, q)

    return q_dot


def dydt(t, y, inertia, inv_inertia, torque):
    """
    Compute the derivative of the attitude and rotational rate of a rigid body.
    :param t: integration time
    :param y: "state" vector [q, w]
    :param inertia: moment of inertia of the rigid body
    :param inv_inertia: inverse of the moment of inertia
    :param torque: torque acting on the body
    :return: derivative of the attitude quaternion and the rotational rate
    """

    assert y.shape == (7,)

    q = y[0:4]
    w = y[4:]

    # assert np.linalg.norm(q) == 1, f'The magnitude of the quaternion changed during integration: {np.linalg.norm(q)}'
    q = q / np.linalg.norm(q)

    # if t != 0:
    #     torque = np.array([0, 0, 0])

    q_dot = quat_derivative(q, w)
    w_dot = angular_acceleration(inertia, inv_inertia, w, torque)

    dy = np.append(q_dot, w_dot)

    return dy


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


# Tests:
if __name__ == '__main__':
    print(normalize_value(np.array([20, 100, -100]), -100, 100))
    print(__doc__)

# i = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]])
# invi = np.linalg.inv(i)
# omega = np.array([1, 0, 1])
# m = np.array([0, 0, 0])
# res = angular_acceleration(i, invi, omega, m)
# print(res)


# a = np.array([1, 0, 0])
# b = np.array([-1, 0, 0])
# print(np.degrees(angle_between_vectors(a, b)))
# print(np.degrees(angle_between_vectors(b, a)))
# print(perpendicular_vector(a, b))
# print(perpendicular_vector(b, a))
