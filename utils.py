"""
This file contains basic utility functions
"""
import numpy as np
from scipy.spatial.transform import Rotation as scipyRot


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


# a = np.array([1, 0, 0])
# b = np.array([-1, 0, 0])
# print(np.degrees(angle_between_vectors(a, b)))
# print(np.degrees(angle_between_vectors(b, a)))
# print(perpendicular_vector(a, b))
# print(perpendicular_vector(b, a))
