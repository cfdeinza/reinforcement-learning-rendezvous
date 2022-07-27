"""
Functions related to quaternions. Always use the scalar-first format.
"""

import numpy as np
# import time


def rot2quat(axis: np.ndarray, theta) -> np.ndarray:
    """
    Create a quaternion (in scalar-first format) using a given rotation axis and rotation angle (in radians).\n
    :param axis: one-dimensional numpy array representing the rotation axis.
    :param theta: rotation angle [rad].
    :return: quaternion
    """

    assert axis.shape == (3,)
    assert np.linalg.norm(axis) > 0

    axis = axis / np.linalg.norm(axis)
    q = np.append(np.cos(theta/2), axis * np.sin(theta/2))

    assert quat_magnitude(q) == 1, f'|q| = {quat_magnitude(q)}'

    return q


def quat2rot(q: np.ndarray) -> tuple:
    """
    Compute the rotation axis and rotation angle from a given quaternion.\n
    :param q: quaternion
    :return: Tuple containing the rotation axis and rotation angle (in radians)
    """

    assert q.shape == (4,)
    # assert quat_magnitude(q) == 1, f'|q| = {quat_magnitude(q)}'
    q = quat_normalize(q)

    theta = 2 * np.arccos(q[0])

    euler = None if theta == 0 else q[1:] / np.sin(theta/2)

    return euler, theta


def convert_to_scalar_first(q: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion from vector-first to scalar-first form. (scipy.spatial.transform.Rotation uses vector-first)\n
    :param q: vector-first quaternion
    :return:
    """

    assert q.shape == (4,)
    assert quat_magnitude(q) == 1, f'|q| = {quat_magnitude(q)}'

    return np.append(q[-1], q[0:3])


def quat_conjugate(q) -> np.ndarray:
    """
    Compute the conjugate (q*) of a quaternion.\n
    :param q: quaternion
    :return: quaternion conjugate
    """

    assert q.shape == (4,)
    # assert quat_magnitude(q) == 1, f'|q| = {quat_magnitude(q)}'
    q = quat_normalize(q)

    return np.append(q[0], -q[1:])


def quat_magnitude(q: np.ndarray):
    """
    Compute the magnitude of a quaternion.\n
    :param q: quaternion
    :return: magnitude of the quaternion
    """

    assert q.shape == (4,)

    return np.linalg.norm(q)  #((q ** 2).sum()) ** 0.5


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """
    Normalize a quaternion.\n
    :param q: quaternion
    :return: normalized quaternion.
    """

    assert q.shape == (4,)

    return q / quat_magnitude(q)


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a quaternion.\n
    :param q: quaternion
    :return: inverse of the quaternion
    """

    assert q.shape == (4,)
    # assert quat_magnitude(q) == 1, f'|q| = {quat_magnitude(q)}'
    q = quat_normalize(q)

    q_inv = quat_conjugate(q) / (quat_magnitude(q)**2)

    return q_inv  # / quat_magnitude(q_inv)


def quat_product(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Compute the product of two quaternions. Note that order matters! (q1 x q2 =/= q2 x q1)\n
    :param q1: first quaternion
    :param q2: second quaternion
    :return: product
    """
    assert q1.shape == (4,)
    assert q2.shape == (4,)
    # assert quat_magnitude(q1) == 1, f'|q1| = {quat_magnitude(q1)}'
    # assert quat_magnitude(q2) == 1, f'|q2| = {quat_magnitude(q2)}'
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)

    s1, v1 = q1[0], q1[1:]
    s2, v2 = q2[0], q2[1:]

    scalar_product = s1 * s2 - np.dot(v1, v2)
    vector_product = s1 * v2 + s2 * v1 + np.cross(v1, v2)

    q_product = np.append(scalar_product, vector_product)

    return q_product  # / quat_magnitude(q_product)


def quat_error(q_est: np.ndarray, q_true: np.ndarray) -> np.ndarray:
    """
    Compute the quaternion error. (the rotation from the estimated axes to the true axes).\n
    :param q_est: quaternion for estimated axes
    :param q_true: quaternion for true axes
    :return: quaternion error
    """

    assert q_est.shape == (4,)
    assert q_true.shape == (4,)
    # assert quat_magnitude(q_est) == 1, f'|q| = {quat_magnitude(q_est)}'
    # assert quat_magnitude(q_true) == 1, f'|q| = {quat_magnitude(q_true)}'
    q_est = quat_normalize(q_est)
    q_true = quat_normalize(q_true)

    qd = quat_product(q_true, quat_inverse(q_est))

    return qd / quat_magnitude(qd)


if __name__ == "__main__":
    # Tests:
    # quat = np.array([1, 2, 3, 4])
    quat = np.array([0, 0, 0, 1])
    print(f'scalar-first: {convert_to_scalar_first(quat)}')
    print(f'conjugate: {quat_conjugate(quat)}')

    # Magnitude:
    # start = time.perf_counter()
    # mag_np = np.linalg.norm(quat)
    # print(mag_np, time.perf_counter() - start)
    # start = time.perf_counter()
    # mag_custom = quat_magnitude(quat)
    # print(mag_custom, time.perf_counter() - start)

    # Inverse:
    inv = quat_inverse(quat)
    print(f'Inverse: {inv}')
    print(f'q x q^-1: {quat_product(quat, inv)}')

    # Quaternion error:
    axis1 = np.array([0, 0, 1])
    theta1 = np.radians(0)
    theta2 = np.radians(36)
    q_1 = rot2quat(axis1, theta1)
    q_2 = rot2quat(axis1, theta2)
    error = quat_error(q_1, q_2)
    print(f'quat error: {error}')
    rot = quat2rot(error)
    print(f'error2rot: axis = {rot[0]},  theta = {np.degrees(rot[1])} deg')
