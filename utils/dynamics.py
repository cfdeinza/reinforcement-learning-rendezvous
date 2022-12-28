import numpy as np
from utils.quaternions import quat2mat


def state_derivative(t, y, n, mc, ic, ic_inv, it, it_inv, force, torque):
    """
    Computes the derivative of the current state.\n
    :return: ndarray
    """

    rc = y[0:3]
    vc = y[3:6]
    qc = y[6:10]
    wc = y[10:13]
    qt = y[13:17]
    wt = y[17:]
    f_lvlh = np.matmul(quat2mat(qc), force)     # force acting on the chaser (expressed in the LVLH frame) [N]
    t_chaser = torque                           # torque acting on the chaser (expressed in the C body frame) [N.m]
    rc_dot = vc
    vc_dot = clohessy_wiltshire_eom(rc, vc, n, force=f_lvlh, m=mc)
    qc_dot = quat_derivative(qc, wc)
    wc_dot = angular_acceleration(ic, ic_inv, wc, torque=t_chaser)
    qt_dot = quat_derivative(qt, wt)
    wt_dot = angular_acceleration(it, it_inv, wt, torque=np.array([0, 0, 0]))

    dydt = np.concatenate((rc_dot, vc_dot, qc_dot, wc_dot, qt_dot, wt_dot))
    return dydt


def clohessy_wiltshire_eom(r, v, n, force: np.ndarray = np.array([0, 0, 0]), m: float = 1):
    """
    Implements the Clohessy-Wiltshire equations of motion. (Computes the second derivative of x, y, and z in the
    LVLH reference frame)\n
    :param r: 3D position array [m]
    :param v: 3D velocity array [m/s]
    :param n: Mean motion of the orbit [rad/s]
    :param force: Force acting on the chaser [N]
    :param m: Mass of the chaser [kg]
    :return: 3D array containing the second time-derivative of x, y, and z
    """

    x_ddot = 3 * n**2 * r[0] + 2 * n * v[1] + force[0] / m
    y_ddot = -2 * n * v[0] + force[1] / m
    z_ddot = -1 * n**2 * r[2] + force[2] / m

    return np.array([x_ddot, y_ddot, z_ddot])


def quat_derivative(q: np.ndarray, w: np.ndarray):
    """
    Compute the derivative of a quaternion using the definition on
    [AHRS](https://ahrs.readthedocs.io/en/latest/filters/angular.html?highlight=quaternion#quaternion-derivative).\n
    :param q: scalar-first quaternion
    :param w: angular velocity [rad/s]
    :return: derivative of the quaternion
    """

    assert q.shape == (4,)
    assert w.shape == (3,)

    q = q / np.linalg.norm(q)

    w1, w2, w3 = w
    skew = np.array([
        [0, -w1, -w2, -w3],
        [w1, 0, w3, -w2],
        [w2, -w3, 0, w1],
        [w3, w2, -w1, 0]
    ])

    q_dot = 0.5 * np.matmul(skew, q)

    return q_dot


def angular_acceleration(inertia: np.ndarray, inv_inertia: np.ndarray, w: np.ndarray, torque: np.ndarray):
    """
    Compute the angular acceleration of a rigid body using Euler's rotation equations.\n
    :param inertia: moment of inertia of the rigid body.
    :param inv_inertia: inverse of the moment of inertia.
    :param w: rotational rate of the rigid body.
    :param torque: torque acting on the rigid body. (expressed in the body reference frame)
    :return: angular acceleration vector
    """

    assert w.shape == (3,)
    assert torque.shape == (3,)

    angular_momentum = np.matmul(inertia, w)
    cross_product = np.cross(w, angular_momentum)
    w_dot = np.matmul(inv_inertia, torque - cross_product)

    assert w_dot.shape == (3,)

    return w_dot
