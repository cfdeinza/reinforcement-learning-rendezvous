import numpy as np
from utils.quaternions import quat2mat


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


def clohessy_wiltshire_solution(r0: np.ndarray, v0: np.ndarray, n: float, t: float) -> tuple:
    """
    Use the closed-for solution of the Clohessy-Wiltshire model to
    compute the position and velocity of the chaser after time t.\n
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


def state_derivative(_t, y, n, mc, ic, ic_inv, it, it_inv, force, torque):
    """
    Computes the derivative of the current state.\n
    :param _t: time (not used)
    :param y: state vector [r, v, qc, wc, qt, wt]
    :param n: orbit mean motion
    :param mc: chaser mass
    :param ic: chaser moment of inertia
    :param ic_inv: chaser moment of inertia inverse
    :param it: target moment of inertia
    :param it_inv: target moment of inertia inverse
    :param force: force acting on the chaser (expressed in C frame)
    :param torque: torque acting on the chaser (expressed in the C frame)
    :return: ndarray containing the derivative of the state
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


def derivative_of_att_and_rot_rate(_t, y, inertia, inv_inertia, torque):
    """
    Compute the derivative of the attitude and rotational rate of a rigid body.
    :param _t: time (not used)
    :param y: "state" vector [q, w]
    :param inertia: moment of inertia of the rigid body
    :param inv_inertia: inverse of the moment of inertia
    :param torque: torque acting on the body
    :return: derivative of the attitude quaternion and the rotational rate
    """

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
