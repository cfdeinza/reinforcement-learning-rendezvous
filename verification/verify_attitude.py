"""
Verify my implementation of the attitude dynamics.
In order to do this, I created a simle class `RigidBody`
"""
import numpy as np
from scipy.integrate import solve_ivp
from utils.general import dydt


class RigidBody:

    def __init__(self, q0: np.ndarray, w0: np.ndarray):
        """
        Initialize an instance of the body
        :param q0: initial attitude of the body
        :param w0: initial rot rate of the body (expressed in the body frame) [rad/s]
        """
        self.q = None  # Current attitude of the body (initialized on reset() call)
        self.w = None  # Current rot rate of the body (initialized on reset() call)
        self.inertia = 0  # Moment of inertia of the body [kg.m2]
        self.inv_inertia = 0  # Inverse of the moment of inertia of the body
        self.q0 = q0  # Initial attitude of the body
        self.w0 = w0  # Initial rot rate of the body [rad/s]
        return

    def reset(self):
        """
        Reset the attitude and rotation rate of the body to their original values.
        :return: None
        """
        self.q = self.q0
        self.w = self.w0
        return

    def integrate(self, dt, torque):
        """
        Use a fourth-order Runge-Kutta integrator to update the attitude and rotation rate of the chaser.\n
        :param dt: Time interval to propagate over [s]
        :param torque: Torque applied at the start of this time step [N.m]
        :return: None
        """

        y0 = np.append(self.q, self.w)

        sol = solve_ivp(
            fun=dydt,
            t_span=(0, dt),
            y0=y0,
            method='RK45',
            t_eval=np.array([dt]),
            rtol=1e-7,
            atol=1e-6,
            args=(self.inertia, self.inv_inertia, torque)
        )

        yf = sol.y.flatten()
        self.q = yf[0:4]
        self.q = self.q / np.linalg.norm(self.q)
        self.w = yf[4:]

        return


def main():
    q0 = np.array([1, 0, 0, 0])     # Initial attitude of the body (relative to an inertial frame)
    w0 = np.array([0, 0, 0])        # Initial rot rate of the body (expressed in body frame) [rad/s]
    body = RigidBody(q0, w0)        # create an instance of the body
    body.reset()                    # reset the state of the body
    dt = 1                          # Time interval to propagate each time integrate() is called [s]
    torque = np.array([0, 0, 0])    # Torque applied on the body [N.m]
    body.integrate(dt, torque)

    return


if __name__ == "__main__":
    main()
    # TODO: find some reference data to compare it with.
