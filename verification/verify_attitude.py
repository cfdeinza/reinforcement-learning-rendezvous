"""
Verify my implementation of the attitude dynamics.
In order to do this, I created a simle class `RigidBody`
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from utils.dynamics import derivative_of_att_and_rot_rate
from utils.environment_utils import make_new_env
from utils.quaternions import quat_error, quat2mat


class RigidBody:

    def __init__(self, q0: np.ndarray, w0: np.ndarray):
        """
        Initialize an instance of the body
        :param q0: initial attitude of the body
        :param w0: initial rot rate of the body (expressed in the body frame) [rad/s]
        """
        self.q = None  # Current attitude of the body (initialized on reset() call)
        self.w = None  # Current rot rate of the body (initialized on reset() call)
        self.mc = 100
        self.lc = 1
        self.inertia = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 1 / 12 * self.mc * (2 * self.lc ** 2)
        self.inv_inertia = np.linalg.inv(self.inertia)
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
            fun=derivative_of_att_and_rot_rate,
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


def run_simulation(env):
    expected_timesteps = int(env.t_max / env.dt)
    t = np.full(shape=(expected_timesteps,), fill_value=np.nan)
    qc = np.full(shape=(expected_timesteps, 4), fill_value=np.nan)
    wc = np.full(shape=(expected_timesteps, 3), fill_value=np.nan)
    sol_qc = np.full(shape=(expected_timesteps, 4), fill_value=np.nan)
    sol_wc = np.full(shape=(expected_timesteps, 3), fill_value=np.nan)

    env.reset()
    body = RigidBody(q0=env.qc, w0=env.wc)  # create an instance of the body
    body.reset()  # reset the state of the body

    k = 0
    while env.t < env.t_max:
        t[k] = env.t
        qc[k, :] = env.qc
        wc[k, :] = env.wc
        sol_qc[k, :] = body.q
        sol_wc[k, :] = body.w
        if env.t < env.t_max/2:
            action = np.array([0, 0, 0, 0, 0, 0]) + 1
        else:
            action = np.array([0, 0, 0, 0, 0, 1]) + 1
        env.step(action)
        body_torque = (action[3:] - 1) * env.torque  # Torque applied on the body [N.m]
        body.integrate(env.dt, body_torque)
        k += 1

    # Errors:
    # print(f"Final quat error: {np.abs(qc[-1] - sol_qc[-1])}")
    q_error = quat_error(q_est=qc[-1], q_true=sol_qc[-1])
    theta = 2*np.arccos(q_error[0])
    print(f"Final attitude error: {np.degrees(theta)} deg")
    print(f"Or: {np.degrees(2*np.arccos(1-np.abs(qc[-1, 0] - sol_qc[-1, 0])))}")
    w_error = sol_wc[-1] - wc[-1]
    print(f"Final rot rate error: {np.degrees(np.linalg.norm(w_error))} deg/s")

    out = dict(
        t=t,
        qc=qc,
        wc=wc,
        sol_qc=sol_qc,
        sol_wc=sol_wc,
    )
    return out


def plot_results(data: dict) -> None:
    t = data.get("t")
    qc = data.get("qc")
    wc = data.get("wc")
    sol_qc = data.get("sol_qc")
    sol_wc = data.get("sol_wc")
    n_rows = 2
    n_cols = 4
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(10, 5))
    # Plot numerical:
    ax[0, 0].plot(t, qc[:, 0], label="numerical")
    ax[0, 1].plot(t, qc[:, 1], label="numerical")
    ax[0, 2].plot(t, qc[:, 2], label="numerical")
    ax[0, 3].plot(t, qc[:, 3], label="numerical")
    ax[1, 0].plot(t, wc[:, 0], label="numerical")
    ax[1, 1].plot(t, wc[:, 1], label="numerical")
    ax[1, 2].plot(t, wc[:, 2], label="numerical")
    # Plot analytical:
    ax[0, 0].plot(t, sol_qc[:, 0], "--", linewidth=1, label="analytical")
    ax[0, 1].plot(t, sol_qc[:, 1], "--", label="analytical")
    ax[0, 2].plot(t, sol_qc[:, 2], "--", label="analytical")
    ax[0, 3].plot(t, sol_qc[:, 3], "--", label="analytical")
    ax[1, 0].plot(t, sol_wc[:, 0], "--", label="analytical")
    ax[1, 1].plot(t, sol_wc[:, 1], "--", label="analytical")
    ax[1, 2].plot(t, sol_wc[:, 2], "--", label="analytical")

    max_q = 1.1
    max_w = max(1e-6, np.max(np.abs(wc))) * 1.1

    y_labels = [
        ["$q_w$", "$q_x$", "$q_y$", "$q_z$"],
        ["$\omega_x$ [rad/s]", "$\omega_y$ [rad/s]", "$\omega_z$ [rad/s]", ""],
    ]

    for row in range(n_rows):
        for col in range(n_cols):
            ax[row, col].grid()
            ax[row, col].legend()
            ax[row, col].set_xlim([t[0], t[-1]])
            # print(row, col)
            ax[row, col].set_ylabel(y_labels[row][col])
            if row == 0:
                lim = max_q
                ax[row, col].set_ylim([-lim, lim])
            else:
                lim = max_w
                ax[row, col].set_ylim([-lim, lim])

    plt.show()
    plt.close()

    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
    # Integration error:
    ax2.plot(t, np.abs(qc[:, 0] - sol_qc[:, 0]), label="$qc_w$ [-]")
    ax2.plot(t, np.abs(qc[:, 1] - sol_qc[:, 1]), label="$qc_x$ [-]")
    ax2.plot(t, np.abs(qc[:, 2] - sol_qc[:, 2]), label="$qc_y$ [-]")
    ax2.plot(t, np.abs(qc[:, 3] - sol_qc[:, 3]), label="$qc_z$ [-]")
    ax2.plot(t, np.abs(wc[:, 0] - sol_wc[:, 0]), label="$wc_x$ [rad/s]")
    ax2.plot(t, np.abs(wc[:, 1] - sol_wc[:, 1]), label="$wc_y$ [rad/s]")
    ax2.plot(t, np.abs(wc[:, 2] - sol_wc[:, 2]), label="$wc_z$ [rad/s]")
    ax2.set_xlabel("Time [s]")
    ax2.set_xlim([t[0], t[-1]])
    ax2.set_ylabel("Error")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.title("Integration errors")
    plt.show()
    plt.close()

    _ = plt.figure(num='3D', clear=True, figsize=(10, 5))
    ax3 = plt.axes(projection="3d")
    lvlh_len = 1
    ax3.plot3D([0, lvlh_len], [0, 0], [0, 0], "r-", label="$X_{LVLH}$")
    ax3.plot3D([0, 0], [0, lvlh_len], [0, 0], "g-", label="$Y_{LVLH}$")
    ax3.plot3D([0, 0], [0, 0], [0, lvlh_len], "b-", label="$Z_{LVLH}$")
    # ax3.plot3D(x, y, z, label=f"Trajectory in {t[-1]} s")  # , mec="k", ms=10, alpha=0.8)
    # ax3.plot3D(sol_x, sol_y, sol_z, "--", label="Analytical")
    # ax3.plot3D(x[0], y[0], z[0], "kx", label="Chaser start")
    # ax3.plot3D(x[-1], y[-1], z[-1], "g*", label="Chaser end")
    # ax3.plot3D([0], [0], [0], "r.", label="Target")
    ax3.set_box_aspect((1, 1, 1))
    lim = 2
    ax3.set_xlim([-lim, lim])
    ax3.set_ylim([-lim, lim])
    ax3.set_zlim([-lim, lim])
    ax3.set_zlabel("$Z_{LVLH}$ (m)")
    plt.xlabel('$X_{LVLH}$ (m)'), plt.ylabel('$Y_{LVLH}$ (m)')
    ax3.set_title("Rotational motion")
    ax3.legend()
    plt.show()
    plt.close()
    return


def main(env):
    results = run_simulation(env)
    plot_results(results)
    return


def chaser2lvlh(qc, vec):
    assert vec.shape == (3,), f"Vector must have a (3,) shape, but has {vec.shape}"
    return np.matmul(quat2mat(qc), vec)


if __name__ == "__main__":
    config = dict(
        rc0=np.array([0, -10, 0]),
        vc0=np.array([0, 0, 0]),
        qc0=np.array([1, 0, 0, 0]),
        wc0=np.radians(np.array([0, 0, 0])),
        qt0=np.array([1, 0, 0, 0]),
        wt0=np.array([0, 0, 0]),
        rc0_range=0,
        vc0_range=0,
        qc0_range=0,
        wc0_range=0,
        qt0_range=0,
        wt0_range=0,
        bt=1,
        dt=1,
        t_max=360*2.1,
        quiet=True,
    )
    eval_env = make_new_env(config)
    main(eval_env)
    # TODO: find some reference data to compare it with.
