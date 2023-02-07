import numpy as np
from utils.environment_utils import make_env
import matplotlib.pyplot as plt

"""
Verified specific trajectories (free drift and continuous thrust).
Results: free drift behaves exactly as expected. Continuous torque shows a slight error because we assume that all of
the thrust is executed instantaneously at the start of the time step, instead of throughout the time step. 
The effect is small, so it is acceptable because it helps to reduce computational time.
"""


def main():

    # Straight-line approach from -y using constant continuous thrust to maintain a constant velocity.
    mu = 3.986004418e14  # m3/s2
    ro = 6371e3 + 800e3
    n = np.sqrt(mu / ro ** 3)
    print(f"n = {n} rad/s")
    # f = 0.5  # N
    # vy0 = f/(2*n*100)  # velocity to keep a straight-line when f is applied
    vy0 = 2
    f = 2*n*100*vy0  # thrust required to maintain constant velocity
    print(f"Thrust force: {f} N")
    print(f"Maintain constant vy velocity: {vy0} m/s")
    action = np.array([-f/10, 0, 0, 0, 0, 0])
    config = dict(
        rc0=np.array([0, -120, 0]),
        vc0=np.array([0, vy0, 0]),
        dt=0.5,
    )

    # Free drift in a co-planar lower orbit: with the right vy0, it results in linear motion along y.
    # dx0 = 10
    # mu = 3.986004418e14  # m3/s2
    # ro = 6371e3 + 800e3
    # n = np.sqrt(mu / ro ** 3)
    # action = np.array([0, 0, 0, 0, 0, 0])
    # config = dict(
    #     rc0=np.array([-dx0, 0, 0]),
    #     vc0=np.array([0, 3/2*n*dx0, 0]),
    #     dt=10,
    #     koz_radius=3,
    # )
    stochastic = False
    env = make_env(reward_kwargs=None, config=config, stochastic=stochastic)
    env2 = make_env(reward_kwargs=None, config=config, stochastic=stochastic)  # used to show the free drift trajectory
    _ = env.reset()
    _ = env2.reset()
    rs = [env.rc]
    vs = [env.vc]
    ts = [env.t]
    r2 = [env2.rc]

    # orbits = 1
    t_max = 60  # 2*np.pi/env.n * orbits
    while env.t < t_max:
        env.step(action)
        rs.append(env.rc)
        vs.append(env.vc)
        ts.append(env.t)
        env2.step(np.zeros((6,)))
        r2.append(env2.rc)

    rs = np.vstack(rs)
    vs = np.vstack(vs)
    ts = np.hstack(ts)  # / t_max
    r2 = np.vstack(r2)

    print(f"Final pos: {rs[-1]}, magnitude = {np.linalg.norm(rs[-1])} m")
    print(f"Final vel: {vs[-1]}, magnitude = {np.linalg.norm(vs[-1])} m/s")
    print(f"Expected:  {vy0}")

    r_lim = round(np.max(np.abs(rs)) + 1)
    v_lim = round(np.max(np.abs(vs)) + 0.5)
    rows, cols = 2, 3
    fig, ax = plt.subplots(rows, cols, figsize=(10, 5))
    ax[0, 0].plot(ts, rs[:, 0])
    ax[0, 1].plot(ts, rs[:, 1])
    ax[0, 2].plot(ts, rs[:, 2])
    ax[1, 0].plot(ts, vs[:, 0])
    ax[1, 1].plot(ts, vs[:, 1])
    ax[1, 2].plot(ts, vs[:, 2])
    ax[0, 0].set_ylabel(f"$r_x$ [m]")
    ax[0, 1].set_ylabel(f"$r_y$ [m]")
    ax[0, 2].set_ylabel(f"$r_z$ [m]")
    ax[1, 0].set_ylabel(f"$v_x$ [m/s]")
    ax[1, 1].set_ylabel(f"$v_y$ [m/s]")
    ax[1, 2].set_ylabel(f"$v_z$ [m/s]")

    for i in range(cols):
        ax[0, i].set_xlim([ts[0], ts[-1]])
        ax[1, i].set_xlim([ts[0], ts[-1]])
        ax[0, i].set_ylim([-r_lim, r_lim])
        ax[1, i].set_ylim([-v_lim, v_lim])
    for i in range(rows):
        for j in range(cols):
            ax[i, j].grid()
            ax[i, j].set_xlabel(f"Time [s]")
    plt.tight_layout()
    plt.show()
    plt.close()

    lw = 2
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(ts, rs[:, 0], label="$r_x$", lw=lw)
    ax[0].plot(ts, rs[:, 1], label="$r_y$", lw=lw)
    ax[0].plot(ts, rs[:, 2], label="$r_z$", lw=lw)
    ax[1].plot(ts, vs[:, 0], label="$v_x$", lw=lw)
    ax[1].plot(ts, vs[:, 1], label="$v_y$", lw=lw)
    ax[1].plot(ts, vs[:, 2], label="$v_z$", lw=lw)
    ax[0].set_ylabel(f"Chaser position [m]")
    ax[1].set_ylabel(f"Chaser velocity [m/s]")
    ax[0].set_ylim([-r_lim, r_lim])
    ax[1].set_ylim([-v_lim, v_lim])
    for a in ax:
        a.set_xlabel("Time [s]")
        a.set_xlim([ts[0], ts[-1]])
        a.grid()
        a.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(r2[:, 1], r2[:, 0], label="Free drift trajectory", lw=lw)
    ax.plot([-120, 0], [0, 0], "--.", label="Ideal cont. thrust trajectory", lw=lw)
    ax.plot(rs[:, 1], rs[:, 0], label="Experimental trajectory", lw=lw)
    ax.set_xlim([-120, 5])
    ax.set_ylim([-0.5, 2])
    ax.set_xlabel("$Y_{LVLH}$ [m]")
    ax.set_ylabel("$X_{LVLH}$ [m]")
    ax.grid()
    ax.legend()
    plt.show()
    plt.close()

    x = rs[:, 0]
    y = rs[:, 1]
    z = rs[:, 2]

    plt.rcParams.update({'font.size': 12})
    _ = plt.figure(num='3D', clear=True, figsize=(10, 5))
    ax3 = plt.axes(projection="3d")
    ax3.plot3D(x, y, z, label=f"Trajectory after {round(t_max)} s")  # , mec="k", ms=10, alpha=0.8)
    # ax3.plot3D(sol_x, sol_y, sol_z, "--", label="Analytical")
    ax3.plot3D(x[0], y[0], z[0], "kx", label="Chaser start")
    ax3.plot3D(x[-1], y[-1], z[-1], "g*", label="Chaser end")
    ax3.plot3D([0], [0], [0], "r.", label="Target")
    ax3.set_box_aspect((1, 1, 1))
    lim = r_lim
    # ax3.set_xlim([-20, 20])
    # ax3.set_ylim([-5, 35])
    # ax3.set_zlim([-20, 20])
    ax3.set_xlim([-lim, lim])
    ax3.set_ylim([-lim, lim])
    ax3.set_zlim([-lim, lim])
    ax3.set_zlabel("$Z_{LVLH}$ (m)")
    plt.xlabel('$X_{LVLH}$ (m)'), plt.ylabel('$Y_{LVLH}$ (m)')
    ax3.set_title("Chaser position")
    ax3.legend()
    plt.show()
    plt.close()
    return


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 16})
    main()
