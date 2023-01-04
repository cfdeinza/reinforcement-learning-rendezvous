"""
Verify my implementation of the CW model.
"""
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from utils.dynamics import clohessy_wiltshire_solution
from utils.environment_utils import make_new_env
# from rendezvous_env import RendezvousEnv


def run_simulation(env):
    expected_timesteps = int(env.t_max / env.dt)
    t = np.full(shape=(expected_timesteps,), fill_value=np.nan)
    x = np.full(shape=(expected_timesteps,), fill_value=np.nan)
    y = np.full(shape=(expected_timesteps,), fill_value=np.nan)
    z = np.full(shape=(expected_timesteps,), fill_value=np.nan)
    u = np.full(shape=(expected_timesteps,), fill_value=np.nan)
    v = np.full(shape=(expected_timesteps,), fill_value=np.nan)
    w = np.full(shape=(expected_timesteps,), fill_value=np.nan)
    sol_x = np.full(shape=(expected_timesteps,), fill_value=np.nan)
    sol_y = np.full(shape=(expected_timesteps,), fill_value=np.nan)
    sol_z = np.full(shape=(expected_timesteps,), fill_value=np.nan)
    sol_u = np.full(shape=(expected_timesteps,), fill_value=np.nan)
    sol_v = np.full(shape=(expected_timesteps,), fill_value=np.nan)
    sol_w = np.full(shape=(expected_timesteps,), fill_value=np.nan)

    env.reset()

    k = 0
    pos_0, vel_0 = env.rc, env.vc
    new_pos, new_vel = pos_0, vel_0
    while env.t < env.t_max:
        t[k] = env.t
        x[k], y[k], z[k] = env.rc
        u[k], v[k], w[k] = env.vc
        action = np.ones(shape=(6,))
        env.step(action)
        sol_x[k], sol_y[k], sol_z[k] = new_pos
        sol_u[k], sol_v[k], sol_w[k] = new_vel
        # new_pos, new_vel = clohessy_wiltshire_solution(new_pos, new_vel, n=env.n, t=env.dt)
        new_pos, new_vel = clohessy_wiltshire_solution(pos_0, vel_0, n=env.n, t=env.t)
        k += 1

    # Compute terminal integration error:
    print(f"Terminal integration errors:")
    print(f"rx: {np.abs(x[-1] - sol_x[-1])} m")
    print(f"ry: {np.abs(y[-1] - sol_y[-1])} m")
    print(f"rz: {np.abs(z[-1] - sol_z[-1])} m")
    print(f"vx: {np.abs(u[-1] - sol_u[-1])} m/s")
    print(f"vy: {np.abs(v[-1] - sol_v[-1])} m/s")
    print(f"vz: {np.abs(w[-1] - sol_w[-1])} m/s")
    r_error = np.sqrt((x - sol_x)**2 + (y - sol_y)**2 + (z - sol_z)**2)
    v_error = np.sqrt((u - sol_u)**2 + (v - sol_v)**2 + (w - sol_w)**2)
    print(f"r: {r_error[-1]} m")
    print(f"v: {v_error[-1]} m/s")

    # Save the results in a dictionary:
    out = dict(
        t=t,
        x=x,
        y=y,
        z=z,
        u=u,
        v=v,
        w=w,
        sol_x=sol_x,
        sol_y=sol_y,
        sol_z=sol_z,
        sol_u=sol_u,
        sol_v=sol_v,
        sol_w=sol_w,
    )
    return out


def plot_results(data: dict) -> None:
    t = data.get("t")
    x = data.get("x")
    y = data.get("y")
    z = data.get("z")
    u = data.get("u")
    v = data.get("v")
    w = data.get("w")
    sol_x = data.get("sol_x")
    sol_y = data.get("sol_y")
    sol_z = data.get("sol_z")
    sol_u = data.get("sol_u")
    sol_v = data.get("sol_v")
    sol_w = data.get("sol_w")
    n_rows = 2
    n_cols = 3
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(10, 5))
    # Plot numerical:
    ax[0, 0].plot(t, x, label="numerical")
    ax[0, 1].plot(t, y, label="numerical")
    ax[0, 2].plot(t, z, label="numerical")
    ax[1, 0].plot(t, u, label="numerical")
    ax[1, 1].plot(t, v, label="numerical")
    ax[1, 2].plot(t, w, label="numerical")
    # Plot analytical:
    ax[0, 0].plot(t, sol_x, "--", linewidth=1, label="analytical")
    ax[0, 1].plot(t, sol_y, "--", label="analytical")
    ax[0, 2].plot(t, sol_z, "--", label="analytical")
    ax[1, 0].plot(t, sol_u, "--", label="analytical")
    ax[1, 1].plot(t, sol_v, "--", label="analytical")
    ax[1, 2].plot(t, sol_w, "--", label="analytical")
    # ax[0, 0].plot(t, np.abs(x - sol_x), "--", linewidth=1, label="error")
    # ax[0, 1].plot(t, np.abs(y - sol_y), "--", label="error")
    # ax[0, 2].plot(t, np.abs(z - sol_z), "--", label="error")
    # ax[1, 0].plot(t, np.abs(u - sol_u), "--", label="error")
    # ax[1, 1].plot(t, np.abs(v - sol_v), "--", label="error")
    # ax[1, 2].plot(t, np.abs(w - sol_w), "--", label="error")

    max_xyz = np.max(np.abs(np.hstack((x, y, z))))
    max_uvw = np.max(np.abs(np.hstack((u, v, w))))

    y_labels = [
        ["$r_x$ [m]", "$r_y$ [m]", "$r_z$ [m]"],
        ["$v_x$ [m/s]", "$v_y$ [m/s]", "$v_z$ [m/s]"],
    ]

    for row in range(n_rows):
        for col in range(n_cols):
            ax[row, col].grid()
            ax[row, col].legend()
            ax[row, col].set_xlim([t[0], t[-1]])
            ax[row, col].set_ylabel(y_labels[row][col])
            if row == 0:
                lim = max_xyz
                ax[row, col].set_ylim([-lim, lim])
            else:
                lim = max_uvw
                ax[row, col].set_ylim([-lim, lim])

    plt.show()
    plt.close()

    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
    # Integration error:
    ax2.plot(t, np.abs(x - sol_x), label="$r_x$ [m]")
    ax2.plot(t, np.abs(y - sol_y), label="$r_y$ [m]")
    ax2.plot(t, np.abs(z - sol_z), label="$r_z$ [m]")
    ax2.plot(t, np.abs(u - sol_u), label="$v_x$ [m/s]")
    ax2.plot(t, np.abs(v - sol_v), label="$v_y$ [m/s]")
    ax2.plot(t, np.abs(w - sol_w), label="$v_z$ [m/s]")
    ax2.set_xlabel("Time [s]")
    ax2.set_xlim([t[0], t[-1]])
    ax2.set_ylabel("Error")
    # ax2.set_ylim(bottom=0, top=None)
    plt.legend()
    plt.grid()
    plt.title("Integration errors")
    plt.show()
    plt.close()

    _ = plt.figure(num='3D', clear=True, figsize=(10, 5))
    ax3 = plt.axes(projection="3d")
    ax3.plot3D(x, y, z, label=f"Trajectory in {t[-1]} s")  # , mec="k", ms=10, alpha=0.8)
    ax3.plot3D(sol_x, sol_y, sol_z, "--", label="Analytical")
    ax3.plot3D(x[0], y[0], z[0], "kx", label="Chaser start")
    ax3.plot3D(x[-1], y[-1], z[-1], "g*", label="Chaser end")
    ax3.plot3D([0], [0], [0], "r.", label="Target")
    ax3.set_box_aspect((1, 1, 1))
    lim = ceil(max_xyz)
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


def main(args):
    # eval_env = RendezvousEnv(quiet=True)
    mu = 3.986004418e14
    n = np.sqrt(mu / (800e3 + 6371e3) ** 3)
    p = 2*np.pi / n
    t_max = int(p/8)
    config = dict(
        rc0=np.array([0, -10, 1]),
        vc0=np.array([-0.01, 0.01, 0]),
        qc0=np.array([1, 0, 0, 0]),
        wc0=np.array([0, 0, 0]),
        qt0=np.array([1, 0, 0, 0]),
        wt0=np.array([0, 0, 0.000001]),
        rc0_range=0,
        vc0_range=0,
        qc0_range=0,
        wc0_range=0,
        qt0_range=0,
        wt0_range=0,
        dt=1,
        t_max=t_max,  # 60*30,
        quiet=True,
    )
    eval_env = make_new_env(config)
    print(f"Integrating over {t_max} seconds ({round(t_max/60, 3)} minutes) ({round(t_max/p, 3)} orbits)")
    results = run_simulation(eval_env)
    if args.show:
        plot_results(results)
    return


class Arguments:
    def __init__(self, show=True):
        self.show = show
        return


if __name__ == "__main__":
    arguments = Arguments(show=True)
    main(arguments)
    # TODO: get some reference data to compare with.
