"""
Verify my implementation of the CW model.
"""
import numpy as np
import matplotlib.pyplot as plt
from utils.general import clohessy_wiltshire
from rendezvous_env import RendezvousEnv


def run_simulation(env, orbits=1):
    env.reset()
    dt = env.dt  # time step interval
    n = env.n  # mean motion
    orbital_period = 2 * np.pi / n  # orbital period [s]
    print(f"dt = {dt}")
    print(f"n = {n} (h = {env.h})")
    print(f"T = {orbital_period}")
    t = 0
    t_max = orbital_period * orbits
    rc = np.array([10, -10, 10])
    vc = np.array([0, 0, 0])
    ts = [0]
    rcs = [rc]
    vcs = [vc]

    while t < t_max:
        rc, vc = clohessy_wiltshire(rc, vc, n, dt)
        t += dt
        ts.append(t)
        rcs.append(rc)
        vcs.append(vc)

    # Convert the lists to arrays:
    ts = np.array(ts)
    rcs = np.array(rcs)
    vcs = np.array(vcs)

    # Save the results in a dictionary:
    out = dict(
        ts=ts,
        rcs=rcs,
        vcs=vcs,
    )
    return out


def plot_results(data: dict) -> None:
    t = data.get("ts")
    rc = data.get("rcs")
    vc = data.get("vcs")
    x = rc[:, 0]
    y = rc[:, 1]
    z = rc[:, 2]
    u = vc[:, 0]
    v = vc[:, 1]
    w = vc[:, 2]
    fig, ax = plt.subplots(2, 3, figsize=(10, 5))
    ax[0, 0].plot(t, x, label="rx")
    ax[0, 1].plot(t, y, label="ry")
    ax[0, 2].plot(t, z, label="rz")
    ax[1, 0].plot(t, u, label="vx")
    ax[1, 1].plot(t, v, label="vy")
    ax[1, 2].plot(t, w, label="vz")

    for i in range(2):
        for j in range(3):
            ax[i, j].grid()
            ax[i, j].legend()
            ax[i, j].set_xlim([t[0], t[-1]])
            if i == 0:
                lim = np.abs(rc).max()
                ax[i, j].set_ylim([-lim, lim])
            else:
                lim = np.abs(vc).max()
                ax[i, j].set_ylim([-lim, lim])

    plt.show()
    plt.close()

    _ = plt.figure(num='3D', clear=True, figsize=(10, 5))
    ax2 = plt.axes(projection="3d")
    ax2.plot3D(x, y, z, label=f"Trajectory in {t[-1]} s")  # , mec="k", ms=10, alpha=0.8)
    ax2.plot3D(x[0], y[0], z[0], "kx", label="Chaser start")
    ax2.plot3D(x[-1], y[-1], z[-1], "g*", label="Chaser end")
    ax2.plot3D([0], [0], [0], "ro", label="Target")
    # ax2.set_aspect('equal')
    ax2.set_box_aspect((1, 1, 1))
    lim = np.abs(rc).max()
    ax2.set_xlim([-lim, lim])
    ax2.set_ylim([-lim, lim])
    ax2.set_zlim([-lim, lim])
    ax2.set_zlabel("$Z_{LVLH}$ (m)")
    plt.xlabel('$X_{LVLH}$ (m)'), plt.ylabel('$Y_{LVLH}$ (m)')
    ax2.set_title("Chaser position")
    ax2.legend()
    plt.show()
    plt.close()

    return


if __name__ == "__main__":
    eval_env = RendezvousEnv(quiet=True)
    results = run_simulation(eval_env, orbits=1)
    plot_results(results)

    # TODO: get some reference data to compare with.
