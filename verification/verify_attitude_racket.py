import numpy as np
import matplotlib.pyplot as plt
from utils.environment_utils import make_env
from utils.quaternions import quat2mat
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

"""
Verify that the dynamics model can replicate complex dynamics effects such as the intermediate axis theorem, 
also known as the Dzhanibekov effect.
Result: It behaves as expected. The intermediate axis starts wobbling and switches directions.
"""


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self):  # , renderer=None
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def main():

    action = np.array([0, 0, 0, 0, 0, 0])
    config = dict(
        rc0=np.array([0, -10, 0]),
        vc0=np.array([0, 0, 0]),
        qc0=np.array([1, 0, 0, 0]),
        wc0=np.radians(np.array([0, 5, 0.01])),
        dt=1,
    )
    stochastic = False
    env = make_env(reward_kwargs=None, config=config, stochastic=stochastic)
    _ = env.reset()
    # rs = [env.rc]
    # vs = [env.vc]
    q = [env.qc]
    w = [env.wc]  # rotation rate expressed in C frame
    wl = [env.chaser2lvlh(env.wc)]  # rotation rate expressed in LVLH
    cbx = np.array([1, 0, 0])
    cby = np.array([0, 1, 0])
    cbz = np.array([0, 0, 1])
    cx = [env.chaser2lvlh(cbx)]
    cy = [env.chaser2lvlh(cby)]  # capture axis expressed in LVLH
    cz = [env.chaser2lvlh(cbz)]
    ts = [env.t]
    print(f"Initial state:")
    print_state(env)
    # orbits = 1
    t_max = 760#688  # 2*np.pi/env.n * orbits
    while env.t < t_max:
        env.step(action)
        # rs.append(env.rc)
        # vs.append(env.vc)
        q.append(env.qc)
        w.append(env.wc)
        wl.append(env.chaser2lvlh(env.wc))
        cx.append(env.chaser2lvlh(cbx))
        cy.append(env.chaser2lvlh(cby))
        cz.append(env.chaser2lvlh(cbz))
        ts.append(env.t)

    print("Final state:")
    print_state(env)
    # rs = np.vstack(rs)
    # vs = np.vstack(vs)
    q = np.vstack(q)
    w = np.degrees(np.vstack(w))  # chaser rot rate (expressed in C) deg/s
    wl = np.degrees(np.vstack(wl))  # chaser rot rate (expressed in LVLH) deg/s
    cx = np.vstack(cx)
    cy = np.vstack(cy)                # capture axis (expressed in LVLH) m
    cz = np.vstack(cz)
    ts = np.hstack(ts)  # / t_max]

    # r_lim = round(np.max(np.abs(rs)) + 1)
    # v_lim = round(np.max(np.abs(vs)) + 0.5)
    q_lim = 1.1
    w_lim = max(5, np.max(np.abs(w)))

    # Plot q and w as a function of t:
    lw = 2
    rows, cols = 3, 4
    fig, ax = plt.subplots(rows, cols, figsize=(10, 5))
    ax[0, 0].plot(ts, q[:, 0])
    ax[0, 1].plot(ts, q[:, 1])
    ax[0, 2].plot(ts, q[:, 2])
    ax[0, 3].plot(ts, q[:, 3])
    ax[1, 0].plot(ts, w[:, 0])
    ax[1, 1].plot(ts, w[:, 1])
    ax[1, 2].plot(ts, w[:, 2])
    ax[2, 0].plot(ts, wl[:, 0])
    ax[2, 1].plot(ts, wl[:, 1])
    ax[2, 2].plot(ts, wl[:, 2])
    ax[0, 0].set_ylabel(f"$q_w$")
    ax[0, 1].set_ylabel(f"$q_x$")
    ax[0, 2].set_ylabel(f"$q_y$")
    ax[0, 3].set_ylabel(f"$q_z$")
    ax[1, 0].set_ylabel(f"$\omega_x$ [deg/s]")
    ax[1, 1].set_ylabel(f"$\omega_y$ [deg/s]")
    ax[1, 2].set_ylabel(f"$\omega_z$ [deg/s]")
    ax[2, 0].set_ylabel("$\omega_{x,LVLH}$ [deg/s]")
    ax[2, 1].set_ylabel("$\omega_{y,LVLH}$ [deg/s]")
    ax[2, 2].set_ylabel("$\omega_{z,LVLH}$ [deg/s]")
    for i in range(rows):
        for j in range(cols):
            ax[i, j].set_xlabel("Time [s]")
            ax[i, j].set_xlim([ts[0], ts[-1]])
            ax[i, j].grid()
            if i == 0:
                ax[i, j].set_ylim([-q_lim, q_lim])
            else:
                ax[i, j].set_ylim([-w_lim, w_lim])
    plt.tight_layout()
    plt.show()
    plt.close()

    # 3D plot of the vector w over time:
    # x = wl[:, 0]
    # y = wl[:, 1]
    # z = wl[:, 2]
    x1 = cx[:, 0]
    y1 = cx[:, 1]
    z1 = cx[:, 2]
    x2 = cy[:, 0]
    y2 = cy[:, 1]
    z2 = cy[:, 2]
    x3 = cz[:, 0]
    y3 = cz[:, 1]
    z3 = cz[:, 2]
    _ = plt.figure(num='3D', clear=True, figsize=(10, 5))
    ax3 = plt.axes(projection="3d")
    # Initial chaser axes:
    # ax3.plot3D([0, x1[0]], [0, y1[0]], [0, z1[0]], ls="--", lw=2, color="tab:red")
    # ax3.plot3D([0, x2[0]], [0, y2[0]], [0, z2[0]], ls="--", lw=2, color="tab:green")
    # ax3.plot3D([0, x3[0]], [0, y3[0]], [0, z3[0]], ls="--", lw=2, color="tab:blue")
    lw = 4
    alpha = 0.4
    # linestyle = "dashed"
    xc0 = Arrow3D([0, x1[0]], [0, y1[0]], [0, z1[0]], mutation_scale=10,
                  lw=lw, arrowstyle="-|>", color="tab:red", alpha=alpha)
    yc0 = Arrow3D([0, x2[0]], [0, y2[0]], [0, z2[0]], mutation_scale=10,
                  lw=lw, arrowstyle="-|>", color="tab:green", alpha=alpha)
    zc0 = Arrow3D([0, x3[0]], [0, y3[0]], [0, z3[0]], mutation_scale=10,
                  lw=lw, arrowstyle="-|>", color="tab:blue", alpha=alpha)
    for item in [xc0, yc0, zc0]:
        ax3.add_artist(item)
    ax3.plot3D([0], [0], [0], "k.", label="LVLH origin", ms=12)
    xcf = Arrow3D([0, x1[-1]], [0, y1[-1]], [0, z1[-1]], mutation_scale=15,
                  lw=lw, arrowstyle="-|>", color="tab:red")
    ycf = Arrow3D([0, x2[-1]], [0, y2[-1]], [0, z2[-1]], mutation_scale=15,
                  lw=lw, arrowstyle="-|>", color="tab:green")
    zcf = Arrow3D([0, x3[-1]], [0, y3[-1]], [0, z3[-1]], mutation_scale=15,
                  lw=lw, arrowstyle="-|>", color="tab:blue")
    for item in [xcf, ycf, zcf]:
        ax3.add_artist(item)

    # Final chaser axes:
    # ax3.plot3D([0, x1[-1]], [0, y1[-1]], [0, z1[-1]], ls="-", lw=2, color="tab:red")
    # ax3.plot3D([0, x2[-1]], [0, y2[-1]], [0, z2[-1]], ls="-", lw=2, color="tab:green")
    # ax3.plot3D([0, x3[-1]], [0, y3[-1]], [0, z3[-1]], ls="-", lw=2, color="tab:blue")
    ax3.plot3D(x2, y2, z2, "--", color="k", lw=1, label=f"Path traced by $Y_C$")
    # ax3.plot3D(x3, y3, z3, label=f"$Z_C$")
    # for i in range(0, len(x2), 60):
    #     ax3.plot3D([0, x2[i]], [0, y2[i]], [0, z2[i]], "k-")
    # ax3.plot3D([0, x2[0]], [0, y2[0]], [0, z2[0]], "k-")
    # ax3.plot3D([0, x2[-1]], [0, y2[-1]], [0, z2[-1]], "k-")
    # ax3.plot3D(x2[0], y2[0], z2[0], "kx", label="Start")
    # ax3.plot3D(x2[-1], y2[-1], z2[-1], "g*", label="End")

    ax3.set_box_aspect((1, 1, 1))
    # lim = np.linalg.norm(np.degrees(env.nominal_wc0))
    lim = 1.5
    # ax3.set_xlim([-20, 20])
    # ax3.set_ylim([-5, 35])
    # ax3.set_zlim([-20, 20])
    ax3.set_xlim([-lim, lim])
    ax3.set_ylim([-lim, lim])
    ax3.set_zlim([-lim, lim])
    ax3.set_zlabel("$Z_{LVLH}$ (m)")
    plt.xlabel('$X_{LVLH}$ (m)'), plt.ylabel('$Y_{LVLH}$ (m)')
    ax3.set_title("Chaser rotation rate [deg/s]")
    # ax3.legend()
    plt.show()
    plt.close()


def print_state(env):
    # print(f"rc: {env.rc.round(3)} m, magnitude: {round(np.linalg.norm(env.rc), 3)} m")
    # print(f"vc: {env.vc.round(3)} m/s, magnitude: {round(np.linalg.norm(env.vc), 3)} m/s")
    print(f"qc: {env.qc.round(3)}, magnitude: {np.degrees(2*np.arccos(env.qc[0])).round(2)} deg")
    print(f"wc: {np.degrees(env.wc).round(3)} deg/s, magnitude: {round(np.linalg.norm(np.degrees(env.wc)), 3)} deg/s")
    # print(f"qt: {env.qt.round(3)}, magnitude: {np.degrees(2*np.arccos(env.qt[0])).round(2)} deg")
    # print(f"wt: {np.degrees(env.wt).round(3)} deg/s, magnitude: {round(np.linalg.norm(np.degrees(env.wt)), 3)} deg/s")
    return


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 14})
    main()
