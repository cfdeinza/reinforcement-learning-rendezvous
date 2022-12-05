"""
Verify that the initial state of the simulation has the expected distribution.
"""
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from rendezvous_env import RendezvousEnv
from utils.quaternions import quat2rot, quat_error, quat2mat
from utils.general import normalize_value
import plotly.graph_objects as go


def get_samples(env: RendezvousEnv, n=100) -> dict:
    """
    Reset the environment `n` times to get `n` samples of the initial state.\n
    :param env: Rendezvous environment
    :param n: Number of samples to take
    :return: dictionary with initial state data
    """

    x, y, z = [], [], []
    rcs = []        # list of (3,) ndarrays containing the chaser initial position [m]
    vcs = []        # list of (3,) ndarrays containing the chaser initial velocity [m/s]
    qcs = []        # list of (4,) ndarrays containing the chaser quaternion error [-]
    qts = []        # list of (4,) ndarrays containing the target quaternion error [-]
    rc_errors = []  # list of chaser position error magnitudes [m]
    vc_errors = []  # list of chaser velocity error magnitudes [m/s]
    qc_errors = []  # list of chaser attitude error magnitudes [deg]
    wc_errors = []  # list of chaser rot rate error magnitudes [deg/s]
    qt_errors = []  # list of target attitude error magnitudes [deg]
    wt_errors = []  # list of target rot rate error magnitudes [deg/s]
    for i in range(n):
        env.reset()
        drc = env.rc - env.nominal_rc0
        dvc = env.vc - env.nominal_vc0
        dqc = quat_error(env.nominal_qc0, env.qc)
        dwc = env.wc - env.nominal_wc0
        dqt = quat_error(env.nominal_qt0, env.qt)
        dwt = env.wt - env.nominal_wt0
        # Add to lists:
        rc_errors.append(np.linalg.norm(drc))
        vc_errors.append(np.linalg.norm(dvc))
        qc_errors.append(np.degrees(quat2rot(dqc)[1]))
        wc_errors.append(np.degrees(np.linalg.norm(dwc)))
        qt_errors.append(np.degrees(quat2rot(dqt)[1]))
        wt_errors.append(np.degrees(np.linalg.norm(dwt)))
        x.append(drc[0])
        y.append(drc[1])
        z.append(drc[2])
        rcs.append(env.rc)
        vcs.append(env.vc)
        qcs.append(dqc)
        qts.append(dqt)

    # Convert to numpy arrays:
    rc_errors = np.array(rc_errors)
    vc_errors = np.array(vc_errors)
    qc_errors = np.array(qc_errors)
    wc_errors = np.array(wc_errors)
    qt_errors = np.array(qt_errors)
    wt_errors = np.array(wt_errors)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # Store in dictionary
    out = dict(
        rc_errors=rc_errors,
        vc_errors=vc_errors,
        qc_errors=qc_errors,
        wc_errors=wc_errors,
        qt_errors=qt_errors,
        wt_errors=wt_errors,
        x=x,
        y=y,
        z=z,
        rcs=rcs,
        vcs=vcs,
        qcs=qcs,
        qts=qts,
    )
    return out


def print_line(name: str, value: np.ndarray, value_range, unit: str):
    unit = "[" + unit + "]"
    print(f"{name.rjust(4)} {unit.ljust(7)} | "
          f"{str(round(value.min(), 3)).center(7)} | "
          f"{str(round(value.max(), 3)).center(7)} | "
          f"{str(round(value.mean(), 3)).center(7)} | "
          f"{str(round(1/2 * value_range, 3)).center(12)} | "
          f"{str(round(value.var(), 3)).center(7)} | "
          f"{str(round(1/12 * value_range**2, 3)).center(12)}")
    return


def print_distribution_stats(env: RendezvousEnv, data: dict) -> None:
    """
    Print the statistics of the samples, along with the expected statistics for a uniform distribution.\n
    :param env: Rendezvous environment
    :param data: dictionary containing data
    :return: None
    """
    # Get values from dictionary:
    rc_errors = data.get("rc_errors")
    vc_errors = data.get("vc_errors")
    qc_errors = data.get("qc_errors")
    wc_errors = data.get("wc_errors")
    qt_errors = data.get("qt_errors")
    wt_errors = data.get("wt_errors")
    # Print stats:
    print(f"Total samples: {rc_errors.size}")
    print(f"Parameter    |   Min   |   Max   |   Avg   | Expected Avg |   Var   |   Expected Var")
    print_line("rc0", rc_errors, env.rc0_range - 0, "m")
    print_line("vc0", vc_errors, env.vc0_range - 0, "m/s")
    print_line("qc0", qc_errors, np.degrees(env.qc0_range) - 0, "deg")
    print_line("wc0", wc_errors, np.degrees(env.wc0_range) - 0, "deg/s")
    print_line("qt0", qt_errors, np.degrees(env.qt0_range) - 0, "deg")
    print_line("wt0", wt_errors, np.degrees(env.wt0_range) - 0, "deg/s")
    return


def plot_histogram(data: dict) -> None:
    """
    Plot a histogram to show the distribution of the initial states.\n
    :param data: dictionary containing data
    :return: None
    """
    # Get values from dictionary:
    # rcs = [np.linalg.norm(i) for i in data.get("rcs")]
    rc_errors = data.get("rc_errors")
    vc_errors = data.get("vc_errors")
    qc_errors = data.get("qc_errors")
    wc_errors = data.get("wc_errors")
    qt_errors = data.get("qt_errors")
    wt_errors = data.get("wt_errors")
    fig, ax = plt.subplots(2, 3, figsize=(10, 5))
    bins = 10
    ax[0, 0].hist(rc_errors, bins=bins), ax[0, 0].grid(), ax[0, 0].set_xlabel("$\Delta r_{c,0}$ [m]")
    ax[0, 1].hist(vc_errors, bins=bins), ax[0, 1].grid(), ax[0, 1].set_xlabel("$\Delta v_{c,0}$ [m/s]")
    ax[0, 2].hist(qc_errors, bins=bins), ax[0, 2].grid(), ax[0, 2].set_xlabel("$\Delta \\theta_{c,0}$ [deg]")
    ax[1, 0].hist(wc_errors, bins=bins), ax[1, 0].grid(), ax[1, 0].set_xlabel("$\Delta \omega_{c,0}$ [deg/s]")
    ax[1, 1].hist(qt_errors, bins=bins), ax[1, 1].grid(), ax[1, 1].set_xlabel("$\Delta \\theta_{t,0}$ [deg/s]")
    ax[1, 2].hist(wt_errors, bins=bins), ax[1, 2].grid(), ax[1, 2].set_xlabel("$\Delta \omega_{t,0}$ [deg/s]")
    n_samples = rc_errors.size
    fig.suptitle(f"Samples: {n_samples}")
    plt.show()
    plt.close()

    return


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


def arrow(origin=np.array([0, 0, 0]), dir0=np.array([0, 1, 0]), q=np.array([1, 0, 0, 0])):
    new_dir = np.matmul(quat2mat(q), dir0)
    arrow_x = [origin[0], origin[0] + new_dir[0]]
    arrow_y = [origin[1], origin[1] + new_dir[1]]
    arrow_z = [origin[2], origin[2] + new_dir[2]]
    return arrow_x, arrow_y, arrow_z


def plot_chaser_plotly(env: RendezvousEnv, data: dict) -> None:
    """
    Plot the initial position and velocity of the chaser using plotly. It looks better than with matplotlib.\n
    :param env: Rendezvous environment
    :param data: dictionary containing data
    :return: None
    """
    vc0_range = env.vc0_range
    vcs = data.get("vcs")
    x, y, z = data.get("x"), data.get("y"), data.get("z")
    fig_dict = {
        'data': [],
        'layout': {
            'scene': dict(
                xaxis=dict(range=[-1, 1], title="X (m)"), xaxis_showspikes=False,
                yaxis=dict(range=[-11, -9], title="Y (m)"), yaxis_showspikes=False,
                zaxis=dict(range=[-1, 1], title="Z (m)"), zaxis_showspikes=False),
            'width': 800,
            'height': 700,
            'scene_aspectmode': 'cube',
        }}
    fig = go.Figure(fig_dict)
    for i in range(len(x)):
        vc = vcs[i]
        vc_mag = np.linalg.norm(vc)
        size = [0.3, 0.31]
        color = normalize_value(vc_mag, low=0, high=vc0_range, custom_range=size)
        vc = vc / vc_mag * color
        fig.add_trace(go.Cone(x=[x[i]], y=[y[i]], z=[z[i]],
                              u=[vc[0]], v=[vc[1]], w=[vc[2]], opacity=1, showlegend=False, showscale=False,
                              cmin=size[0], cmax=size[1], sizemode="absolute",
                              colorscale=[[0.2, "rgb(0,0,255)"], [0.21, "rgb(255,0,0)"]],
                              )
                      )
    fig.add_trace(
        (go.Cone(x=[0], y=[-10], z=[0], u=[0.01], v=[0.01], w=[0.01], visible=True, showlegend=False, showscale=True,
                 cmin=0, cmax=0.1, colorbar=dict(title="Initial velocity (m/s)"),
                 colorscale=[[0, "rgb(0,0,255)"], [0.1, "rgb(255,0,0)"]]))
    )
    fig.show()
    return


def plot_target(data: dict) -> None:
    """
    Plot the initial attitude of the target.\n
    :param data: dictionary containing data
    :return: None
    """
    # 3D plot of the target initial state:
    qt_errors = data.get("qt_errors")
    qts = data.get("qts")
    _ = plt.figure(num="3D", clear=True, figsize=(10, 5))
    ax3 = plt.axes(projection="3d")
    ax3.plot3D([0], [0], [0], "go", mec="k")
    for i in range(len(qt_errors)):
        xa, ya, za = arrow(origin=np.array([0, 0, 0]), dir0=np.array([0, -1, 0]), q=qts[i])
        a = Arrow3D(xa, ya, za, mutation_scale=10, lw=1, arrowstyle="-|>", color="k")
        ax3.add_artist(a)
    ax3.set_box_aspect((1, 1, 1))
    ax3.set_xlim([-2, 2])
    ax3.set_ylim([-2, 2])
    ax3.set_zlim([-2, 2])
    ax3.set_zlabel("z (m)")
    plt.xlabel('x (m)'), plt.ylabel('y (m)')
    ax3.set_title("Target initial attitude")
    plt.show()
    plt.close()
    return


def plot_all(data: dict) -> None:
    """
    Plot the chaser and the target together.\n
    :param data: dictionary containing data
    :return: None
    """
    # 3D plot of whole system:
    x, y, z = data.get("x"), data.get("y"), data.get("z")
    qt_errors = data.get("qt_errors")
    qts = data.get("qts")
    _ = plt.figure(num='3D', clear=True, figsize=(10, 5))
    ax4 = plt.axes(projection="3d")
    ax4.plot3D(x, y, z, "r.", mec="k", ms=10, label="Chaser")
    ax4.plot3D([0], [0], [0], "go", mec="k", ms=10, label="Target")
    for i in range(len(qt_errors)):
        xa, ya, za = arrow(origin=np.array([0, 0, 0]), dir0=np.array([0, -8, 0]), q=qts[i])
        a = Arrow3D(xa, ya, za, mutation_scale=10, lw=1, arrowstyle="-|>", color="k")
        ax4.add_artist(a)
    ax4.set_box_aspect((1, 1, 1))
    ax4.set_xlim([-11, 11])
    ax4.set_ylim([-21, 1])
    ax4.set_zlim([-11, 11])
    ax4.set_xlabel("x [m]"), ax4.set_ylabel("y [m]"), ax4.set_zlabel("z [m]")
    ax4.legend()
    plt.show()
    plt.close()
    return


if __name__ == "__main__":
    eval_env = RendezvousEnv()
    results = get_samples(eval_env, n=1000)
    print_distribution_stats(eval_env, results)
    plot_histogram(results)
    plot_target(results)
