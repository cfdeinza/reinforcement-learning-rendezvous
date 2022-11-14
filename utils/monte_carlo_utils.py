import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


def plot_errors(var: str, datas: list, constraints: np.ndarray, t_max: int):
    """
    Plot a figure showing the position, velocity, attitude, and rotation rate errors of each Monte Carlo run.\n
    :param var: name of the variable that is being randomly sampled
    :param datas: list of dictionaries containing the results of each run
    :param constraints: array of constraints for each state variable (pos, vel, att, rot rate)
    :param t_max: maximum episode length allowed by the environment
    :return: None
    """
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))

    # Plot the errors for each run:
    for run in datas:
        symbol = get_symbol(var)
        value = check_rad2deg(name=var, values=run[var])
        units = get_units(var)
        ax[0, 0].plot(run["t"], run["errors"][0], label=f"{symbol} = {round(value, 2)} {units}")
        ax[0, 1].plot(run["t"], run["errors"][1], label=f"{symbol} = {round(value, 2)} {units}")
        ax[1, 0].plot(run["t"], np.degrees(run["errors"][2]), label=f"{symbol} = {round(value, 2)} {units}")
        ax[1, 1].plot(run["t"], np.degrees(run["errors"][3]), label=f"{symbol} = {round(value, 2)} {units}")

    # Specify the title for each subplot:
    titles = np.array([
        ["Position error [m]", "Velocity error [m/s]"],
        ["Attitude error [deg]", "Rotation rate error [deg/s]"],
    ])

    # List with the y-limit for each subplot:
    ylims = np.array([
        [[0, 20], [0, 2]],
        [[0, 35], [0, 10]],
    ])

    # Plot the constraints for each state variable, and format the axes:
    for i in range(2):
        for j in range(2):
            ax[i, j].plot([0, t_max], [constraints[i, j]] * 2, "k--", label="Constraint")
            format_ax(ax[i, j], xlim=[0, t_max], xlabel="Time [s]", ylim=ylims[i, j], title=titles[i, j])

    plt.tight_layout()
    plt.show()
    plt.close()

    return


def get_symbol(name: str):
    """
    Returns the name of the variable in symbol form. Used for some plot labels.\n
    :param name: name of the variable (e.g. 'rc0', 'qt0', etc)
    :return:
    """
    symbols = {
        "rc0": "$r_{c,0}$",
        "vc0": "$v_{c,0}$",
        "qc0": "$q_{c,0}$",
        "wc0": "$\\omega_{c,0}$",
        "qt0": "$q_{t,0}$",
        "wt0": "$\\omega_{t,0}$",
        "koz_radius": "$R_{KOZ}$",
        "corridor_half_angle": "$\\theta_{corr}$",
        "h": "$h$",
        "dt": "$\\Delta t$"
    }
    return symbols.get(name)


def get_units(name: str):
    """
    Returns the units associated with the given state variable:\n
    - position: meters
    - velocity: meters per second
    - angular displacement: degrees (assumes it has already been converted from radians)
    - angulaer rate: degrees per second (assumes it has already been converted from radians)\n
    :param name: name of the state variable (e.g. 'rc0', 'qt0', etc)
    :return:
    """
    if name == "rc0":
        units = "m"
    elif name == "vc0":
        units = "m/s"
    elif name == "qc0" or name == "qt0":
        units = "deg"
    elif name == "wc0" or name == "wt0":
        units = "deg/s"
    else:
        units = "[?]"
    return units


def check_rad2deg(name: str, values):
    """
    Checks if the given variable contains attitude or rotation rate values. If so, it converts the values to degrees.\n
    :param name: name of the variable
    :param values: numeric value(s) associated to the variable name
    :return: values converted to degrees
    """
    names = ["qc0", "qt0", "wc0", "wt0"]
    if name in names:
        values = np.degrees(values)
    return values


def format_ax(ax, xlim=None, ylim=None, xlabel=None, ylabel=None, title=None):
    """
    Adds axis limits, axis labels, title, grid, and legend to a given set of axes.\n
    :param ax: set of axes
    :param xlim: limits of the x-axis
    :param ylim: limits of the y-axis
    :param xlabel: label of the x-axis
    :param ylabel: label of the y-axis
    :param title: title of the set of axes
    :return:
    """
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid()
    ax.legend()
    return


def print_results(var: str, results: list, max_rd_error: float):
    """
    Prints the results of the Monte Carlo simulation in table format.
    Note: the rise time is defined as the first time during which the position error is within the allowed constraint.
    It does not consider whether or not the position error stays within the constraint after that point.\n
    :return: None
    """
    headers = ["Run",
               f"{var} {get_units(name=var)}",
               "ep_len [s]",
               "rise time [s]",
               "SS pos error [m]",
               "SS vel error [m/s]",
               "SS att error [deg]",
               "SS rot error [deg/s]",
               "Delta V (m/s)",
               "Collisions",
               "Success"]
    all_rows = []
    for i, run in enumerate(results):
        out = list()
        out.append(i + 1)                                       # Run number
        out.append(check_rad2deg(name=var, values=run[var]))    # Value of the random variable
        out.append(run["t"][-1])                                # Episode length
        pos_error = run["errors"][0]
        if np.any(pos_error < max_rd_error):
            index_ss = np.argmax(pos_error < max_rd_error)                  # First index where pos error < constraint
            out.append(run["t"][index_ss])                                  # Rise time (s)
            out.append(run["errors"][0, index_ss:-1].mean())                # Average steady-state pos error (m)
            out.append(run["errors"][1, index_ss:-1].mean())                # Average steady-state vel error (m/s)
            out.append(np.degrees(run["errors"][2, index_ss:-1].mean()))    # Average steady-state vel error (m/s)
            out.append(np.degrees(run["errors"][3, index_ss:-1].mean()))    # Average steady-state rot error (deg/s)
        else:
            out.append(None)                # Rise time (s)
            out.append(None)                # Average steady-state pos error (m)
            out.append(None)                # Average steady-state vel error (m/s)
            out.append(None)                # Average steady-state vel error (m/s)
            out.append(None)                # Average steady-state rot error (deg/s)
        out.append(run["total_delta_v"])    # Total delta V used throughout the episode (m/s)
        out.append(run["collisions"])       # Number of collisions during episode
        out.append(run["success"])          # Amount of time steps where terminal conditions were achieved
        all_rows.append(out)
    print(tabulate(all_rows, headers=headers))
    return
