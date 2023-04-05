"""
This script generates trajectories using an existing model with varying initial conditions,
and then prints and plots the results.
"""

import os
import numpy as np
from math import radians, degrees
from arguments import get_monte_carlo_args
from utils.general import load_model, interp
from utils.environment_utils import make_env
from utils.plot_utils import print_results, plot_errors, plot_errors_vs_var
from rendezvous_eval import evaluate


def main(args):

    # Load an existing model:
    saved_model = load_model(path=args.model, env=None)  # env=dummy_env

    # Define the variable to be varied:
    variable = {"name": "wt0", "values": [radians(i) for i in [0, 0.5, 1, 1.5, 2]]}
    # variable = {"name": "rc0", "values": [9, 10, 11]}  # [19, 20, 21]
    # variable = {"name": "vc0", "values": []}

    var_name = variable["name"]
    var_values = variable["values"]
    datas = []
    eval_env = None

    for value in var_values:
        # Create an environment to evaluate the model:
        config = {}
        if var_name == "rc0":
            config[var_name] = np.array([0, -value, 0])
        elif var_name == "wt0":
            config[var_name] = np.array([0, 0, value])
        eval_env = make_env(reward_kwargs=None, quiet=False, config=config, stochastic=False)

        # Evaluate the model:
        data = evaluate(saved_model, eval_env, args)
        t = data["t"][0]
        new_t = np.arange(t[0], t[-1], 0.1)
        new_errors = interp(data["errors"], t, new_t)
        datas.append(
            {
                var_name: value,
                "t": new_t,
                "errors": new_errors,
                "total_delta_v": data["total_delta_v"],
                "collisions": data["collisions"],
                "successes": data["successes"],
            })

    # Print results:
    # dummy_env = make_env(reward_kwargs=None, quiet=True, config=None, stochastic=True)
    print_results(var=var_name, results=datas, max_rd_error=eval_env.max_rd_error, max_vd_error=eval_env.max_vd_error)

    # Plot the errors of each run:
    constraints = np.array([
            [eval_env.max_rd_error, eval_env.max_vd_error],
            [degrees(eval_env.max_qd_error), degrees(eval_env.max_wd_error)],
        ])
    plot_errors(var=var_name, datas=datas, constraints=constraints, t_max=eval_env.t_max)

    # Plot the errors as a function of `var_name`:
    plot_errors_vs_var(var=var_name, datas=datas, constraints=constraints)


if __name__ == "__main__":

    arguments = get_monte_carlo_args()
    # arguments.save = False
    # arguments.model = os.path.join("models", "model_rnn.zip")
    main(arguments)
