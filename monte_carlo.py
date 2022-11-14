"""
This script generates trajectories using an existing model with varying initial conditions,
and then prints and plots the results.
"""

import os
import numpy as np
from math import radians, degrees
from arguments import get_monte_carlo_args
from utils.general import load_model
from utils.environment_utils import make_env
from utils.monte_carlo_utils import plot_errors, print_results
from rendezvous_eval import evaluate
from utils.vpython_utils import interp


def main(args):

    # Make an instance of the environment:
    reward_kwargs = None
    config = None
    dummy_env = make_env(reward_kwargs, quiet=True, config=config, stochastic=True)

    # Load an existing model:
    saved_model = load_model(path=args.model, env=dummy_env)

    # Define the variable to be varied:  # TODO: needs to be randomized
    variable = {"name": "wt0", "values": [radians(i) for i in [0, 1, 2]]}
    # variable = {"name": "rc0", "values": [19, 20, 21]}

    var = variable["name"]
    values = variable["values"]
    datas = []

    for value in values:
        # Create an environment to evaluate the model:
        config = {}
        if var == "rc0":
            config[var] = np.array([0, -value, 0])
        elif var == "wt0":
            config[var] = np.array([0, 0, value])
        eval_env = make_env(reward_kwargs, quiet=False, config=config, stochastic=False)

        # Evaluate the model:
        data = evaluate(saved_model, eval_env, args)
        t = data["t"][0]
        new_t = np.arange(t[0], t[-1], 0.1)
        new_errors = interp(data["errors"], t, new_t)
        datas.append(
            {
                var: value,
                "t": new_t,
                "errors": new_errors,
                "total_delta_v": data["total_delta_v"],
                "collisions": data["collisions"],
                "success": data["success"],
            })

    # Print results:
    print_results(var=var, results=datas, max_rd_error=dummy_env.max_rd_error)

    # Plot the errors of each run:
    plot_errors(
        var=var,
        datas=datas,
        constraints=np.array([
            [dummy_env.max_rd_error, dummy_env.max_vd_error],
            [degrees(dummy_env.max_qd_error), degrees(dummy_env.max_wd_error)],
        ]),
        t_max=dummy_env.t_max
    )

    # TODO: Plot errors as a function of var


if __name__ == "__main__":

    arguments = get_monte_carlo_args()
    # arguments.save = False
    arguments.model = os.path.join("models", "mlp_model_att_01.zip")
    main(arguments)
