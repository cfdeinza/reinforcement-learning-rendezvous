"""
Generate initial conditions for the environment.
The initial conditions are saved in a csv file.
"""

import numpy as np
import pandas as pd
from utils.environment_utils import make_env
from arguments import get_main_args


def main(args):

    if args.save:
        print(f"Results will be saved in '{args.output_file}'\n")
    else:
        print("Results will NOT be saved. Use the `--save` argument to save the results.\n")
    # Create an environment to get the initial conditions from:
    env_config = dict()
    env = make_env(reward_kwargs=None, quiet=False, config=env_config, stochastic=True)

    n_samples = 5

    data = []
    for i in range(n_samples):
        env.reset()
        data.append(np.hstack((env.rc, env.vc, env.qc, env.wc, env.qt, env.wt)))

    col_names = [
        'rcx', 'rcy', 'rcz',
        'vcx', 'vcy', 'vcz',
        'qcw', 'qcx', 'qcy', 'qcz',
        'wcx', 'wcy', 'wcz',
        'qtw', 'qtx', 'qty', 'qtz',
        'wtx', 'wty', 'wtz'
    ]
    df = pd.DataFrame(data, columns=col_names)

    print(df)
    if args.save:
        print(f"Saving the initial conditions in '{args.output_file}'...", end=" ")
        df.to_csv(args.output_file)
        print("Done")
    return


if __name__ == "__main__":
    arguments = get_main_args()
    arguments.output_file = "initial.csv"
    arguments.save = True
    main(arguments)
