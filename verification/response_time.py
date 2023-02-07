from utils.general import load_model
from utils.environment_utils import make_env
from time import perf_counter
from arguments import get_main_args
import numpy as np
import matplotlib.pyplot as plt


def evaluate_time(model, env):

    env.reset()

    obs = env.get_observation()
    lstm_states = None
    ep_start = np.ones(shape=(1,), dtype=bool)
    done = False

    # Create empty arrays to save values:
    expected_timesteps = int(env.t_max / env.dt) + 1  # size of the empty arrays
    # Time:
    t = np.full((1, expected_timesteps), np.nan)  # times
    response_times = np.full((1, expected_timesteps), np.nan)

    # Save the initial values:
    t[0, 0] = env.t
    # response_times[0, 0] = 0

    k = 1
    while not done:

        # Select action:
        start = perf_counter()
        action, lstm_states = model.predict(
            observation=obs,  # input to the policy network
            state=lstm_states,  # last hidden state (used for recurrent policies)
            episode_start=ep_start,  # the last mask? (used for recurrent policies)
            deterministic=True  # whether or not to return deterministic actions (default is False)
        )
        stop = perf_counter()

        # Step forward in time:
        obs, reward, done, info = env.step(action)
        ep_start[0] = done

        # Save current values:
        t[0, k] = env.t
        response_times[0, k-1] = stop - start
        print(f'Step: {env.t}', end='\r')
        k += 1

    # trajectory = info['trajectory']
    # actions = info['actions']

    # Remove nans if the episode was terminated before t_max:
    if env.t < env.t_max:
        t = np.array([t[~np.isnan(t)]])
        response_times = response_times[:, 0:t.size]

    return dict(t=t, response_times=response_times)


def plot_times(ax, x, y, title=None, xlim=None, ylim=None):
    ax.plot(x, y)
    ax.set_ylabel("NN response time [ms]")
    ax.set_xlabel("Time [s]")
    ax.grid()
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    return


def main(args):
    # Make eval environment:
    reward_kwargs = None
    env_config = {}
    env = make_env(reward_kwargs, config=env_config, quiet=False)

    # Make MLP model:
    mlp_model = load_model(path=args.mlp, env=None)

    # Measure MLP response times:
    mlp_results = evaluate_time(mlp_model, env)

    # Make RNN model:
    rnn_model = load_model(path=args.rnn, env=None)

    # Measure RNN response time:
    rnn_results = evaluate_time(rnn_model, env)

    # Plot response times:
    i_0 = 0
    i_end = -1
    mlp_x = mlp_results.get("t")[0, i_0:i_end]
    mlp_y = mlp_results.get("response_times")[0, i_0:i_end] * 1000
    rnn_x = rnn_results.get("t")[0, i_0:i_end]
    rnn_y = rnn_results.get("response_times")[0, i_0:i_end] * 1000
    max_x = np.max(np.append(mlp_x, rnn_x)) + 1
    max_y = np.max(np.append(mlp_y, rnn_y)) * 1.1
    mlp_title = f"MLP response time (avg = {mlp_y.mean().round(3)} ms)"
    rnn_title = f"RNN response time (avg = {rnn_y.mean().round(3)} ms)"
    fig, ax = plt.subplots(1, 2, figsize=(5, 5))
    plot_times(ax[0], mlp_x, mlp_y, title=mlp_title, xlim=[0, max_x], ylim=[0, max_y])
    plot_times(ax[1], rnn_x, rnn_y, title=rnn_title, xlim=[0, max_x], ylim=[0, max_y])

    plt.show()
    plt.close()

    return


if __name__ == "__main__":
    arguments = get_main_args()
    arguments.mlp = r"C:\Users\charl\Downloads\mlp_model_box_05_01.zip"
    arguments.rnn = r"C:\Users\charl\Downloads\rnn_model_box_03_03.zip"
    main(arguments)
