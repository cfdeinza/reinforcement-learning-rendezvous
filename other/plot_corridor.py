import matplotlib.pyplot as plt
from other.plot_trajectory import get_args
from utils.animations import load_data, plot_2dcomponents
import os


if __name__ == '__main__':
    arguments = get_args()
    arguments.path = os.path.join('logs', 'eval_trajectory_0.pickle')
    trajectory_data = load_data(arguments)
    corr = trajectory_data['trajectory'][6:, :]
    print(corr.shape)
    t = range(len(corr[0]))

    # 2D:
    fig = plt.figure(num=1, clear=True, figsize=(10, 6))
    ax = plt.axes()
    plot_2dcomponents(ax, t, corr[0], corr[1], corr[2],
                      labels=['x', 'y', 'z'], xlabel='Time (s)', ylabel='val', title='Corridor')
    plt.show()

# C:\Users\charl\.PyCharmCE2018.2\config\scratches\plot_corridor.py
