## Information:
This repository contains the work of Carlos F. De Inza Niemeijer for his MSc thesis project at TU Delft.
The purpose of this project is to use reinforcement learning to train a controller that can perform an autonomous rendezvous with a tumbling target. 

A custom environment was created to simulate the rendezvous scenario. This environment is located in `rendezvous_env.py`. It follows the guidelines specified in [OpenAI Gym](https://www.gymlibrary.dev/content/environment_creation/). 
The learning algorithm chosen to train the controller is the Proximal Policy Optimization (PPO) algorithm implemented by [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).

## Contents:

### Relevant files:
- `main.py`: the main script that is executed to train the controller.
- `rendezvous_env.py`: a custom gym environment to simulate a 6-DOF rendezvous.
- `rendezvous_eval.py`: a script to evaluate the performance of the rendezvous controller.
- `rendezvous_plot_vpython.py`: a script to plot a 3D animation of a rendezvous trajectory.
- `rendezvous_plot_2d.py`: a script to plot 2D graphs of the position, velocity, and attitude of the chaser.

### Directories:
- `custom/`: custom features used on Stable-Baselines3 (callback functions, policies, environments, etc).
- `data/`: pickle files containing trajectory data.
- `ideas/`: relevant information to be implemented in the future.
- `logs/`: logs from previous training and evaluation runs.
- `models/`: trained RL models.
- `other/`: miscellaneous files that may be needed for future work.
- `plots/`: plots and animations of the trajectories.
- `utils/`: utility functions.