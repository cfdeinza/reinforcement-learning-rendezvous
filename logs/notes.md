# Notes on the files:

## Trajectory files:
- `rdv_trajectory_0` is one of the earliest trajectories that successfully entered the (static) cone.
- `rdv_trajectory_1&2` also achieved corridor entry. (rewarded position and angle, and penalized collision)
- `rdv_trajectory_y` entered the corridor pointed in the +y direction, but it is not very efficient (it makes a very big arc before entering)
- `rdv_trajectory_rot` is the first trajectory that entered a rotating corridor
- `rdv_trajectory_rot2` also entered the rotating corridor, but using a simpler reward function: closing bubble, penalize control effort, +2 reward for entering the corridor, no collision penalty (see `demo` on colab). I think I rewarded the corridor entry too much, because the agent tries to get to the corridor as fast as possible.
- `rdv_trajectory_rot3`: this is the first trajectory where I used the corridor axis as part of the observed state. I used the bubble reward function. Took about 250k timesteps to consistently achieve successful rendezvous. 

## Evaluations files:
Each `evaluations_#.npz` file corresponds to the evaluations recorded while training one of the models. (except for `evaluations_5M.npz`, that just shows how inefficient the learning is during 5M time-steps)