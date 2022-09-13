# Data files:
Description of the trajectory on each `data` file:
- `rdv_data_bubble`: the response obtained using the bubble model. It learns to enter the corridor much quicker.
- `rdv_data_cont0` & `...cont1`: show the response of a controller trained with a very basic reward function.
- `rdv_data_racket`: shows the tennis racket (or Dzhanibekov) effect as the chaser rotates about its intermediate axis of inertia.
- `rdv_data_rot+mov`: shows the motion of the chaser when given a single thrust input in the y direction and torque about the xyz direction.
- `rdv_data_rot`: shows the rotation of the chaser, first about one axis, and then another.
- `rdv_data_discrete`: performance obtained after 200k steps of training using a MultiDiscrete action space.