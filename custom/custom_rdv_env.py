"""
This file contains the custom environment used to train reinforcement learning models. 
The environment was created following the guidelines specified by OpenAI Gym.
The environment is defined as a class:
    -Rendezvous3DOF() simulates the rendezvous of a chaser and a passive target.

Written by C. F. De Inza Niemeijer
"""

import gym
from gym import spaces
import numpy as np
from math import sin, cos, radians, sqrt
from gym.envs.classic_control import rendering
from scipy.spatial.transform import Rotation as scipyRot
from collections import OrderedDict
from utils.general import angle_between_vectors, perpendicular_vector, rotate_vector_about_axis
# from ray.rllib.env.env_context import EnvContext


class Rendezvous3DOF(gym.Env):

    """
    This environment simulates a 3-DOF rendezvous scenario of a chaser with a tumbling target.
    The state vector contains the position and velocity of the chaser and the entry corridor axis.
    Both the action space and the observation space are continuous.
    The dynamics are modeled with the Clohessy Wiltshire equations,
    and the thrust is represented as an instantaneous change in velocity.
    The initial state of the chaser is within a certain range of a hold-point.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, pos0=None, vel0=None, corr0=None, randomize=True):
        # add "config: EnvContext" param to train rllib (evaluating still does not work)

        # The ideal starting state of the system (randomness is introduced in reset()):
        self.ideal_start_pos = np.array([0, -40, 0]) if pos0 is None else pos0
        self.ideal_start_vel = np.array([0, 0, 0]) if vel0 is None else vel0
        self.ideal_start_corridor = np.array([0, -1, 0]) if corr0 is None else corr0
        self.randomize_start = randomize  # Whether or not to randomize the initial conditions
        self.state = None       # Placeholder until the state is defined in reset()
        self.pos = None         # Chaser position (m)
        self.vel = None         # Chaser velocity (m)
        self.collided = None    # boolean that shows if the chaser has collided with the target during the current ep.
        self.t = None           # Current time step. (Placeholder until reset() is called)
        self.dt = 1             # Time interval between actions (s)
        self.t_max = 180        # Max time per episode

        # Properties of the reward function:
        self.bubble_radius = None           # Radius of the virtual bubble that determines the allowed space (m)
        self.bubble_decrease_rate = 0.3     # Rate at which the size of the bubble decreases (m/s)

        # Orbit properties:
        self.mu = 3.986004418e14            # Gravitational parameter of Earth (m3/s2)
        self.Re = 6371e3                    # Radius of the Earth (m)
        self.h = 800e3                      # Altitude of the orbit (m)
        self.r = self.Re + self.h           # Radius of the orbit (m)
        self.n = sqrt(self.mu / self.r**3)  # Mean motion of the orbit (rad/s)

        # Action and state constraints:
        self.max_delta_v = 1                            # Maximum delta V for each axis (m/s)
        self.max_axial_distance = np.ones((3,)) * 100   # Maximum distance for each axis (m)
        self.max_axial_speed = np.ones((3,)) * 10       # Maximum velocity for each axis (m/s)

        # Properties of the target:
        self.target_radius = 5                          # Radius of the target (m)
        self.cone_half_angle = radians(30)              # Half-angle of the entry corridor (rad)
        self.w_norm = np.array([0, 0, 1])               # Normalized angular velocity of the target (unitless)
        self.w_mag = radians(3)                         # Magnitude of the angular velocity (rad/s)
        self.corridor_axis = None                       # Direction of the corridor axis

        # Range of initial conditions:
        self.initial_position_range = 2             # Range of possible initial positions for each axis (m)
        self.initial_velocity_range = 0.1           # Range of possible initial velocities for each axis (m/s)
        self.initial_corridor_range = radians(30)   # Range of initial corridor direction (radians)

        # Attributes used in render():
        self.trajectory = None  # Array of states (6xn)
        self.actions = None     # Array of actions (3xn)
        self.viewer = None      # Viewer window
        self.chaser_transform = None  # Chaser transform (Not sure if it needs to be here)
        self.viewer_bounds = max(np.abs(self.ideal_start_pos)) \
            + self.initial_position_range/2  # Max distance shown in the viewer (m)

        # Action space:
        self.action_space = spaces.Box(
            low=np.float32(np.ones((3,)) * -self.max_delta_v),
            high=np.float32(np.ones((3,)) * self.max_delta_v),
            shape=(3,)
        )
        # self.action_space = spaces.MultiDiscrete([3, 3, 3])
        # self.action_space = spaces.MultiBinary(6)

        # Observation space: pos (3,), vel (3,), corridor_axis (3,)
        obs_limit = np.array([1] * 9)  # Limits for the normalized observation space

        self.observation_space = spaces.Box(
            low=np.float32(-obs_limit),
            high=np.float32(obs_limit),
            shape=(9,)
        )

        # Dictionary observation space:
        # self.observation_space = spaces.Dict({
        #     'pos': spaces.Box(low=-1, high=1, shape=(3,)),
        #     'vel': spaces.Box(low=-1, high=1, shape=(3,)),
        #     'cor': spaces.Box(low=-1, high=1, shape=(3,))
        # })

        return

    def step(self, action: np.ndarray):
        """
        Propagate the state of the system one step forward in time (t -> t+1).\n
        :param action: Action taken at time t. Shape: (3,).
        :return: Observation, Reward, Done, Info
        """

        # Clip the action? (It might already be done automatically)
        # action = np.clip(action, self.action_space.low, self.action_space.high)

        # The action causes an immediate change in the chaser's velocity:
        # action = (action - 1) * self.max_delta_v  # for multi-discrete action space
        # action = np.rint(action * 1 / self.max_delta_v) * self.max_delta_v  # to turn continuous thrust into on/off
        # action = (action[0:3] - action[3:]) * self.max_delta_v  # for multi-binary action space
        vel_after_impulse = self.vel + action

        # Compute the new state:
        self.pos, self.vel = self.clohessy_wiltshire(self.dt, np.concatenate((self.pos, vel_after_impulse)))

        # Compute the new corridor orientation:
        euler_axis = self.w_norm
        theta = self.w_mag * self.dt
        quaternion = np.append(euler_axis * np.sin(theta/2), np.cos(theta/2))  # Rotation quaternion
        rotation = scipyRot.from_quat(quaternion)
        self.corridor_axis = rotation.apply(self.corridor_axis)

        self.state = np.concatenate((self.pos, self.vel, self.corridor_axis))

        # Save the new values:
        self.trajectory = np.append(
            self.trajectory,
            self.state.reshape((len(self.state), 1)),
            axis=1)
        self.actions = np.append(
            self.actions,
            action.reshape((action.shape[0], 1)),
            axis=1)

        # Update the time step:
        self.t += self.dt

        # Update the bubble:
        if self.bubble_radius > self.target_radius:
            self.bubble_radius -= self.bubble_decrease_rate

        # Observation: the observation is equal to the normalized state
        # obs = self.normalized_state().astype(self.observation_space.dtype)
        obs = self.get_current_observation().astype(self.observation_space.dtype)

        # Done condition:
        dist = self.absolute_position()
        vel = self.absolute_velocity()

        done = False

        if (not self.observation_space.contains(obs)) or (self.t >= self.t_max) or (dist > self.bubble_radius):
            done = True

        # Reward:

        # Fuel efficiency:
        rew = 1 - np.linalg.norm(action) / sqrt(3 * self.max_delta_v**2)  # 1 - ( |u| / u_max )

        if dist < self.target_radius:

            # Compute angle between the corridor axis and the chaser:
            angle_from_corridor = angle_between_vectors(self.pos, self.corridor_axis)

            # Inside corridor:
            if angle_from_corridor < self.cone_half_angle:
                if (dist < 3) and (self.collided is False):
                    vel_radial = abs(np.dot(self.vel, self.pos) / dist)
                    vel_tangential = np.sqrt(vel ** 2 - vel_radial ** 2)
                    # Success:
                    if (vel_radial < 0.1) and (vel_tangential < self.w_mag * self.target_radius):
                        # done = True
                        rew += 10  # 500
                        # print('Success!')
            # Collided:
            else:
                self.collided = True
                rew -= 0.5

        # Info: auxiliary information
        info = {
            'Distance': self.absolute_position(),
            'Speed': self.absolute_velocity(),
            'Last action': action
        }

        if done:
            print(f"Episode end. "
                  f"dist: {str(round(info['Distance'], 1)).rjust(5, ' ')} m at "
                  f"t = {str(self.t).rjust(3, ' ')} "
                  f"{' (Collided) ' if self.collided else ''}")
            # Save the array of states and actions in 'info' (this is used for the eval_callback)
            info['trajectory'] = self.trajectory[:, 1:]
            info['actions'] = self.actions[:, 1:]

        return obs, rew, done, info

    def reset(self):
        """
        Reset the environment to a random state near the ideal starting position.\n
        :return: An observation of the state
        """
        random_initial_position = np.random.uniform(
            low=np.ones((3,)) * -self.initial_position_range / 2,
            high=np.ones((3,)) * self.initial_position_range / 2,
            size=(3,)
        )
        random_initial_velocity = np.random.uniform(
            low=np.ones((3,)) * -self.initial_velocity_range / 2,
            high=np.ones((3,)) * self.initial_velocity_range / 2,
            size=(3,)
        )

        # random_initial_corridor = np.array([0, 0, 0])
        # random_initial_corridor = np.array([
        #     np.random.uniform(low=-0.5, high=0.5),
        #     np.random.uniform(low=-1, high=0),
        #     0
        # ])
        # random_initial_corridor = random_initial_corridor / np.linalg.norm(random_initial_corridor)  # normalize
        self.corridor_axis = self.ideal_start_corridor
        if self.randomize_start:
            self.randomize_corridor()

        self.pos = self.ideal_start_pos + random_initial_position
        self.vel = self.ideal_start_vel + random_initial_velocity
        self.state = np.concatenate((self.pos, self.vel, self.corridor_axis))
        self.trajectory = np.zeros((self.state.shape[0], 1)) * np.nan
        self.actions = np.zeros((self.action_space.shape[0], 1)) * np.nan
        self.t = 0  # Reset time steps
        self.collided = False
        self.bubble_radius = self.absolute_position() + self.target_radius

        # obs = self.normalized_state()
        obs = self.get_current_observation().astype(self.observation_space.dtype)
        return obs

    def render(self, mode='human'):
        """
        Renders the environment.\n
        :param mode: "human" renders the environment to the current display
        :return:
        """
        screen_width = 700
        screen_height = 700

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            lim = self.viewer_bounds
            self.viewer.set_bounds(-lim, lim, -lim, lim)  # left, right, bottom, top

            # Draw a grid:
            self.draw_grid([-lim, lim])

            # Make a circle to represent the chaser:
            chaser = rendering.make_circle(
                radius=1,  # Radius of the circle
                res=6,  # Resolution (number of points)
                filled=True  # Filled or empty
            )
            chaser.set_color(0.9, 0.4, 0.4)
            self.chaser_transform = rendering.Transform()
            chaser.add_attr(self.chaser_transform)
            self.viewer.add_geom(chaser)

            # Make another circle for the target:
            target = rendering.make_circle(
                radius=self.target_radius,
                res=30,
                filled=False
            )
            target.set_color(0.4, 0.4, 0.9)
            target.set_linewidth(3)
            self.viewer.add_geom(target)

        # Use a poligon to show the corridor:
        self.draw_corridor()

        # Use a polyline to show the path of the chaser:
        path = rendering.make_polyline(list(zip(self.trajectory[0, 1:], self.trajectory[1, 1:])))
        path.set_color(0.5, 0.5, 0.5)
        self.viewer.add_onetime(path)

        # Draw arrows to show the actions taken by the agent:
        self.draw_arrow(self.actions[0, -1], 'horizontal', length=30)
        self.draw_arrow(self.actions[1, -1], 'vertical', length=30)

        # Update the chaser position:
        pos_x = self.pos[0]
        pos_y = self.pos[1]
        self.chaser_transform.set_translation(pos_x, pos_y)  # new_x, new_y

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        """
        Close the viewer.\n
        :return:
        """

        if self.viewer:
            self.viewer.close()
            self.viewer = None

        return

    """ ======================================== Rendezvous3DOF Utils: ========================================= """

    def randomize_corridor(self):
        """
        Rotates the corridor by some random angle.
        :return: None
        """

        y = np.array([0, 0, 1])  # y axis
        z = np.array([0, 0, 1])  # z axis

        # Compute a vector perpendicular to the corridor axis:
        if 0.1 < angle_between_vectors(self.corridor_axis, z) < np.pi-0.1:
            perp = perpendicular_vector(self.corridor_axis, z)
        else:
            perp = perpendicular_vector(self.corridor_axis, y)

        # Rotate the perpendicular vector around the corridor axis by some random degrees:
        perp = rotate_vector_about_axis(
            vec=perp,
            axis=self.corridor_axis,
            theta=np.random.uniform(
                low=0,
                high=2 * np.pi
            )
        )

        # Rotate the corridor around the perpendicular vector:
        self.corridor_axis = rotate_vector_about_axis(
            vec=self.corridor_axis,
            axis=perp,
            theta=np.random.uniform(
                low=0,
                high=self.initial_corridor_range
            )
        )

        return

    def draw_grid(self, limits):
        """
        Draw a grid to be rendered in the viewer.\n
        :param limits: A list containing the upper and lower limits of the viewer
        :return:
        """

        lower, upper = limits
        amount = max(3, round((upper - lower) / 10) + 1)  # Amount of grid lines (minimum 3)
        values = np.linspace(lower, upper, amount)

        for x in values:
            line = rendering.Line(start=(x, lower), end=(x, upper))
            if x != 0:
                line.set_color(0.5, 0.5, 0.5)
            else:
                line.set_color(0, 0, 0)
            self.viewer.add_geom(line)

        for y in values:
            line = rendering.Line(start=(lower, y), end=(upper, y))
            if y != 0:
                line.set_color(0.5, 0.5, 0.5)
            else:
                line.set_color(0, 0, 0)
            self.viewer.add_geom(line)

        return

    def draw_arrow(self, action, direction, length=10):
        """
        Draw an arrow on the viewer to represent the action taken by the agent.\n
        :param action: the action taken by the agent
        :param direction: either "vertical" or "horizontal"
        :param length: length scale for the arrow
        :return:
        """

        x0, y0 = self.pos[0:2]
        # inc = length * np.sign(action)
        # inc = np.sign(action) * ((high - low)*abs(action) + 4)

        if direction == 'horizontal':
            inc = length * action  # / self.action_space.high[0]
            # norm_action = action / self.action_space.high[0]
            # inc = length
            x_points = [x0, x0 + inc, x0 + inc*0.9, x0 + inc, x0 + inc*0.9]
            y_points = [y0, y0, y0 + inc*0.05, y0, y0 - inc*0.05]
        elif direction == 'vertical':
            inc = length * action  # / self.action_space.high[1]
            # norm_action = action / self.action_space.high[1]
            # inc = length
            x_points = [x0, x0, x0 - inc*0.05, x0, x0 + inc*0.05]
            y_points = [y0, y0 + inc, y0 + inc*0.9, y0 + inc, y0 + inc*0.9]
        else:
            x_points, y_points = None, None
            # norm_action = None

        arrow = rendering.make_polyline(list(zip(x_points, y_points)))
        arrow.set_color(0, 0, 0)
        self.viewer.add_onetime(arrow)

        return

    def draw_corridor(self):
        """
        Draw a polygon on the viewer to represent the entry corridor.\n
        :return:
        """
        corr = self.corridor_axis * self.target_radius
        perp = np.cross(corr, np.array([0, 0, 1]))
        perp = perp / np.linalg.norm(perp) * self.target_radius * np.tan(self.cone_half_angle)
        x = [0, corr[0] + perp[0], corr[0] - perp[0], 0]
        y = [0, corr[1] + perp[1], corr[1] - perp[1], 0]
        corridor = rendering.make_polygon(list(zip(x, y)), filled=False)
        corridor.set_color(0.4, 0.4, 0.9)
        corridor.set_linewidth(2)
        self.viewer.add_onetime(corridor)
        return

    def absolute_position(self):
        """
        Compute the magnitude of the chaser's position (i.e. distance from the origin).\n
        :return: pos
        """

        pos = np.linalg.norm(self.pos)

        return pos

    def absolute_velocity(self):
        """
        Compute the magnitude of the chaser's velocity w.r.t the target.\n
        :return: vel
        """

        vel = np.linalg.norm(self.vel)

        return vel

    def normalized_pos(self, custom_range=None) -> np.ndarray:
        """
        Returns the position coordinates of the chaser, normalized to values between -1 and 1 (or a custom range).\n
        :param custom_range: The desired range [a, b] for the normalized state. Default is [-1, 1]
        :return: Normalized position coordinates
        """

        if custom_range is None:
            custom_range = [-1, 1]

        a = custom_range[0]     # Lowest possible value in normalized range
        b = custom_range[1]     # Highest possible value in normalized range
        pos = self.pos          # Chaser position
        norm_pos = a + (pos + self.max_axial_distance)*(b-a) / (2*self.max_axial_distance)
        return norm_pos

    def normalized_vel(self, custom_range=None) -> np.ndarray:

        """
        Returns the velocity components of the chaser, normalized to values between -1 and 1 (or a custom range).\n
        :param custom_range: The desired range [a, b] for the normalized state. Default is [-1, 1]
        :return: Normalized velocity components
        """

        if custom_range is None:
            custom_range = [-1, 1]

        a = custom_range[0]     # Lowest possible value in normalized range
        b = custom_range[1]     # Highest possible value in normalized range
        vel = self.vel          # Chaser velocity
        norm_vel = a + (vel + self.max_axial_speed)*(b-a) / (2*self.max_axial_speed)
        return norm_vel

    def get_current_observation(self):
        """
        Create an observation based on the current state. Works for Box and Dict observation spaces.\n
        :return: observation
        """

        if isinstance(self.observation_space, gym.spaces.box.Box):
            obs = np.concatenate((self.normalized_pos(), self.normalized_vel(), self.corridor_axis))
        elif isinstance(self.observation_space, gym.spaces.dict.Dict):
            obs = OrderedDict([
                ('pos', self.normalized_pos()),
                ('vel', self.normalized_vel()),
                ('cor', self.corridor_axis)
            ])
        else:
            raise TypeError(f'Observation space type is incompatible: {type(self.observation_space)}')

        return obs

    def clohessy_wiltshire(self, t, x0):
        """
        Uses the Clohessy-Wiltshire model to find the state at time t,
        given an initial state (x0) and a time length (t).\n
        :param t: Time duration
        :param x0: Initial position and velocity (6,)
        :return: State at time t
        """

        n = self.n  # Mean motion (rad/s)
        nt = n * t  # Product of mean motion and time interval (rad)

        # State transition matrix:
        stm = np.array([
            [4 - 3 * cos(nt), 0, 0, 1 / n * sin(nt), 2 / n * (1 - cos(nt)), 0],
            [6 * (sin(nt) - nt), 1, 0, 2 / n * (cos(nt) - 1), 1 / n * (4 * sin(nt) - 3 * nt), 0],
            [0, 0, cos(nt), 0, 0, 1 / n * sin(nt)],
            [3 * n * sin(nt), 0, 0, cos(nt), 2 * sin(nt), 0],
            [6 * n * (cos(nt) - 1), 0, 0, -2 * sin(nt), 4 * cos(nt) - 3, 0],
            [0, 0, -n * sin(nt), 0, 0, cos(nt)]
        ])

        # State at time t:
        xt = np.matmul(stm, x0)

        return xt[0:3], xt[3:]
