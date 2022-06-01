import gym
from gym import spaces
import numpy as np
from math import sin, cos, radians, sqrt
from gym.envs.classic_control import rendering
from scipy.spatial.transform import Rotation as scipyRot

"""
This file contains the custom environment used to train reinforcement learning models. 
The environment was created following the guidelines specified by OpenAI Gym.
The environment is defined as a class:
    -Rendezvous3DOF() simulates the rendezvous of a chaser and a passive target.

Written by C. F. De Inza Niemeijer
"""


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

    def __init__(self):

        # The ideal starting state of the system (randomness is introduced in reset()):
        self.ideal_start_pos = np.array([0, -40, 0])
        self.ideal_start_vel = np.array([0, 0, 0])
        self.ideal_start_corridor = np.array([0, -1, 0])
        self.state = None       # Placeholder until the state is defined in reset()
        self.pos = None         # Chaser position (m)
        self.vel = None         # Chaser velocity (m)
        self.collided = None    # boolean that shows if the chaser has collided with the target during the current ep.
        self.t = None           # Current time step. (Placeholder until reset() is called)
        self.dt = 1             # Time interval between actions (s)
        self.t_max = 120        # Max time per episode
        self.bubble_radius = None
        self.bubble_decrease_rate = 0.3  # Give the target enough time to turn once?

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

        # Range of initial conditions for the chaser:
        self.initial_position_range = 2     # Range of possible initial positions for each axis (m)
        self.initial_velocity_range = 0.1   # Range of possible initial velocities for each axis (m/s)

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

        # Observation space: pos (3,), vel (3,), corridor_axis (3,)
        # obs_limit = np.append(self.max_axial_distance, self.max_axial_speed)  # Not normalized!
        obs_limit = np.array([1] * 9)  # Limits for the normalized observation space

        self.observation_space = spaces.Box(
            low=np.float32(-obs_limit),
            high=np.float32(obs_limit),
            shape=(9,)
        )

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
        # action = (action - 1) * self.max_delta_v  # for discrete action space
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
            self.state.reshape((self.observation_space.shape[0], 1)),
            axis=1)
        self.actions = np.append(
            self.actions,
            action.reshape((self.action_space.shape[0], 1)),
            axis=1)

        # Update the time step:
        self.t += self.dt

        # Update the bubble:
        if self.bubble_radius > self.target_radius:
            self.bubble_radius -= self.bubble_decrease_rate

        # Observation: the observation is equal to the normalized state
        # obs = self.state
        obs = self.normalize_state().astype(self.observation_space.dtype)

        # Done condition:
        dist = self.absolute_position()
        vel = self.absolute_velocity()

        done = False
        if ~self.observation_space.contains(obs) or (self.t >= self.t_max) or dist > self.bubble_radius:
            done = True

        # Reward:

        # Fuel efficiency:
        rew = 1 - np.linalg.norm(action) / sqrt(3 * self.max_delta_v**2)  # 1 - ( |u| / u_max )

        if dist < self.target_radius:
            # Inside corridor:
            if self.angle_from_corridor() < self.cone_half_angle:
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

        random_initial_corridor = np.array([0, 0, 0])

        self.pos = self.ideal_start_pos + random_initial_position
        self.vel = self.ideal_start_vel + random_initial_velocity
        self.corridor_axis = self.ideal_start_corridor + random_initial_corridor  # TODO: Randomize initial corr axis
        self.state = np.concatenate((self.pos, self.vel, self.corridor_axis))
        self.trajectory = np.zeros((self.observation_space.shape[0], 1)) * np.nan
        self.actions = np.zeros((self.action_space.shape[0], 1)) * np.nan
        self.t = 0  # Reset time steps
        self.collided = False
        self.bubble_radius = self.absolute_position() + self.target_radius

        # obs = self.state
        obs = self.normalize_state()

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

    def angle_from_corridor(self) -> np.float:
        """
        Computes the angle between the chaser and the corridor axis, using the geometric definition of the dot product:
        cos(theta) = (u . v) / (|u|*|v|)\n
        :return: angle (in radians) between the chaser and the corridor axis
        """

        pos = self.pos              # Chaser position vector
        axis = self.corridor_axis   # Corridor vector

        # Apply the dot product formula:
        angle = np.arccos(np.dot(pos, axis) / (np.linalg.norm(pos) * np.linalg.norm(axis)))

        return angle

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

    def normalize_state(self, custom_range=None) -> np.ndarray:
        """
        Returns the state of the system normalized to values between -1 & 1 (or to some custom range).
        Rationale: According to the tips in SB3, you should "always normalize your observation space when you can,
        i.e. when you know the boundaries."\n
        :param custom_range: The desired range [a, b] for the normalized state. Default is [-1, 1]
        :return: The normalized state
        """

        if custom_range is None:
            custom_range = [-1, 1]

        a = custom_range[0]     # Lowest possible value in normalized range
        b = custom_range[1]     # Highest possible value in normalized range
        pos = self.pos          # Chaser position
        vel = self.vel          # Chaser velocity
        norm_pos = a + (pos + self.max_axial_distance)*(b-a) / (2*self.max_axial_distance)
        norm_vel = a + (vel + self.max_axial_speed)*(b-a) / (2*self.max_axial_speed)
        norm_state = np.concatenate((norm_pos, norm_vel, self.corridor_axis))
        # Corridor axis does not need normalization
        return norm_state

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
