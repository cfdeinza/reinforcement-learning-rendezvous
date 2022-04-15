import gym
from gym import error, spaces, utils
# from gym.utils import seeding
import numpy as np
from math import sin, cos, acos, degrees, radians, pi, sqrt, atan2
from gym.envs.classic_control import rendering
from scipy.spatial.transform import Rotation as R
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection, Line3D
import plotly.graph_objects as go

"""
This file contains the custom environments used to train reinforcement learning models. 
The environments were created following the guidelines specified by OpenAI Gym.
Each environment is defined as a class:
    -Rendezvous3DOF() simulates the rendezvous of a chaser and a passive target.
    -Attitude() simulates an attitude control system.

Written by C. F. De Inza Niemeijer
"""


class Rendezvous3DOF(gym.Env):

    """
    This environment simulates a 3-DOF rendezvous scenario.
    The state vector contains the position and velocity of the chaser w.r.t. the target.
    Both the action space and the observation space are continuous.
    The dynamics are modeled with the Clohessy Wiltshire equations,
    and the thrust is represented as an instantaneous change in velocity.
    The initial state of the chaser is within a certain range of a hold-point.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self):

        # The ideal starting state for the chaser (randomness is introduced in reset()):
        self.ideal_start = np.array([0, -40, 0, 0, 0, 0])
        self.state = None       # Placeholder until the state is defined in reset()
        self.collided = None    # boolean that shows if the chaser has collided with the target during the current ep.
        self.t = None           # Current time step. (Placeholder until reset() is called)
        self.dt = 1             # Time interval between actions (s)
        self.t_max = 300        # Max time per episode

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
        self.viewer_bounds = max(np.abs(self.ideal_start)) \
            + self.initial_position_range/2  # Max distance shown in the viewer (m)

        # Action space:
        self.action_space = spaces.Box(
            low=np.float32(np.ones((3,)) * -self.max_delta_v),
            high=np.float32(np.ones((3,)) * self.max_delta_v),
            shape=(3,)
        )

        # Observation space:
        # obs_limit = np.append(self.max_axial_distance, self.max_axial_speed)  # Not normalized!
        obs_limit = np.array([1, 1, 1, 1, 1, 1])  # Limits for the normalized observation space

        self.observation_space = spaces.Box(
            low=np.float32(-obs_limit),
            high=np.float32(obs_limit),
            shape=(6,)
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
        state_after_impulse = self.state + np.append(np.zeros((3,)), action)

        # Compute the new state:
        self.state = self.clohessy_wiltshire(self.dt, state_after_impulse)

        # Save the new values:
        self.trajectory = np.append(
            self.trajectory,
            self.state.reshape((self.observation_space.shape[0], 1)),
            axis=1)
        self.actions = np.append(
            self.actions,
            action.reshape((self.action_space.shape[0], 1)),
            axis=1)

        # Compute the new corridor orientation:
        euler_axis = self.w_norm
        theta = self.w_mag * self.dt
        quaternion = np.append(euler_axis * np.sin(theta/2), np.cos(theta/2))  # Rotation quaternion
        rotation = R.from_quat(quaternion)
        self.corridor_axis = rotation.apply(self.corridor_axis)

        # Update the time step:
        self.t += self.dt

        # Observation: the observation is equal to the normalized state
        # obs = self.state
        obs = self.normalize_state().astype(self.observation_space.dtype)

        # Reward: for now, the reward penalizes distance from the origin
        dist = self.absolute_position()
        vel = self.absolute_velocity()
        rew = max(100 - dist, 0) * 1e-2  # range: 0-1

        # Reward for staying close to the corridor axis at all times:
        # rew += (1 - self.angle_from_corridor()/pi) * 1e-1  # range: 0-0.1
        if self.angle_from_corridor() < self.cone_half_angle:
            rew += 0.1

        # Penalize collision:
        if (dist < self.target_radius) & (self.angle_from_corridor() > self.cone_half_angle):
            self.collided = True
            rew -= 0.1
        # rew = (1000 - dist)*1e-2 - np.linalg.norm(action)
        # rew = - np.linalg.norm(action)  # Should the action be part of the state?

        # Done: End the episode if the chaser leaves the observation state or if the max time is reached.
        if ~self.observation_space.contains(obs) or (self.t >= self.t_max):
            done = True
        # Constraint for successful rendezvous:
        elif (dist < 3) and (vel < 0.1) and (self.angle_from_corridor() < self.cone_half_angle) and \
                (self.collided is False):
            rew += 2 * (self.t_max - self.t)
            done = True
        else:
            done = False

        # Info: auxiliary information
        info = {
            'Distance': self.absolute_position(),
            'Speed': self.absolute_velocity(),
            'Last action': action
        }

        if done:
            print(f"Final distance: {round(info['Distance'], 2)} m at t = {self.t} "
                  f"{'(Collided)' if self.collided else ''}")
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

        random_state = np.append(random_initial_position, random_initial_velocity)

        self.state = self.ideal_start + random_state
        self.corridor_axis = np.array([0, -1, 0])  # TODO: Randomize initial corridor orientation?
        self.trajectory = np.zeros((self.observation_space.shape[0], 1)) * np.nan
        self.actions = np.zeros((self.action_space.shape[0], 1)) * np.nan
        self.t = 0  # Reset time steps
        self.collided = False

        # obs = self.state
        obs = self.normalize_state()

        return obs

    def render(self, mode='human'):
        """
        Renders the environment.\n
        :param mode: "human" renders the environment to the current display
        :return:
        """
        # TODO: switch to plotly?
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
        pos_x = self.state[0]
        pos_y = self.state[1]
        self.chaser_transform.set_translation(pos_x, pos_y)  # newx, newy

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

        pos = self.state[0:3]       # Chaser position vector
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
        vals = np.linspace(lower, upper, amount)

        for x in vals:
            line = rendering.Line(start=(x, lower), end=(x, upper))
            if x != 0:
                line.set_color(0.5, 0.5, 0.5)
            else:
                line.set_color(0, 0, 0)
            self.viewer.add_geom(line)

        for y in vals:
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

        x0, y0 = self.state[0:2]
        # inc = length * np.sign(action)
        # inc = np.sign(action) * ((high - low)*abs(action) + 4)

        if direction == 'horizontal':
            inc = length * action / self.action_space.high[0]
            # norm_action = action / self.action_space.high[0]
            # inc = length
            x_points = [x0, x0 + inc, x0 + inc*0.9, x0 + inc, x0 + inc*0.9]
            y_points = [y0, y0, y0 + inc*0.05, y0, y0 - inc*0.05]
        elif direction == 'vertical':
            inc = length * action / self.action_space.high[1]
            # norm_action = action / self.action_space.high[1]
            # inc = length
            x_points = [x0, x0, x0 - inc*0.05, x0, x0 + inc*0.05]
            y_points = [y0, y0 + inc, y0 + inc*0.9, y0 + inc, y0 + inc*0.9]
        else:
            x_points, y_points = None, None
            # norm_action = None

        arrow = rendering.make_polyline(list(zip(x_points, y_points)))
        arrow.set_color(0, 0, 0)
        # arrow.set_linewidth(2)
        self.viewer.add_onetime(arrow)

        return

    def absolute_position(self):
        """
        Compute the magnitude of the chaser's position (i.e. distance from the origin).\n
        :return: pos
        """
        pos = np.linalg.norm(self.state[0:3])

        return pos

    def absolute_velocity(self):
        """
        Compute the magnitude of the chaser's velocity w.r.t the target.\n
        :return: vel
        """
        vel = np.linalg.norm(self.state[3:])

        return vel

    def normalize_state(self, custom_range=None) -> np.ndarray:
        """
        Normalize the position and velocity of the chaser to values between -1 & 1 (or to some custom range).
        Rationale: According to the tips in SB3, you should "always normalize your observation space when you can,
        i.e. when you know the boundaries."\n
        :param custom_range: The desired range [a, b] for the normalized state. Default is [-1, 1]
        :return: The normalized state
        """

        if custom_range is None:
            custom_range = [-1, 1]

        a = custom_range[0]     # Lowest possible value in normalized range
        b = custom_range[1]     # Highest possible value in normalized range
        pos = self.state[0:3]   # Position
        vel = self.state[3:]    # Velocity
        norm_pos = a + (pos + self.max_axial_distance)*(b-a) / (2*self.max_axial_distance)
        norm_vel = a + (vel + self.max_axial_speed)*(b-a) / (2*self.max_axial_speed)
        norm_state = np.append(norm_pos, norm_vel)
        return norm_state

    def clohessy_wiltshire(self, t, x0):
        """
        Uses the Clohessy Wiltshire model to find the state at time t,
        given an initial state (x0) and a time length (t).\n
        :param t: Time duration
        :param x0: Initial state
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

        return xt


class Attitude(gym.Env):

    """
    This environment simulates the attitude control of a spacecraft.\n
    """
    # TODO: Use the error quaternion as the state?

    metadata = {'render.modes': ['human']}

    def __init__(self):

        # Initial conditions:  TODO: Introduce randomness during reset() call
        self.theta0 = radians(0)                        # Angle for initial quaternion
        self.e0 = np.array([0, 0, 1])                   # Euler axis for initial quaternion
        self.e0 = self.e0 / np.linalg.norm(self.e0)     # Normalize e0 just in case
        self.w0 = np.array([0, 0, 0])                   # Initial rotational rate (expressed in body frame)
        self.state = None                               # Placeholder until the state is defined in reset()
        self.goal = np.array([0, 0, sin(pi/4), cos(pi/4), 0, 0, 0])  # Desired state

        # Time:
        self.dt = 0.1       # Time interval between steps (s)
        self.t = None       # Current time step. (Placeholder until reset() is called)
        self.t_max = 30     # Max time per episode

        # Moment of inertia:  (kg.m2)
        self.inertia = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        self.inv_I = np.linalg.inv(self.inertia)

        # Action and state constraints:
        self.max_torque = 50                                     # Maximum torque on each axis (N.m)
        self.max_rotational_rate = np.ones((3,)) * radians(100)  # Maximum rotational rate for each axis

        # Attributes used in render():
        # self.viewer = None  # Viewer
        self.fig = None
        self.ax = None
        self.viewer_lim = 10  # Viewer limits
        self.trajectory = None  # Array of states (7xn)
        self.actions = None     # Array of actions (3xn)

        # Action space:
        self.action_space = spaces.Box(
            low=np.float32(np.ones((3,)) * -self.max_torque),
            high=np.float32(np.ones((3,)) * self.max_torque),
            shape=(3,)
        )

        # Observation space:
        obs_limit = np.append(np.array([1, 1, 1, 1]), self.max_rotational_rate)

        self.observation_space = spaces.Box(
            low=np.float32(-obs_limit),
            high=np.float32(obs_limit),
            shape=(7,)
        )

        return

    def step(self, action: np.ndarray):

        # Compute angular acceleration:
        w_dot = self.angular_acceleration(action)

        # Integrate orientation and angular velocity:
        q = self.integrate_quat()
        w = self.integrate_w_dot(w_dot)

        # Update the state:
        self.state = np.append(q, w)

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

        # Observation: the observation is equal to the state
        obs = self.state

        # Reward:
        q_error = self.error_quaternion()  # Error quaternion (the rotation btw the current and desired attitude)
        rew = 8 - np.sum(np.abs(q_error - np.array([0, 0, 0, 1])))

        # Done:
        if ~self.observation_space.contains(obs) or (self.t >= self.t_max):
            done = True

        else:
            done = False

        # Info: auxiliary information
        info = {
            'state': self.state,
            'reward': rew,
            't': self.t
        }

        if done:
            # Render the episode:
            self.fig.frames = tuple(self.ax)
            self.fig.show()
            # Print the final errors:
            print(f"Final angle errors: {self.angle_error()} deg")
            # Remove the first row of NaNs from the actions and trajectory arrays
            self.trajectory = self.trajectory[:, 1:]
            self.actions = self.actions[:, 1:]

        return obs, rew, done, info

    def reset(self):
        """
        Reset the environment to a random state near the ideal starting position.
        :return: An observation of the state
        """

        # Orientation:
        q = np.array([
            self.e0[0] * sin(self.theta0/2),
            self.e0[1] * sin(self.theta0/2),
            self.e0[2] * sin(self.theta0/2),
            cos(self.theta0/2)
        ])

        self.state = np.append(q, self.w0)
        self.trajectory = np.zeros((self.observation_space.shape[0], 1)) * np.nan
        self.actions = np.zeros((self.action_space.shape[0], 1)) * np.nan
        self.t = 0  # Reset time steps
        self.fig = None
        self.ax = None

        obs = self.state

        return obs

    def render(self, mode='human'):
        """
        Renders the environment.
        :param mode: "human" renders the environment to the current display
        :return:
        """

        if self.fig is None:
            self.make_fig_plotly()
            self.ax = []  # These are the frames
        frame = {"data": self.plotly_data(), "name": str(self.t)}
        self.ax.append(frame)

        return

    def close(self):
        """
        Close the viewer.
        :return:
        """

        if self.fig:
            self.fig = None
            self.ax = None

        return

    """ =========================================== Attitude Utils: ============================================ """

    def angular_acceleration(self, torque: np.ndarray):
        """
        Compute the angular acceleration of a rigid body using Euler's rotational equation of motion.
        :param torque: Body frame torque (3,)
        :return: Body frame angular acceleration (3,)
        """
        w = self.state[4:]
        w_dot = np.matmul(self.inv_I, (torque - np.cross(w, np.matmul(self.inertia, w))))

        return w_dot

    def integrate_w_dot(self, w_dot: np.ndarray):
        """Obtain the next angular velocity by using forward integration."""

        w = self.state[4:]

        w_new = w + w_dot*self.dt

        return w_new

    def integrate_quat(self):
        """Obtain the next orientation quaternion by using forward integration."""

        q = self.state[0:4]  # q(n-1)

        dq = self.quat_derivative()  # Time derivative of q(n-1)

        q_new = q + dq*self.dt  # q(n)

        q_new = q_new / np.linalg.norm(q_new)  # Normalize q(n)

        return q_new

    def quat_derivative(self):
        """
        Compute the time-derivative of a quaternion, using the formula shown in AE4313 Notes 2 (slide 50).
        :return: Derivative of the quaternion (4,)
        """

        q_vec = self.state[0:3]                             # Vector part of the quaternion
        q_scalar = self.state[3]
        w = self.state[4:]                                  # Angular velocity (rad/s)

        dq_vec = 0.5 * (q_scalar * w + np.cross(q_vec, w))  # Derivative of the vector part

        dq_scalar = -0.5 * np.dot(q_vec, w)                 # Derivative of the scalar part

        dq = np.append(dq_vec, dq_scalar)                   # Full derivative

        return dq

    def error_quaternion(self) -> np.ndarray:
        """
        Compute the error quaternion (the difference between the current orientation and the desired orientation).
        :return: Error quaternion (4,) (scalar-last format)
        """

        q = self.state[0:4]                                         # Current orientation
        q_des = self.goal[0:4]                                      # Desired orientation

        # Compute the inverse of the desired orientation quaternion:
        q_des_conjugate = np.append(-q_des[0:3], q_des[-1])         # Conjugate
        q_des_inverse = q_des_conjugate / np.linalg.norm(q_des)**2  # Inverse

        # To compute the error quaternion, we use the "Rotation" class from the SciPy package.
        # Alternatively, we could also use quaternion multiplication.
        q_e = R.from_quat(q) * R.from_quat(q_des_inverse)

        return q_e.as_quat()

    def make_fig_plotly(self):
        # Set the camera settings:
        camera = dict(
            up=dict(x=0, y=0, z=1),             # Determines the 'up' direction on the page.
            center=dict(x=0, y=0, z=0),         # The projection of the center point lies at the center of the view.
            eye=dict(x=1.25, y=-1.25, z=1.25)   # The eye vector determines the position of the camera.
        )

        lim = self.viewer_lim

        fig_dict = {
            'data': self.plotly_data(),
            'layout': {
                'scene': dict(
                    xaxis=dict(nticks=8, range=[-lim, lim], zerolinecolor="black"),
                    yaxis=dict(nticks=8, range=[-lim, lim], zerolinecolor="black"),
                    zaxis=dict(nticks=8, range=[-lim, lim], zerolinecolor="black"),
                    xaxis_showspikes=False,
                    yaxis_showspikes=False,
                    zaxis_showspikes=False
                ),
                'width': 800,
                # 'margin': dict(r=20, l=10, b=10, t=10),
                'scene_aspectmode': 'cube',
                'scene_camera': camera,
                'title': 'Rotation',
                'updatemenus': [
                    {
                        "buttons": [
                            {
                                "args": [None, {"frame": {"duration": 15},
                                                'mode': 'immediate',
                                                "fromcurrent": True,
                                                "transition": {"duration": 0}}],
                                "label": "Play",
                                "method": "animate"
                            },
                            {
                                "args": [[None], {"frame": {"duration": 0},
                                                  "mode": "immediate",
                                                  "transition": {"duration": 0}}],
                                "label": "Pause",
                                "method": "animate"
                            }
                        ],
                        'type': 'buttons'
                    }
                ]
            },
            'frames': []
        }
        self.fig = go.Figure(fig_dict)
        return

    def plotly_data(self):

        # Inertial axes:
        len_i = self.viewer_lim * 0.8   # Length of the inertial axes
        width_i = 4                     # Width of the body axes
        ms = 1                          # Marker size
        x_i = go.Scatter3d(
            x=[0, len_i],
            y=[0, 0],
            z=[0, 0],
            line={'color': 'darkred', 'dash': 'solid', 'width': width_i},
            marker={'size': ms},
            name='x_inertial')
        y_i = go.Scatter3d(
            x=[0, 0],
            y=[0, len_i],
            z=[0, 0],
            line={'color': 'darkgreen', 'dash': 'solid', 'width': width_i},
            marker={'size': ms},
            name='y_inertial')
        z_i = go.Scatter3d(
            x=[0, 0],
            y=[0, 0],
            z=[0, len_i],
            line={'color': 'darkblue', 'dash': 'solid', 'width': width_i},
            marker={'size': ms},
            name='z_inertial')

        # Body axes:
        len_b = self.viewer_lim/2  # Length of the body axes
        width_b = 8  # Width of the body axes
        vecs = np.array([
            [len_b, 0, 0],
            [0, len_b, 0],
            [0, 0, len_b]
        ])
        rotation = R.from_quat(self.state[0:4])
        rot_vecs = rotation.apply(vecs)  # Rotated vectors
        x_b = go.Scatter3d(
            x=[0, rot_vecs[0, 0]],
            y=[0, rot_vecs[0, 1]],
            z=[0, rot_vecs[0, 2]],
            line={'color': 'red', 'dash': 'solid', 'width': width_b},
            marker={'size': ms},
            name='x_body')
        y_b = go.Scatter3d(
            x=[0, rot_vecs[1, 0]],
            y=[0, rot_vecs[1, 1]],
            z=[0, rot_vecs[1, 2]],
            line={'color': 'green', 'dash': 'solid', 'width': width_b},
            marker={'size': ms},
            name='y_body')
        z_b = go.Scatter3d(
            x=[0, rot_vecs[2, 0]],
            y=[0, rot_vecs[2, 1]],
            z=[0, rot_vecs[2, 2]],
            line={'color': 'blue', 'dash': 'solid', 'width': width_b},
            marker={'size': ms},
            name='z_body')

        # Angular velocity:
        len_w = 100
        w = self.state[4:]
        w_i = rotation.apply(w)  # Angular velocity expressed in the inertial frame
        vel = go.Scatter3d(
            x=[0, w_i[0] * len_w],
            y=[0, w_i[1] * len_w],
            z=[0, w_i[2] * len_w],
            line={'color': 'black', 'dash': 'solid', 'width': 10},
            marker={'size': 1},
            name='angular velocity')
        data_list = [x_i, y_i, z_i, x_b, y_b, z_b, vel]
        return data_list

    def angle_error(self):
        """
        For each axis (x, y, z) compute the angle between the current direction and the desired direction.
        The angles are computed using the cosine formula: cos(theta) = (u . v) / (|u|*|v|)
        :return: List containing 3 angles (in degrees)
        """

        q = R.from_quat(self.state[0:4])  # Current quaternion
        qd = R.from_quat(self.goal[0:4])  # Desired quaternion

        # Array containing unit vectors x, y and z
        axes = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]])

        # Rotate the axes:
        current = q.apply(axes)
        desired = qd.apply(axes)

        # For each axis, compute the angle between the current and desired directions
        errors = []
        for i in range(axes.shape[0]):
            vec1 = current[i]
            vec2 = desired[i]
            # Apply the cosine formula:
            angle = acos(np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2)))
            errors.append(round(degrees(angle), 2))

        return errors

    # def quat_inverse(self, q: np.ndarray):
    #     conjugate = np.append(-q[0:3], q[-1])
    #     norm = np.linalg.norm(q)
    #     inverse = conjugate / norm**2
    #     return inverse

    # def inertial2body(self, w_i: np.ndarray):
    #
    #     rot = Rotation.from_quat(self.state[0:4])
    #
    #     w_b = rot.apply(w_i)
    #
    #     return w_b
    #
    # def body2inertial(self):
    #
    #     rot = Rotation.from_quat(self.state[0:4])
    #     w_b = self.state[4:]
    #
    #     w_i = rot.apply(w_b, inverse=True)
    #
    #     return w_i
    #
    # def w_from_quat(self, q, dq):
    #     """
    #     Compute the angular velocity in terms of the quaternion.
    #     Based on the formula used in AE4313 Notes 2 (slide 49).
    #     :param q: Quaternion in scalar-last form (4,)
    #     :param dq: Time-derivative of the quaternion (4,)
    #     :return: Angular velocity (3,)
    #     """
    #
    #     q1, q2, q3, q4 = q          # Individual components of the quaternion
    #
    #     dq1, dq2, dq3, dq4 = dq     # Individual components of the derivative of the quaternion
    #
    #     w1 = 2*(dq1*q4 + dq2*q3 - dq3*q2 - dq4*q1)
    #     w2 = 2*(dq2*q4 + dq3*q1 - dq1*q3 - dq4*q2)
    #     w3 = 2*(dq3*q4 + dq1*q2 - dq2*q1 - dq4*q3)
    #
    #     w = np.array([w1, w2, w3])  # Angular velocity
    #
    #     return w
    #
    # def quat_multiply(self, q1: np.ndarray, q2: np.ndarray):
    #     """
    #     Multiply two quaternions.
    #     :param q1: First quaternion in scalar-last form.
    #     :param q2: Second quaternion in scalar-last form.
    #     :return: Product of the two quaternions.
    #     """
    #     v1, s1 = q1[0:3], q1[-1:]
    #     v2, s2 = q2[0:3], q2[-1:]
    #
    #     product = np.append(
    #         s1*v2 + s2*v1 + np.cross(v1, v2),
    #         s1*s2 - np.dot(v1, v2)
    #     )
    #     normalized_product = product / np.linalg.norm(product)
    #
    #     return normalized_product
