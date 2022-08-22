import gym
from gym import error, spaces, utils
# from gym.utils import seeding
import numpy as np
from math import sin, cos, acos, degrees, radians, pi, sqrt, atan2
# from gym.envs.classic_control import rendering
from scipy.spatial.transform import Rotation as R
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection, Line3D
import plotly.graph_objects as go

"""
This file contains the custom environment used to train reinforcement learning models. 
The environment was created following the guidelines specified by OpenAI Gym.
The environment is defined as a class:
    -Attitude() simulates an attitude control system.

Written by C. F. De Inza Niemeijer
"""


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
        obs = self.state.astype(self.observation_space.dtype)

        # Reward:
        q_error = self.error_quaternion()  # Error quaternion (the rotation btw the current and desired attitude)
        rew = 8 - np.sum(np.abs(q_error - np.array([0, 0, 0, 1])))

        # Done:
        # print(f'in obs_space: {self.observation_space.contains(obs)}')
        if (not self.observation_space.contains(obs)) or (self.t >= self.t_max):
            done = True

        else:
            done = False

        # Info: auxiliary information
        info = {
            'state': self.state,
            'reward': rew,
            't': self.t
        }

        self.render()

        if done:
            # Render the episode:
            self.fig.frames = tuple(self.ax)
            self.fig.show()
            # Print the final errors:
            print(f"Final angle errors: {self.angle_error()} deg")
            # Remove the first row of NaNs from the actions and trajectory arrays
            self.trajectory = self.trajectory[:, 1:]
            self.actions = self.actions[:, 1:]
            info['trajectory'] = self.trajectory
            info['actions'] = self.actions
            # print(len(self.fig.frames))
            # print(f't: {self.t}')

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
