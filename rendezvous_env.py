import gym
from gym import spaces
import numpy as np
from scipy.integrate import solve_ivp
from utils.general import clohessy_wiltshire, dydt, normalize_value  # angular_acceleration, quat_derivative


class RendezvousEnv(gym.Env):

    def __init__(self, rc0=None, vc0=None, qc0=None, wc0=None, qt0=None, wt0=None):

        # State variables:  (initialized on reset() call)
        # self.state = None   # state of the system [r_c, v_c, q_c, w_c, q_t] (17,)
        self.rc = None      # position of the chaser [m]
        self.vc = None      # velocity of the chaser [m/s]
        self.qc = None      # attitude of the chaser [-]
        self.wc = None      # rotation rate of the chaser [rad/s]
        self.qt = None      # attitude of the target [-]
        self.wt = None      # rotation rate of the target [rad/s] (not included in observation)

        # Nominal initial state:
        self.nominal_rc0 = np.array([0, -40, 0]) if rc0 is None else rc0
        self.nominal_vc0 = np.array([0, 0, 0]) if vc0 is None else vc0
        self.nominal_qc0 = np.array([1, 0, 0, 0]) if qc0 is None else qc0
        self.nominal_wc0 = np.array([0, 0, 0]) if wc0 is None else wc0
        self.nominal_qt0 = np.array([1, 0, 0, 0]) if qt0 is None else qt0
        self.nominal_wt0 = np.array([0, 0, 0]) if wt0 is None else wt0

        # Chaser properties:
        self.inertia = np.array([  # moment of ineria [kg.m^2]
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3]
        ])
        self.inv_inertia = np.linalg.inv(self.inertia)
        self.max_delta_v = 1    # Maximum delta V for each axis [m/s]
        self.max_torque = 5     # Maximum torque for each axis [N.m]

        # Chaser state limits:
        self.max_axial_distance = 100   # maximum distance allowed on each axis [m]
        self.max_axial_speed = 10       # maximum speed allowed on each axis [m]
        self.max_rotation_rate = np.pi  # maximum chaser rotation rate [rad/s]

        # Target properties:
        self.koz_radius = 5                         # radius of the keep-out zone [m]
        self.corridor_half_angle = np.radians(30)   # half-angle of the conic entry corridor [rad]
        self.inertia_target = np.array([            # moment of inertia of the target [kg.m^2]
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        self.inv_inertia_target = np.linalg.inv(self.inertia_target)

        # Time:
        self.t = None       # current time of episode [s]
        self.dt = 1         # interval between timesteps [s]
        self.t_max = 180    # maximum length of episode [s]

        # Orbit properties:
        self.mu = 3.986004418e14                    # Gravitational parameter of Earth [m^3/s^2]
        self.Re = 6371e3                            # Radius of the Earth [m]
        self.h = 800e3                              # Altitude of the orbit [m]
        self.ro = self.Re + self.h                  # Radius of the orbit [m]
        self.n = np.sqrt(self.mu / self.ro ** 3)    # Mean motion of the orbit [rad/s]

        # Viewer:
        self.viewer = None

        # Observation space:
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(17,)
        )

        # Action space:
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(6,)
        )

        return

    def step(self, action: np.ndarray):
        """
        Propagate the state of the system one step forward in time.\n
        :param action: action selected by the controller.
        :return: observation, reward, done, info
        """

        assert action.shape == (6,)

        delta_v = action[0:3]  # TODO: rotate the delta V based on chaser attitude
        torque = action[3:]

        # Apply the delta V:
        self.vc += delta_v

        # Update the position and velocity of the chaser:
        self.rc, self.vc = clohessy_wiltshire(self.rc, self.vc, self.n, self.dt)

        # wc_dot = angular_acceleration(self.inertia, self.inv_inertia, self.wc, torque)
        # qc_dot = quat_derivative(self.qc, self.wc)

        # Update the attitude and rotation rate of the chaser:
        self.integrate_chaser_attitude(torque)

        # Update the attitude and rotation rate of the target:
        self.integrate_target_attitude(np.array([0, 0, 0]))

        # Update the time step:
        self.t = round(self.t + self.dt, 3)  # rounding to prevent issues when dt is a decimal

        # Create observation: (normalize all values except quaternions)
        obs = self.get_observation()

        # Calculate reward:
        rew = self.get_reward()

        # Check if episode is done:
        done = self.get_done_condition()

        # Info: auxiliary information
        info = {
            'observation': obs,
            'reward': rew,
            'done': done,
            'action': action,
        }

        return obs, rew, done, info

    def reset(self):
        """
        Initialize episode (chaser, target, time)
        :return: observation
        """

        # TODO: randomize initial conditions

        self.rc = self.nominal_rc0
        self.vc = self.nominal_vc0
        self.qc = self.nominal_qc0
        self.wc = self.nominal_wc0
        self.qt = self.nominal_qt0
        self.wt = self.nominal_wt0

        self.t = 0

        obs = self.get_observation()

        return obs

    def render(self, mode="human"):
        """
        Renders the environment.\n
        :param mode: "human" renders the environment to the current display
        :return:
        """

        return

    def close(self):
        """
        Close the viewer.\n
        :return: None
        """

        if self.viewer:
            self.viewer.close()
            self.viewer = None

        return

    """=============================================== Utils: ======================================================="""
    def get_observation(self):
        """
        Get the current observation of the system.
        This is done by normalizing all of the values, except for the quaternions.\n
        :return: the observation
        """

        assert self.rc is not None

        obs = np.hstack((
            normalize_value(self.rc, -self.max_axial_distance, self.max_axial_distance),
            normalize_value(self.vc, -self.max_axial_speed, self.max_axial_speed),
            self.qc,
            normalize_value(self.wc, -self.max_rotation_rate, self.max_rotation_rate),
            self.qt,
        ))
        return obs

    def get_reward(self):
        """
        Calculate the reward for the current timestep.\n
        :return: reward
        """

        assert self.rc is not None

        rew = (self.max_axial_distance - np.linalg.norm(self.rc)) / self.max_axial_distance
        return rew

    def get_done_condition(self) -> bool:
        """
        Check if the episode is done.\n
        :return: done condition
        """

        assert self.t is not None

        if self.t >= self.t_max:
            done = True
        else:
            done = False

        return done

    def integrate_chaser_attitude(self, torque):
        """
        Use a fourth-order Runge-Kutta integrator to update the attitude and rotation rate of the chaser.\n
        :param torque: torque applied at the start of this time step
        :return: None
        """

        y0 = np.append(self.qc, self.wc)

        sol = solve_ivp(
            fun=dydt,
            t_span=(0, self.dt),
            y0=y0,
            method='RK45',
            t_eval=np.array([self.dt]),
            rtol=1e-7,
            atol=1e-6,
            args=(self.inertia, self.inv_inertia, torque)
        )

        yf = sol.y.flatten()
        self.qc = yf[0:4]
        self.qc = self.qc / np.linalg.norm(self.qc)
        self.wc = yf[4:]

        # assert np.linalg.norm(self.qc) == 1, f'Quaternion magnitude != 1 after integration. {np.linalg.norm(self.qc)}'

        return

    def integrate_target_attitude(self, torque):
        """
        Use a fourth-order Runge-Kutta integrator to update the attitude and rotation rate of the target.\n
        :param torque: Torque applied to the target (always zero).
        :return: None
        """

        y0 = np.append(self.qt, self.wt)

        sol = solve_ivp(
            fun=dydt,
            t_span=(0, self.dt),
            y0=y0,
            method='RK45',
            t_eval=np.array([self.dt]),
            rtol=1e-7,
            atol=1e-6,
            args=(self.inertia_target, self.inv_inertia_target, torque)
        )

        yf = sol.y.flatten()
        self.qt = yf[0:4]
        self.qt = self.qt / np.linalg.norm(self.qt)
        self.wt = yf[4:]

        return
