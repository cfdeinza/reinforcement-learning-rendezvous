import gym
from gym import spaces
import numpy as np
from scipy.integrate import solve_ivp
# from scipy.spatial.transform import Rotation as scipyRot
from utils.quaternions import quat2mat  # , put_scalar_last
from utils.general import clohessy_wiltshire, dydt, normalize_value, angle_between_vectors


class RendezvousEnv(gym.Env):

    def __init__(self, rc0=None, vc0=None, qc0=None, wc0=None, qt0=None, wt0=None):

        # State variables:  (initialized on reset() call)
        # self.state = None   # state of the system [r_c, v_c, q_c, w_c, q_t] (17,)
        self.rc = None      # position of the chaser [m] expressed in LVLH frame
        self.vc = None      # velocity of the chaser [m/s] expressed in LVLH frame
        self.qc = None      # attitude of the chaser [-]
        self.wc = None      # rotation rate of the chaser [rad/s] expressed in chaser body frame
        self.qt = None      # attitude of the target [-]
        self.wt = None      # rotation rate of the target [rad/s] expressed in target body frame

        # Nominal initial state:
        self.nominal_rc0 = np.array([0., -20., 0.]) if rc0 is None else rc0
        self.nominal_vc0 = np.array([0., 0., 0.]) if vc0 is None else vc0
        self.nominal_qc0 = np.array([1., 0., 0., 0.]) if qc0 is None else qc0
        self.nominal_wc0 = np.array([0., 0., 0.]) if wc0 is None else wc0
        self.nominal_qt0 = np.array([1., 0., 0., 0.]) if qt0 is None else qt0
        self.nominal_wt0 = np.array([0., 0., np.radians(3)]) if wt0 is None else wt0

        # Chaser properties:
        self.capture_axis = np.array([0, 1, 0])  # capture axis expressed in the chaser body frame (constant)
        self.inertia = np.array([  # moment of ineria [kg.m^2]
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3]
        ])
        self.inv_inertia = np.linalg.inv(self.inertia)
        self.max_delta_v = 1                # Maximum delta V for each axis [m/s]
        self.max_delta_w = np.radians(10)   # Maximum delta omega for each axis [rad/s]
        # self.max_torque = 5                 # Maximum torque for each axis [N.m]

        # Chaser state limits:
        self.max_axial_distance = 100   # maximum distance allowed on each axis [m]
        self.max_axial_speed = 10       # maximum speed allowed on each axis [m]
        self.max_rotation_rate = np.pi  # maximum chaser rotation rate [rad/s]

        # Target properties:
        self.koz_radius = 5                         # radius of the keep-out zone [m]
        self.corridor_half_angle = np.radians(30)   # half-angle of the conic entry corridor [rad]
        self.corridor_axis = np.array([0, -1, 0])   # entry corridor expressed in the target body frame (constant)
        self.inertia_target = np.array([            # moment of inertia of the target [kg.m^2]
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        self.inv_inertia_target = np.linalg.inv(self.inertia_target)

        # Capture conditions:
        self.rd = np.array([0, -2, 0])      # desired capture position [m] expressed in target body frame
        self.max_rd_error = 0.5             # allowed error for the capture position [m]
        self.max_vd_error = 0.1             # allowed error for the relative capture velocity [m/s]
        self.max_qd_error = np.radians(5)   # allowed error for the capture attitude [rad]
        self.max_wd_error = np.radians(1)   # allowed error for the relative capture rotation rate [rad/s]
        self.collided = None                # whether or not the chaser has collided during the current episode

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

        # delta_v = action[0:3] * self.max_delta_v
        # torque = action[3:] * self.max_torque

        delta_v = np.matmul(quat2mat(self.qc), action[0:3] * self.max_delta_v)  # expressed in LVLH frame
        delta_w = action[3:] * self.max_delta_w                                 # expressed in the body frame

        # Apply the delta V:
        # print(f't = {self.t}, vc: {self.vc}')
        # print(f'dV: {delta_v}, {delta_v.dtype}')
        # self.vc += delta_v

        # Update the position and velocity of the chaser:
        vc_after_impulse = self.vc + delta_v
        self.rc, self.vc = clohessy_wiltshire(self.rc, vc_after_impulse, self.n, self.dt)

        # wc_dot = angular_acceleration(self.inertia, self.inv_inertia, self.wc, torque)
        # qc_dot = quat_derivative(self.qc, self.wc)

        # Update the attitude and rotation rate of the chaser:
        self.wc = self.wc + delta_w
        self.integrate_chaser_attitude(np.array([0, 0, 0]))
        # self.integrate_chaser_attitude(torque)

        # Update the attitude and rotation rate of the target:
        self.integrate_target_attitude(np.array([0, 0, 0]))

        # Check for collision:
        if not self.collided:
            self.collided = self.check_collision()

        # Update the time step:
        self.t = round(self.t + self.dt, 3)  # rounding to prevent issues when dt is a decimal

        # Create observation: (normalize all values except quaternions)
        obs = self.get_observation()

        # Calculate reward:
        rew = self.get_reward()

        # Check if episode is done:
        done = self.get_done_condition(obs)

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

        self.collided = self.check_collision()
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
        return obs.astype(self.observation_space.dtype)

    def get_state(self):
        """
        Get the state of the system (only used for debugging right now).\n
        :return: state
        """
        return np.hstack((self.rc, self.vc, self.qc, self.wc, self.qt))

    def get_reward(self):
        """
        Calculate the reward for the current timestep. Maybe use a quadratic penalty for velocity and control effort.\n
        :return: reward
        """

        assert self.rc is not None

        goal_pos = np.matmul(quat2mat(self.qt), self.rd)                # Goal position in the LVLH frame [m]
        goal_vel = np.cross(self.wt, goal_pos)                          # Goal velocity in the LVLH frame [m/s]
        pos_error = np.linalg.norm(self.rc - goal_pos)                  # Position error [m]
        vel_error = np.linalg.norm(self.vc - goal_vel)                  # Velocity error [m/s]
        capture_axis = np.matmul(quat2mat(self.qc), self.capture_axis)  # capture axis expressed in LVLH
        att_error = angle_between_vectors(-self.rc, capture_axis)       # Attitude error [rad]
        rot_error = np.linalg.norm(self.wc - self.wt)                   # Rotation rate error [rad/s]

        rew = 0

        # Coefficients:
        c_r = 2     # position-based reward coefficient
        c_v = 0     # velocity-based reward coefficient
        c_q = 1     # attitude-based reward coefficient
        c_w = 0     # rotation-based reward coefficient

        reverse_sigmoid = 1 / (1 + np.e**(5 * pos_error - 10))

        rew += c_r * (1 - pos_error / self.max_axial_distance)
        rew += c_v * (1 - vel_error / self.max_axial_speed) * reverse_sigmoid
        rew += c_q * (1 - att_error / np.pi)
        rew += c_w * (1 - rot_error / self.max_rotation_rate) * reverse_sigmoid

        # Distance-based reward:
        # rew += (self.max_axial_distance - np.linalg.norm(pos_error)) / self.max_axial_distance
        # Attitude-based reward:
        # rew += 1e-1 * (np.pi - att_error) / np.pi

        # Success bonus:
        errors = np.array([
            pos_error,
            vel_error,
            att_error,
            rot_error,
        ])

        ranges = np.array([
            self.max_rd_error,
            self.max_vd_error,
            self.max_qd_error,
            self.max_wd_error,
        ])

        if not self.collided:
            if np.all(errors < ranges):
                rew += 1000
            else:
                threshold = self.max_rd_error * 10
                if pos_error < threshold:
                    rew += 10 * (1 - pos_error / threshold)

        # SciPy method:
        # rot = scipyRot.from_quat(put_scalar_last(self.qc))
        # chaser_neg_y = rot.apply(np.array([0, -1, 0]))
        # print(f'reward: {rew}')

        return rew

    def get_done_condition(self, obs) -> bool:
        """
        Check if the episode is done.\n
        :param obs: current observation
        :return: done condition
        """

        assert self.t is not None

        if (not self.observation_space.contains(obs)) or (self.t >= self.t_max):
            done = True
            print(f'Episode done. dist = {round(np.linalg.norm(self.rc), 2)} at t = {self.t}')
        else:
            done = False

        return done

    def check_collision(self):
        """
        Check if the chaser is within the keep-out zone.\n
        :return: True if there is a collision, otherwise False.
        """

        collided = False

        # Check distance to target:
        if np.linalg.norm(self.rc) < self.koz_radius:

            # Check angle between chaser and entry corridor:
            angle_to_corridor = angle_between_vectors(self.rc, np.matmul(quat2mat(self.qt), self.corridor_axis))
            if angle_to_corridor > self.corridor_half_angle:
                collided = True

        return collided

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


if __name__ == "__main__":
    env = RendezvousEnv()
    env.reset()
    # d = vars(env)  # create a dictionary of the object
    # print(type(d))
    # print(d)
    finished = False
    while not finished:
        observation, reward, finished, information = env.step(np.ones((6,)) * 1e-2)
