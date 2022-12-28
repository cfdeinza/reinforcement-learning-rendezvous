import gym
import numpy as np
from scipy.integrate import solve_ivp
from utils.dynamics import state_derivative
from utils.quaternions import quat2mat
from utils.general import normalize_value, angle_between_vectors
from gym import spaces


class NewEnv(gym.Env):
    """
    This environment is basically the same as RendezvousEnv, but with more accurate dynamics.\n
    """

    def __init__(self, config: dict):
        # State             Parameter                   Unit        Reference frame
        self.rc = None      # Position of the chaser    [m]         [expressed in LVLH frame]
        self.vc = None      # Velocity of the chaser    [m/s]       [expressed in LVLH frame]
        self.qc = None      # Attitude of the chaser    [-]         [relative to LVLH frame]
        self.wc = None      # Rot rate of the chaser    [rad/s]     [expressed in C frame]
        self.qt = None      # Attitude of the chaser    [-]         [relative to LVLH frame]
        self.wt = None      # Rot rate of the target    [rad/s]     [expressed in T frame]

        # Chaser properties:  Parameter                   Unit        Reference frame
        self.mc = None      # Mass of the chaser        [kg]        [-]
        self.dc = None      # Length of chaser side     [m]         [-]
        self.ic = None      # Chaser mom of inertia     [kg.m^2]    [expressed in C frame]
        self.ic_inv = None  # Inverse of I_C
        self.capture_axis = np.array([0, 1, 0])  # capture axis expressed in the chaser body frame (constant)

        # Target properties:
        self.mt = None      # Mass of the target        [kg]        [-]
        self.dt = None      # Length of target side     [m]         [-]
        self.it = None      # Target mom of inertia     [kg.m^2]    [expressed in T frame]
        self.it_inv = None  # Inverse of I_T
        self.corridor_axis = np.array([0, -1, 0])  # entry corridor expressed in the target body frame (constant)
        self.corridor_half_angle = config.get("corridor_half_angle", np.radians(30))  # [rad]
        self.koz_radius = config.get("koz_radius", 5)  # Radius of the KOZ [m]

        # Control properties:
        self.thrust = 10    # Thrust along each axis    [N]
        self.torque = 0.2   # Torque along each axis    [N.m]
        self.bt = 0.2       # Thruster burn time        [s]
        self.dt = 1         # Duration of each time step [s]
        self.coast_time = round(self.dt - self.bt, 3)

        # Orbit properties:
        self.mu = 3.986004418e14    # Gravitational parameter of Earth [m^3/s^2]
        self.Re = 6371e3            # Radius of the Earth [m]
        self.h = 800e3              # Altitude of the orbit [m]
        self.ro = self.Re + self.h  # Radius of the orbit [m]
        self.n = np.sqrt(self.mu / self.ro ** 3)  # Mean motion of the orbit [rad/s]

        # Time:
        self.t = None
        self.t_max = config.get("t_max", 120)

        # Nominal initial conditions:
        self.nominal_rc0 = config.get("rc0", np.array([0., -10., 0.]))
        self.nominal_vc0 = config.get("vc0", np.array([0., 0., 0.]))
        self.nominal_qc0 = config.get("qt0", np.array([1., 0., 0., 0.]))
        self.nominal_wc0 = config.get("wc0", np.array([0., 0., 0.]))
        self.nominal_qt0 = config.get("qt0", np.array([1., 0., 0., 0.]))
        self.nominal_wt0 = config.get("wt0", np.array([0., 0., 0.]))

        # Range of initial conditions: (initial conditions are sampled uniformly from this range)
        self.rc0_range = config.get("rc0_range", 1)                 # 1   [m]
        self.vc0_range = config.get("vc0_range", 0.1)               # 0.1 [m/s]
        self.qc0_range = config.get("qc0_range", np.radians(1))     # 5   [deg]
        self.wc0_range = config.get("wc0_range", np.radians(0.1))   # 3   [deg/s]
        self.qt0_range = config.get("qt0_range", np.radians(45))    # 45  [deg]
        self.wt0_range = config.get("wt0_range", np.radians(3))     # 3   [deg/s]

        # Capture conditions:
        self.rd = np.array([0, -2, 0])  # desired capture position [m] expressed in target body frame
        self.max_rd_error = 0.5  # allowed error for the capture position [m]
        self.max_vd_error = 0.1  # allowed error for the relative capture velocity [m/s]
        self.max_qd_error = np.radians(5)  # allowed error for the capture attitude [rad]
        self.max_wd_error = np.radians(1)  # allowed error for the relative capture rotation rate [rad/s]
        self.collided = None  # whether or not the chaser has collided during the current episode
        self.success = None

        # Chaser state limits:
        self.max_axial_distance = np.linalg.norm(self.nominal_rc0) + 10  # maximum distance allowed on each axis [m]
        self.max_axial_speed = 10  # maximum speed allowed on each axis [m]
        self.max_rotation_rate = np.pi  # maximum chaser rotation rate [rad/s]
        self.max_attitude_error = np.radians(30)  # maximum allowed pointing error (episode ends)

        # Reward:
        self.collision_coef = config.get("collision_coef", 0.5)  # Coefficient to penalize collisions
        self.bonus_coef = config.get("bonus_coef", 8)  # Coefficient to reward success
        self.att_coef = config.get("att_coef", 2)  # Coefficient to reward attitude
        self.prev_potential = None

        # Other:
        self.viewer = None
        self.quiet = config.get("quiet", False)

        # Observation space:
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(17,)
        )

        # Action space:
        self.action_space = spaces.MultiDiscrete([3, 3, 3, 3, 3, 3])

        # Sanity checks:
        # Check the shape of the state variables:
        assert self.nominal_rc0.shape == (3,), f"Incorrect shape for chaser position {self.nominal_rc0.shape}"
        assert self.nominal_vc0.shape == (3,), f"Incorrect shape for chaser velocity {self.nominal_vc0.shape}"
        assert self.nominal_qc0.shape == (4,), f"Incorrect shape for chaser attitude {self.nominal_qc0.shape}"
        assert self.nominal_wc0.shape == (3,), f"Incorrect shape for chaser rot rate {self.nominal_wc0.shape}"
        assert self.nominal_qt0.shape == (4,), f"Incorrect shape for target attitude {self.nominal_qt0.shape}"
        assert self.nominal_wt0.shape == (3,), f"Incorrect shape for target rot rate {self.nominal_wt0.shape}"
        # Check that the desired terminal position is within the corridor:
        assert np.linalg.norm(self.rd) < self.koz_radius, "Error: terminal position lies outside corridor."
        assert np.linalg.norm(self.rd) - self.max_rd_error > 0, "Error: position constraint allows collisions"

        return

    def step(self, action: np.ndarray):

        # Process the action:
        processed_action = action - 1
        force = processed_action[0:3]
        torque = processed_action[3:]

        # Execute the action:
        self.integrate_state(force, torque, time=self.bt)
        # Coast:
        if self.coast_time > 0:
            self.integrate_state(force=np.array([0, 0, 0]), torque=np.array([0, 0, 0]), time=self.coast_time)

        # Update params:
        self.t = round(self.t + self.dt, 3)  # rounding to prevent issues when dt is a decimal
        if not self.collided:
            self.collided = self.check_collision()
            self.success += int(self.check_success())

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
        self.rc = np.array([0, -10, 0])
        self.vc = np.array([0, 0, 0.1])
        self.qc = np.array([1, 0, 0, 0])
        self.wc = np.array([0, 0, 0])
        self.qt = np.array([1, 0, 0, 0])
        self.wt = np.array([0, 0, np.radians(1)])
        self.mc = 100
        self.dc = 1
        self.ic = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 1/12 * self.mc * (2 * self.dc**2)
        self.ic_inv = np.linalg.inv(self.ic)
        self.mt = 100
        self.dt = 1
        self.it = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 1/12 * self.mt * (2 * self.dt**2)
        self.it_inv = np.linalg.inv(self.it)

        self.t = 0
        self.collided = self.check_collision()
        self.success = int(self.check_success())
        self.prev_potential = self.get_potential()

        obs = self.get_observation()
        return obs

    def render(self, mode="human"):
        """
        Renders the environment. Currently not used.\n
        :param mode: "human" renders the environment to the current display
        :return:
        """

        return

    def close(self):
        """
        Close the viewer. Currently not used.\n
        :return: None
        """

        if self.viewer:
            self.viewer.close()
            self.viewer = None

        return

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

    def get_reward(self):

        phi = self.get_potential()
        rew = phi - self.prev_potential
        self.prev_potential = phi

        return rew

    def get_potential(self):
        """
        Get the potential of the current state.\n
        :return:
        """

        dist = np.linalg.norm(self.rc)

        if self.check_collision():
            # Collided:
            phi = (1 - self.koz_radius / self.max_axial_distance) ** 2 - (self.koz_radius - dist)
        else:
            phi = (1 - max(dist, np.linalg.norm(self.rd)) / self.max_axial_distance) ** 2

        # Success bonus:
        if np.linalg.norm(self.rc) < self.koz_radius and not self.collided:

            goal_pos = self.target2lvlh(self.rd)
            pos_error = self.get_pos_error(goal_pos)
            if pos_error < self.max_rd_error:
                phi += 2 - pos_error / self.max_rd_error
                att_error = self.get_attitude_error()
                if att_error < self.max_qd_error:
                    phi += 2 - att_error / self.max_qd_error

        return phi

    def get_done_condition(self, obs) -> bool:
        """
        Check if the episode is done.\n
        :param obs: current observation
        :return: done condition
        """

        assert self.t is not None

        dist = np.linalg.norm(self.rc)

        end_conditions = [
            not self.observation_space.contains(obs),               # observation outside of observation space
            self.t >= self.t_max,                                   # episode time limit reached
            self.get_attitude_error() > self.max_attitude_error,    # chaser attitude error too large
        ]

        # (not self.observation_space.contains(obs)) or (self.t >= self.t_max) or (dist > self.bubble_radius)
        if any(end_conditions):
            done = True
            if not self.quiet:
                end_reasons = ['obs', 'time', 'attitude']  # reasons corresponding to end_conditions
                print("Episode end" +
                      " | r = " + str(round(dist, 2)).rjust(5) +                    # chaser position at episode end
                      " | t = " + str(self.t).rjust(4) +                            # time of episode end
                      " | " + end_reasons[end_conditions.index(True)].center(8) +   # reason for episode end
                      " | " + ("Collided" if self.collided else " "))               # whether chaser entered KOZ
        else:
            done = False

        return done

    def get_pos_error(self, goal_position):
        """
        Compute the position error (the distance between the chaser and the goal position).\n
        :param goal_position: position of the goal [m] (expressed in LVLH)
        :return: position error [m]
        """
        return np.linalg.norm(self.rc - goal_position)

    def get_attitude_error(self):
        """
        Compute the chaser's attitude error. The chaser should always be pointing in the direction of the target's
        center of mass.\n
        :return: angle between the actual and desired pointing direction (in radians)
        """

        capture_axis = self.chaser2lvlh(self.capture_axis)      # capture axis of the chaser expressed in LVLH
        error = angle_between_vectors(-self.rc, capture_axis)   # Attitude error [rad]

        return error

    def get_errors(self) -> np.ndarray:
        """
        Compute the position error, velocity error, attitude error, and rotational velocity error.
        (All relative to the desired terminal conditions).\n
        :return: numpy array containing the errors
        """

        wc_lvlh = self.chaser2lvlh(self.wc)             # Chaser rot rate in the LVLH frame [rad/s]
        wt_lvlh = self.target2lvlh(self.wt)             # Target rot rate in the LVLH frame [rad/s]
        rd_lvlh = self.target2lvlh(self.rd)             # Goal position in the LVLH frame [m]
        vd_lvlh = np.cross(wt_lvlh, rd_lvlh)            # Goal velocity in the LVLH frame [m/s]

        pos_error = self.get_pos_error(rd_lvlh)         # Position error (magnitude) [m]
        vel_error = np.linalg.norm(self.vc - vd_lvlh)   # Velocity error (magnitude) [m/s]
        att_error = self.get_attitude_error()           # Attitude error (magnitude) [rad]
        rot_error = np.linalg.norm(wc_lvlh - wt_lvlh)   # Rot rate err. (mag) [rad/s]

        return np.array([pos_error, vel_error, att_error, rot_error])

    def check_collision(self):
        """
        Check if the chaser is within the keep-out zone.\n
        :return: True if there is a collision, otherwise False.
        """

        collided = False

        # Check distance to target:
        if np.linalg.norm(self.rc) < self.koz_radius:

            # Check angle between chaser and entry corridor:
            angle_to_corridor = angle_between_vectors(self.rc, self.target2lvlh(self.corridor_axis))
            if angle_to_corridor > self.corridor_half_angle:
                collided = True

        return collided

    def check_success(self) -> bool:
        """
        Check if the chaser has achieved the terminal conditions without entering the KOZ.\n
        :return: True if the terminal conditions are met, otherwise False
        """

        if self.collided:
            success = False
        else:
            errors = self.get_errors()
            error_ranges = np.array([self.max_rd_error, self.max_vd_error, self.max_qd_error, self.max_wd_error])
            if np.all(errors <= error_ranges):
                success = True
            else:
                success = False

        return success

    def chaser2lvlh(self, vec: np.ndarray) -> np.ndarray:
        """
        Convert a vector from the chaser body (CB) frame into the LVLH frame.\n
        :param vec: 3D vector expressed in the CB frame
        :return: same vector expressed in the LVLH frame
        """
        assert self.qc is not None, "Cannot convert vector from chaser to LVLH frame. Chaser attitude is undefined."
        assert vec.shape == (3,), f"Vector must have a (3,) shape, but has {vec.shape}"
        return np.matmul(quat2mat(self.qc), vec)

    def target2lvlh(self, vec: np.ndarray) -> np.ndarray:
        """
        Convert a vector from the Target Body (TB) frame into the LVLH frame.\n
        :param vec: 3D vector expressed in the TB frame
        :return: same vector expressed in the LVLH frame
        """
        assert self.qt is not None, "Cannot convert vector from target to LVLH frame. Target attitude is undefined."
        assert vec.shape == (3,), f"Vector must have a (3,) shape, but has {vec.shape}"
        return np.matmul(quat2mat(self.qt), vec)

    def integrate_state(self, force, torque, time):

        y0 = np.concatenate((self.rc, self.vc, self.qc, self.wc, self.qt, self.wt))

        sol = solve_ivp(
            fun=state_derivative,
            t_span=(0, time),
            y0=y0,
            method='RK45',
            t_eval=np.array([time]),
            rtol=1e-7,
            atol=1e-6,
            args=(self.n, self.mc, self.ic, self.ic_inv, self.it, self.it_inv, force, torque),
        )

        yf = sol.y.flatten()
        self.rc = yf[0:3]
        self.vc = yf[3:6]
        self.qc = yf[6:10]
        self.wc = yf[10:13]
        self.qt = yf[13:17]
        self.wt = yf[17:]
        self.qc = self.qc / np.linalg.norm(self.qc)
        self.qt = self.qt / np.linalg.norm(self.qt)
        return


if __name__ == "__main__":
    env_config = dict()
    sys = NewEnv(env_config)
    sys.reset()
