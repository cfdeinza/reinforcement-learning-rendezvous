import gym
import numpy as np
from scipy.integrate import solve_ivp
from utils.dynamics import state_derivative
from utils.quaternions import quat2mat, rot2quat, quat_product
from utils.general import normalize_value, angle_between_vectors, random_unit_vector, rotate_vector_about_axis
from gym import spaces


class NewEnv(gym.Env):
    """
    This environment is basically the same as RendezvousEnv, but with more accurate dynamics.\n
    """
    # TODO: Introduce navigation noise
    # TODO: Introduce actuator errors
    # TODO: Try max_reward
    # TODO: Try bubble_reward() as potential

    def __init__(self, config: dict):
        # State             Parameter                   Unit        Reference frame
        self.rc = None      # Position of the chaser    [m]         [expressed in LVLH frame]
        self.vc = None      # Velocity of the chaser    [m/s]       [expressed in LVLH frame]
        self.qc = None      # Attitude of the chaser    [-]         [orientation of C relative to LVLH]
        self.wc = None      # Rot rate of the chaser    [rad/s]     [expressed in C frame]
        self.qt = None      # Attitude of the chaser    [-]         [orientation of T relative to LVLH]
        self.wt = None      # Rot rate of the target    [rad/s]     [expressed in T frame]

        # Chaser properties:  Parameter                   Unit        Reference frame
        self.mc = None      # Mass of the chaser        [kg]        [-]
        self.lc = None      # Length of chaser side     [m]         [-]
        self.ic = None      # Chaser mom of inertia     [kg.m^2]    [expressed in C frame]
        self.ic_inv = None  # Inverse of I_C                        [expressed in C frame]
        self.capture_axis = np.array([0, 1, 0])  # capture axis expressed in the chaser body frame (constant)

        # Target properties:
        self.mt = None      # Mass of the target        [kg]        [-]
        self.lt = None      # Length of target side     [m]         [-]
        self.it = None      # Target mom of inertia     [kg.m^2]    [expressed in T frame]
        self.it_inv = None  # Inverse of I_T                        [expressed in T frame]
        self.corridor_axis = np.array([0, -1, 0])  # entry corridor expressed in the target body frame (constant)
        self.corridor_half_angle = config.get("corridor_half_angle", np.radians(30))  # [rad]
        self.koz_radius = config.get("koz_radius", 5)  # Radius of the KOZ [m]

        # Control properties:
        self.thrust = config.get("thrust", 10)      # Thrust along each axis    [N]
        self.torque = config.get("torque", 0.2)     # Torque along each axis    [N.m]
        self.bt = config.get("bt", 0.2)             # Thruster burn time        [s]

        # Orbit properties:
        self.mu = 3.986004418e14                    # Gravitational parameter of Earth [m^3/s^2]
        self.Re = 6371e3                            # Radius of the Earth [m]
        self.h = 800e3                              # Altitude of the orbit [m]
        self.ro = self.Re + self.h                  # Radius of the orbit [m]
        self.n = np.sqrt(self.mu / self.ro ** 3)    # Mean motion of the orbit [rad/s]

        # Time:
        self.t = None                                   # Time of current step [s]
        self.t_max = config.get("t_max", 120)           # Maximum length of each episode [s]
        self.dt = config.get("dt", 1)                   # Duration of each time step [s]
        self.coast_time = round(self.dt - self.bt, 3)   # Time between end of action and start of next action [s]

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
        self.consecutive_success = None
        self.required_success = 1  # How much (consecutive) success time is required to end the episode [s]
        self.successful_trajectory = None

        # Chaser state limits:
        self.max_axial_distance = np.linalg.norm(self.nominal_rc0) + 10  # maximum distance allowed on each axis [m]
        self.max_axial_speed = 10  # maximum speed allowed on each axis [m]
        self.max_rotation_rate = np.pi  # maximum chaser rotation rate [rad/s]
        self.max_attitude_error = np.radians(30)  # maximum allowed pointing error (episode ends)

        # Reward:
        self.collision_coef = config.get("collision_coef", 0.5)  # Coefficient to penalize collisions
        self.bonus_coef = config.get("bonus_coef", 8)  # Coefficient to reward success
        self.fuel_coef = config.get("fuel_coef", 0)  # Coefficient to penalize fuel use
        self.att_coef = config.get("att_coef", 0)  # 2Coefficient to reward attitude
        self.prev_potential = None
        self.total_delta_v = None
        self.total_delta_w = None
        self.bubble_radius = None  # Radius of the virtual bubble that determines the allowed space (m)
        self.initial_bubble_radius = self.max_axial_distance
        self.bubble_decrease_rate = 0.5 * self.dt  # Rate at which the size of the bubble decreases (m/s)
        self.total_rew = None
        self.max_rew = 5_000

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
        # self.action_space = spaces.MultiDiscrete([3, 3, 3, 3, 3, 3])
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(6,)
        )

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
        # processed_action = action - 1
        processed_action = action
        force = processed_action[0:3] * self.thrust
        torque = processed_action[3:] * self.torque

        # Execute the action:
        self.integrate_state(force, torque, time=self.bt)
        # Coast:
        if self.coast_time > 0:
            self.integrate_state(force=np.array([0, 0, 0]), torque=np.array([0, 0, 0]), time=self.coast_time)

        # Check collision:
        if not self.collided:
            self.collided = self.check_collision()
            # self.success += int(self.check_success())
            if self.check_success():
                self.success += 1
                self.consecutive_success += round(self.dt, 3)
            else:
                self.consecutive_success = 0  # Reset the consecutive success count when no success occurs

        # Update params:
        self.t = round(self.t + self.dt, 3)  # rounding to prevent issues when dt is a decimal
        self.total_delta_v += np.sum(np.abs(force)) / self.mc * self.bt
        self.total_delta_w += np.sum(np.abs(np.matmul(torque, self.ic_inv))) * self.bt
        if self.bubble_radius > self.koz_radius:
            self.bubble_radius -= self.bubble_decrease_rate

        # Create observation: (normalize all values except quaternions)
        obs = self.get_observation()

        # Check if episode is done:
        done = self.get_done_condition(obs)

        # Calculate reward:
        # rew = self.get_reward()
        if done and self.successful_trajectory:
            rew = self.max_rew - self.total_rew  # - self.fuel_coef * self.total_delta_v
        else:
            rew = self.get_bubble_reward(processed_action)
            self.total_rew += rew

        # Info: auxiliary information
        info = {
            'observation': obs,
            'reward': rew,
            'done': done,
            'action': action,
        }
        return obs, rew, done, info

    def reset(self):

        # Generate random deviations for the state of the system:
        rc0_deviation = random_unit_vector() * np.random.uniform(low=0, high=self.rc0_range)
        vc0_deviation = random_unit_vector() * np.random.uniform(low=0, high=self.vc0_range)
        qc_deviation = rot2quat(axis=random_unit_vector(), theta=np.random.uniform(0, self.qc0_range))
        wc0_deviation = random_unit_vector() * np.random.uniform(low=0, high=self.wc0_range)  # Expressed in LVLH
        qt_deviation = rot2quat(axis=random_unit_vector(), theta=np.random.uniform(0, self.qt0_range))
        wt0_deviation = random_unit_vector() * np.random.uniform(low=0, high=self.wt0_range)  # Expressed in LVLH

        # Apply the random deviations to the nominal initial conditions:
        self.rc = self.nominal_rc0 + rc0_deviation
        self.vc = self.nominal_vc0 + vc0_deviation
        self.qc = quat_product(self.nominal_qc0, qc_deviation)  # apply sequential rotation (first nominal, then dev)
        self.wc = self.lvlh2chaser(self.nominal_wc0 + wc0_deviation)  # convert wc0 from LVLH to chaser body frame
        self.qt = quat_product(self.nominal_qt0, qt_deviation)  # apply sequential rotation (first nominal, then dev)
        self.wt = self.lvlh2target(self.nominal_wt0 + wt0_deviation)  # convert wt0 from LVLH to target body frame

        # Mass parameters:
        self.mc = 100
        self.lc = 1
        self.ic = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 1/12 * self.mc * (2 * self.lc**2)
        self.ic_inv = np.linalg.inv(self.ic)
        self.mt = 100
        self.lt = 1
        self.it = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 1/12 * self.mt * (2 * self.lt**2)
        self.it_inv = np.linalg.inv(self.it)

        # Wait for the target to rotate to a convenient attitude:
        if np.linalg.norm(self.wt) > 0.002:
            self.wait_for_target()

        self.t = 0
        self.total_delta_v = 0
        self.total_delta_w = 0
        self.bubble_radius = self.initial_bubble_radius  # OR: np.linalg.norm(self.rc) + self.koz_radius
        self.total_rew = 0
        self.collided = self.check_collision()
        self.successful_trajectory = False
        if self.check_success():
            self.success = 1
            self.consecutive_success = round(self.dt, 3)
        else:
            self.success = 0
            self.consecutive_success = 0
        # self.prev_potential = self.get_potential()

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

    def get_bubble_reward(self, action):
        """
        Simpler reward function that only considers fuel efficiency, collision penalties, and success bonuses.\n
        :param action: current action chosen by the policy
        :return: Current reward
        """

        assert action.shape == (6,)

        rew = self.dt  # 0

        # Attitude bonus:
        rew += self.dt * self.att_coef * (1 - self.get_attitude_error() / self.max_attitude_error)
        # rew -= self.dt * att_coef * (self.get_attitude_error() / self.max_attitude_error)**0.5

        # Fuel efficiency:
        rew -= self.dt * np.abs(action[0:3]).sum() / 3 * self.fuel_coef

        # Collision penalty:
        if self.check_collision():
            rew -= self.dt * self.collision_coef

        # Success bonus:
        if np.linalg.norm(self.rc) < self.koz_radius and not self.collided:

            pos_error, vel_error, att_error, rot_error = self.get_errors()  # compute errors

            # Bonus for entering the corridor without touching KOZ:
            rew += self.dt * self.bonus_coef  # constant bonus

            # Bonus for achieving terminal conditions:
            if pos_error < self.max_rd_error:  # give terminal rewards only if the terminal position has been achieved
                bonus = self.dt * self.bonus_coef
                rew += bonus * (2 - pos_error / self.max_rd_error)
                if att_error < self.max_qd_error:
                    rew += bonus * (2 - att_error / self.max_qd_error)

        return rew

    # def get_reward(self):
    #
    #     phi = self.get_potential()
    #     rew = phi - self.prev_potential
    #     self.prev_potential = phi
    #
    #     return rew
    #
    # def get_potential(self):
    #     """
    #     Get the potential of the current state.\n
    #     :return:
    #     """
    #
    #     dist = np.linalg.norm(self.rc)
    #
    #     if self.check_collision():
    #         # Collided:
    #         phi = (1 - self.koz_radius / self.max_axial_distance) ** 2 - (self.koz_radius - dist)
    #     else:
    #         phi = (1 - max(dist, np.linalg.norm(self.rd)) / self.max_axial_distance) ** 2
    #
    #     # Success bonus:
    #     if np.linalg.norm(self.rc) < self.koz_radius and not self.collided:
    #
    #         goal_pos = self.target2lvlh(self.rd)
    #         pos_error = self.get_pos_error(goal_pos)
    #         if pos_error < self.max_rd_error:
    #             phi += 2 - pos_error / self.max_rd_error
    #             att_error = self.get_attitude_error()
    #             if att_error < self.max_qd_error:
    #                 phi += 2 - att_error / self.max_qd_error
    #
    #     return phi

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
            dist > self.bubble_radius,                              # chaser outside of bubble
            self.get_attitude_error() > self.max_attitude_error,    # chaser attitude error too large
            self.consecutive_success >= self.required_success,      # chaser has achieved enough success
        ]

        # (not self.observation_space.contains(obs)) or (self.t >= self.t_max) or (dist > self.bubble_radius)
        if any(end_conditions):
            done = True
            if end_conditions[-1] is True:
                self.successful_trajectory = True
            if not self.quiet:
                end_reasons = ['obs', 'time', 'bubble', 'attitude', 'success']  # reasons corresp to end_conditions
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

    def lvlh2chaser(self, vec: np.ndarray) -> np.ndarray:
        """
        Convert a vector from the LVLH frame into the chaser body (CB) frame.\n
        :param vec: 3D vector expressed in the LVLH frame
        :return: same vector expressed in the CB frame
        """
        assert self.qc is not None, "Cannot convert vector to chaser body frame. Chaser attitude is undefined."
        assert vec.shape == (3,), f"Vector must have a (3,) shape, but has {vec.shape}"
        return np.matmul(quat2mat(self.qc).T, vec)

    def lvlh2target(self, vec: np.ndarray) -> np.ndarray:
        """
        Convert a vector from the LVLH frame into the Target Body (TB) frame.\n
        :param vec: 3D vector expressed in the LVLH frame
        :return: same vector expressed in the TB frame
        """
        assert self.qt is not None, "Cannot convert vector to target body frame. Target attitude is undefined."
        assert vec.shape == (3,), f"Vector must have a (3,) shape, but has {vec.shape}"
        return np.matmul(quat2mat(self.qt).T, vec)

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

    def wait_for_target(self):
        """
        Rotate the target to a more favorable attitude along its expected path.\n
        :return: None
        """

        # Define the desired entry point:
        re_lvlh = self.target2lvlh(self.rd / np.linalg.norm(self.rd) * self.koz_radius)

        # Compute the circle that the entry point traces throughout one rotation:
        wt_lvlh = self.target2lvlh(self.wt)
        angles = np.linspace(0, 2 * np.pi - 0.17, 72)
        vecs = []
        for angle in angles:
            vecs.append(rotate_vector_about_axis(re_lvlh, wt_lvlh, angle))
        all_re_points = np.vstack(vecs)  # (LVLH)

        # Find the point that is closest to the chaser:
        index_min = np.argmin(np.linalg.norm(all_re_points - self.rc, axis=1))
        angle_min = angles[index_min]
        re_min = all_re_points[index_min]  # (LVLH)

        # Estimate how long it would take the chaser to reach that point:
        a = self.thrust / self.mc * self.bt / self.dt
        coef = 1  # Coefficient to add more time if necessary
        min_time_to_entry = coef * np.sqrt(2 * np.linalg.norm(self.rc - re_min) / a)

        # Compute the angular displacement of the entry point during that time:
        angle_back = np.linalg.norm(wt_lvlh) * min_time_to_entry

        # Rotate the target to the new orientation:
        self.qt = quat_product(self.qt, rot2quat(axis=self.wt, theta=angle_min-angle_back))
        return

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
