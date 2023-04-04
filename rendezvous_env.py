import gym
from gym import spaces
import numpy as np
from scipy.integrate import solve_ivp
from utils.quaternions import quat2mat, rot2quat, quat_product
from utils.general import normalize_value, angle_between_vectors, random_unit_vector
from utils.dynamics import clohessy_wiltshire_solution, derivative_of_att_and_rot_rate


class RendezvousEnv(gym.Env):
    """
    Environment to simulate a 6-DOF rendezvous scenario with a rotating target.

    Written by C. F. De Inza Niemeijer.\n
    """

    def __init__(self,
                 rc0: np.ndarray=None,              # nominal initial position of the chaser [m]
                 vc0: np.ndarray=None,              # nominal initial velocity of the chaser [m/s]
                 qc0: np.ndarray=None,              # nominal initial attitude of the chaser [-]
                 wc0: np.ndarray=None,              # nominal initial rotation rate of the chaser [rad/s]
                 qt0: np.ndarray=None,              # nominal initial attitude of the target [-]
                 wt0: np.ndarray=None,              # nominal initial rotation rate of the target [rad/s]
                 rc0_range: float=None,             # range of the initial position of the chaser [m] (float)
                 vc0_range: float=None,             # range of the initial velocity of the chaser [m/s] (float)
                 qc0_range: float=None,             # range of the initial attitude of the chaser [-] (float)
                 wc0_range: float=None,             # range of the initial rotation rate of the chaser [rad/s] (float)
                 qt0_range: float=None,             # range of the initial attitude of the target [-] (float)
                 wt0_range: float=None,             # range of the initial rotation rate of the target [rad/s] (float)
                 reward_kwargs: dict=None,          # dictionary with keyword arguments for the reward function (dict)
                 koz_radius: float=None,            # radius of the keep-out zone around the target [m] (float)
                 corridor_half_angle: float=None,   # half-angle of the conic entry corridor [rad] (float)
                 h: float=None,                     # altitude of the orbit [m]
                 dt: float=None,                    # size of time step [s]
                 t_max: float=None,                 # maximum length of the episode [s]
                 quiet: bool=False,                 # whether or not to print episode results
                 ):
        """
        This function nitializes the attributes of the environment, including chaser properties, target properties,
        orbit properties, observation space, action space, and more.\n
        """

        # State variables:  (initialized on reset() call)
        self.rc = None      # position of the chaser [m] expressed in LVLH frame
        self.vc = None      # velocity of the chaser [m/s] expressed in LVLH frame
        self.qc = None      # attitude of the chaser [-] expressed as a quaternion orientation relative to LVLH
        self.wc = None      # rotation rate of the chaser [rad/s] expressed in chaser body frame
        self.qt = None      # attitude of the target [-] expressed as a quaternion orientation relative to LVLH
        self.wt = None      # rotation rate of the target [rad/s] expressed in target body frame

        # Nominal initial state:
        self.nominal_rc0 = np.array([0., -10., 0.]) if rc0 is None else rc0
        self.nominal_vc0 = np.array([0., 0., 0.]) if vc0 is None else vc0
        self.nominal_qc0 = np.array([1., 0., 0., 0.]) if qc0 is None else qc0
        self.nominal_wc0 = np.array([0., 0., 0.]) if wc0 is None else wc0
        self.nominal_qt0 = np.array([1., 0., 0., 0.]) if qt0 is None else qt0
        self.nominal_wt0 = np.array([0., 0., 0.]) if wt0 is None else wt0

        # Range of initial conditions: (initial conditions are sampled uniformly from this range)
        self.rc0_range = 1 if rc0_range is None else rc0_range                  # 1   [m]
        self.vc0_range = 0.1 if vc0_range is None else vc0_range                # 0.1 [m/s]
        self.qc0_range = np.radians(1) if qc0_range is None else qc0_range      # 5   [deg]
        self.wc0_range = np.radians(0.1) if wc0_range is None else wc0_range    # 3   [deg/s]
        self.qt0_range = np.radians(45) if qt0_range is None else qt0_range     # 45  [deg]
        self.wt0_range = np.radians(3) if wt0_range is None else wt0_range      # 3   [deg/s]

        # Time:
        self.t = None                                   # current time of episode [s]
        self.dt = 1 if dt is None else dt               # interval between timesteps [s]
        self.t_max = 120 if t_max is None else t_max    # maximum length of episode [s]

        # Chaser properties:
        self.capture_axis = np.array([0, 1, 0])     # capture axis expressed in the chaser body frame (constant)
        self.m = 100                                # mass [kg]
        self.inertia = np.array([                   # moment of ineria [kg.m^2]
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]) * 1/12 * self.m * (2 * 1**2)
        self.inv_inertia = np.linalg.inv(self.inertia)
        self.max_delta_v = 10 / self.m * 0.5                # 0.05 m/s per step
        self.max_delta_w = 0.2 / self.inertia[0, 0] * 0.5   # 0.006 rad/s per step

        # Chaser state limits:
        self.max_axial_distance = np.linalg.norm(self.nominal_rc0) + 10     # maximum distance allowed on each axis [m]
        self.max_axial_speed = 5                                            # maximum speed allowed on each axis [m/s]
        self.max_wc = np.radians(10)                                        # maximum chaser rotation rate [rad/s]
        self.max_wt = np.radians(10)                                        # maximum target rotation rate [rad/s]
        self.max_attitude_error = np.radians(30)                            # maximum allowed pointing error (ep. ends)
        # Old values: axial_speed=10 m/s, wc=pi rad

        # Target properties:
        self.koz_radius = 5 if koz_radius is None else koz_radius  # radius of the keep-out zone [m]
        self.corridor_half_angle = np.radians(30) if corridor_half_angle is None else corridor_half_angle
        self.corridor_axis = np.array([0, -1, 0])       # entry corridor expressed in the target body frame (constant)
        self.inertia_target = np.array([                # moment of inertia of the target [kg.m^2]
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]) * 1/12 * self.m * (2 * 1**2)
        self.inv_inertia_target = np.linalg.inv(self.inertia_target)

        # Capture conditions:
        self.rd = np.array([0, -2, 0])      # desired capture position [m] expressed in target body frame
        self.max_rd_error = 0.5             # allowed error for the capture position [m]
        self.max_vd_error = 0.1             # allowed error for the relative capture velocity [m/s]
        self.max_qd_error = np.radians(5)   # allowed error for the capture attitude [rad]
        self.max_wd_error = np.radians(1)   # allowed error for the relative capture rotation rate [rad/s]
        self.collided = None                # whether or not the chaser has collided during the current episode
        self.success = None                 # shows if the chaser has achieved the terminal conditions w/o colliding

        # Properties of the reward function:
        self.bubble_radius = None                       # Radius of the virtual sphere that sets the allowed space (m)
        self.bubble_radius0 = self.max_axial_distance   # Initial radius of the bubble
        self.bubble_decrease_rate = 0.5 * self.dt       # Rate at which the size of the bubble decreases (m/s)
        self.bubble_min = np.linalg.norm(self.rd) + 2 * self.max_rd_error
        self.total_delta_v = None                       # Keeps track of delta_V used during the episode
        self.total_delta_w = None                       # Keeps track of delta_omega used during the episode
        self.reward_kwargs = {} if reward_kwargs is None else reward_kwargs  # keyword args for the reward function

        # Orbit properties:
        self.mu = 3.986004418e14                        # Gravitational parameter of Earth [m^3/s^2]
        self.Re = 6371e3                                # Radius of the Earth [m]
        self.h = 800e3 if h is None else h              # Altitude of the orbit [m]
        self.ro = self.Re + self.h                      # Radius of the orbit [m]
        self.n = np.sqrt(self.mu / self.ro ** 3)        # Mean motion of the orbit [rad/s]

        # Other:
        self.viewer = None
        self.quiet = quiet

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
        """
        Propagate the state of the system one step forward in time.\n
        :param action: action selected by the controller.
        :return: observation, reward, done, info
        """

        # Process the action:
        assert action.shape == (6,)
        # processed_action = action - 1
        processed_action = action.copy()

        delta_v = self.chaser2lvlh(processed_action[0:3] * self.max_delta_v)
        delta_w = processed_action[3:] * self.max_delta_w  # already expressed in the body frame (no rotation needed)

        # Update the position and velocity of the chaser:
        vc_after_impulse = self.vc + delta_v
        self.rc, self.vc = clohessy_wiltshire_solution(self.rc, vc_after_impulse, self.n, self.dt)

        # Update the attitude and rotation rate of the chaser:
        self.wc = self.wc + delta_w
        self.integrate_chaser_attitude(np.array([0, 0, 0]))

        # Update the attitude and rotation rate of the target:
        self.integrate_target_attitude(np.array([0, 0, 0]))

        # Check for collision and success:
        if not self.collided:
            self.collided = self.check_collision()
            if self.check_success():
                self.success += 1

        # Update the time step:
        self.t = round(self.t + self.dt, 3)  # rounding to prevent issues when dt is a decimal

        # Update bubble:
        self.bubble_radius -= self.bubble_decrease_rate
        if self.bubble_radius < self.bubble_min:
            self.bubble_radius = self.bubble_min

        # Update the deltas:
        self.total_delta_v += np.abs(processed_action[0:3]).sum() * self.max_delta_v
        self.total_delta_w += np.abs(processed_action[3:]).sum() * self.max_delta_w

        # Create observation: (normalize all values except quaternions)
        obs = self.get_observation()

        # Check if episode is done:
        done = self.get_done_condition(obs)

        # Calculate reward:
        rew = self.get_bubble_reward(processed_action, **self.reward_kwargs)

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

        # Generate random deviations for the state of the system:
        # Chaser position:
        rc0_deviation = random_unit_vector() * np.random.uniform(low=0, high=self.rc0_range)

        # Chaser velocity:
        vc0_deviation = random_unit_vector() * np.random.uniform(low=0, high=self.vc0_range)

        # Chaser attitude:
        theta_c = np.random.uniform(low=0, high=self.qc0_range)
        euler_c = random_unit_vector()
        qc_deviation = rot2quat(axis=euler_c, theta=theta_c)

        # Chaser rotation rate:
        wc0_deviation = random_unit_vector() * np.random.uniform(low=0, high=self.wc0_range)  # Expressed in LVLH

        # Target attitude:
        theta_t = np.random.uniform(low=0, high=self.qt0_range)
        euler_t = random_unit_vector()
        qt_deviation = rot2quat(axis=euler_t, theta=theta_t)

        # Target rotation rate:
        wt0_deviation = random_unit_vector() * np.random.uniform(low=0, high=self.wt0_range)  # Expressed in LVLH

        # Apply randomness to nominal initial conditions:
        self.rc = self.nominal_rc0 + rc0_deviation
        self.vc = self.nominal_vc0 + vc0_deviation
        self.qc = quat_product(qc_deviation, self.nominal_qc0)          # apply seq. rotations (first nominal, then dev)
        self.wc = self.lvlh2chaser(self.nominal_wc0 + wc0_deviation)    # convert wc0 from LVLH to chaser body frame
        self.qt = quat_product(qt_deviation, self.nominal_qt0)          # apply seq. rotations (first nominal, then dev)
        self.wt = self.lvlh2target(self.nominal_wt0 + wt0_deviation)    # convert wt0 from LVLH to target body frame

        # Reset time and other parameters:
        self.collided = self.check_collision()
        self.success = int(self.check_success())
        self.bubble_radius = self.bubble_radius0
        self.total_delta_v = 0
        self.total_delta_w = 0
        self.t = 0

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

    """=============================================== Utils: ======================================================="""
    def get_observation(self):
        """
        Get the current observation of the system.
        This is done by normalizing the state variables to a range between -1 & 1.\n
        :return: the observation
        """

        assert self.rc is not None

        obs = np.hstack((
            normalize_value(self.rc, -self.max_axial_distance, self.max_axial_distance),
            normalize_value(self.vc, -self.max_axial_speed, self.max_axial_speed),
            self.qc,
            normalize_value(self.wc, -self.max_wc, self.max_wc),
            self.qt,
            # normalize_value(self.wt, -self.max_wt, self.max_wt),
        ))
        return obs.astype(self.observation_space.dtype)

    def get_bubble_reward(self, action, collision_coef=0.5, bonus_coef=8, fuel_coef=0.2, att_coef=1):
        """
        Simpler reward function that only considers fuel efficiency, collision penalties, and success bonuses.\n
        :param action: current action chosen by the policy
        :param collision_coef: coefficient that modifies the reward when the chaser is within the KOZ
        :param bonus_coef: coefficient that modifies the reward when it achieves a goal
        :param fuel_coef: coefficient that modifies the reward when it uses fuel
        :param att_coef: coefficient that modifies the reward based on the attitude error of the chaser
        :return: Current reward
        """

        assert action.shape == (6,)

        rew = 0

        # Attitude bonus:
        rew += self.dt * att_coef * (1 - self.get_attitude_error() / self.max_attitude_error)

        # Fuel efficiency bonus:
        # rew -= self.dt * np.abs(action[0:3]).sum() / 3 * fuel_coef
        rew += self.dt * fuel_coef * np.abs(action[0:3]).sum() / (3 * self.max_delta_v)

        # Collision penalty:
        if self.check_collision():
            rew -= self.dt * collision_coef

        # Success bonus:
        if np.linalg.norm(self.rc) < self.koz_radius and not self.collided:

            pos_error, vel_error, att_error, rot_error = self.get_errors()  # compute errors

            # Bonus for entering the corridor without touching KOZ:
            # rew += self.dt * bonus_coef

            # Bonus for achieving terminal conditions:
            if pos_error < self.max_rd_error:
                rew += self.dt * bonus_coef * (2 - pos_error / self.max_rd_error)
                if att_error < self.max_qd_error:
                    rew += self.dt * bonus_coef * (2 - att_error / self.max_qd_error)

        return rew

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
            # self.consecutive_success >= self.required_success,      # chaser has achieved enough success
        ]

        if any(end_conditions):
            done = True
            if not self.quiet:
                end_reasons = ['obs', 'time', 'bubble', 'attitude']                 # reasons corr. to end_conditions
                print("Episode end" +
                      " | r = " + str(round(dist, 2)).rjust(5) +                    # chaser position at episode end
                      " | t = " + str(self.t).rjust(4) +                            # time of episode end
                      " | " + end_reasons[end_conditions.index(True)].center(8) +   # reason for episode end
                      " | " + ("Collided" if self.collided else " "))               # whether chaser entered KOZ
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
            angle_to_corridor = angle_between_vectors(self.rc, self.target2lvlh(self.corridor_axis))
            if angle_to_corridor > self.corridor_half_angle:
                collided = True

        return collided

    def check_success(self) -> int:
        """
        Check if the chaser has achieved the terminal conditions without entering the KOZ.\n
        :return: True if trajectory was successful, otherwise False
        """

        if self.collided:
            success = 0
        else:
            errors = self.get_errors()
            error_ranges = np.array([self.max_rd_error, self.max_vd_error, self.max_qd_error, self.max_wd_error])
            if np.all(errors <= error_ranges):
                success = 1
            else:
                success = 0

        return success

    def get_attitude_error(self):
        """
        Compute the chaser's attitude error. The chaser should always be pointing in the direction of the target's
        center of mass.\n
        :return: angle between the actual and desired pointing direction (in radians)
        """

        capture_axis = self.chaser2lvlh(self.capture_axis)      # Capture axis of the chaser expressed in LVLH
        error = angle_between_vectors(-self.rc, capture_axis)   # Attitude error [rad]

        return error

    def get_goal_pos(self):
        """
        Compute the position of the goal (expressed in the LVLH frame).\n
        :return: position of the goal (in meters)
        """
        return self.target2lvlh(self.rd)

    def get_pos_error(self, goal_position):
        """
        Compute the position error (the distance between the chaser and the goal position).\n
        :param goal_position: position of the goal [m] (expressed in LVLH)
        :return: position error [m]
        """
        return np.linalg.norm(self.rc - goal_position)

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

    def dist_from_koz(self) -> float:
        """
        Compute the minimum distance between the chaser and the surface of the KOZ. If the chaser is outside the koz,
        the distance is positive. If the chaser is inside the KOZ, the distance is negative.\n
        :return: distance (in meters) to the closest point on the surface of the KOZ
        """
        pos_mag = np.linalg.norm(self.rc)       # Magnitude of the chaser position [m]
        r_koz = self.koz_radius                 # radius of the KOZ [m]
        theta_corr = self.corridor_half_angle   # Half-angle of the entry corridor [rad]
        theta_chaser = angle_between_vectors(self.rc, self.target2lvlh(self.corridor_axis))  # angle btw rc & corr axis
        if pos_mag < r_koz:
            if theta_chaser >= theta_corr:
                # Collided:
                d_rad = r_koz - pos_mag  # positive
                d_tan = pos_mag * np.sin(min(theta_chaser - theta_corr, np.pi / 2))  # positive
                d_koz = -1 * min(d_rad, d_tan)  # negative
                assert d_koz <= 0, f"Error computing distance from KOZ. Dist = {d_koz} (expected negative value)"
            else:
                # Within corridor
                d_koz = pos_mag * np.sin(theta_corr - theta_chaser)  # positive
        else:
            if theta_chaser >= theta_corr:
                d_koz = pos_mag - r_koz  # positive
            else:
                d_rad = pos_mag - r_koz * np.cos(theta_corr - theta_chaser)  # positive
                d_tan = r_koz * np.sin(theta_corr - theta_chaser)
                d_koz = np.sqrt(d_rad ** 2 + d_tan ** 2)
        return d_koz

    def check_vec_in_corridor(self, vec_lvlh: np.ndarray):
        """
        Check if a given vector is inside the entry corridor.\n
        :param vec_lvlh: 3D vector expressed in the LVLH reference frame.
        :return: True if the vector lies inside the entry corridor, otherwise False.
        """
        in_corridor = False
        dist = np.linalg.norm(vec_lvlh)
        angle_from_corridor = angle_between_vectors(vec_lvlh, self.target2lvlh(self.corridor_axis))
        if dist < self.koz_radius and angle_from_corridor < self.corridor_half_angle:
            in_corridor = True
        return in_corridor

    def integrate_chaser_attitude(self, torque):
        """
        Use a fourth-order Runge-Kutta integrator to update the attitude and rotation rate of the chaser.\n
        :param torque: torque applied at the start of this time step
        :return: None
        """

        y0 = np.append(self.qc, self.wc)

        sol = solve_ivp(
            fun=derivative_of_att_and_rot_rate,
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

        return

    def integrate_target_attitude(self, torque):
        """
        Use a fourth-order Runge-Kutta integrator to update the attitude and rotation rate of the target.\n
        :param torque: Torque applied to the target (always zero).
        :return: None
        """

        y0 = np.append(self.qt, self.wt)

        sol = solve_ivp(
            fun=derivative_of_att_and_rot_rate,
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

    from stable_baselines3.common.env_checker import check_env
    env = RendezvousEnv()

    # Check the custom environment and output warnings if needed.
    print("Checking environment...")
    check_env(env)
