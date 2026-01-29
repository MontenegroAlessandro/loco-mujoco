import numpy as np
from scipy.spatial.transform import Rotation as np_R
import copy
import mujoco


class GaitGenerator:
    def __init__(
        self,
        feet_distance: float = 0.2,
        stop_steps: int = 2,
        gait_frequency: float = 1.0,
        policy_dt: float = 0.02,
        is_gp_adaptive: bool = False,
        min_gp_delta: float = 0.01,
        max_gp_delta: float = 0.04,
    ):
        # map the parameters
        self.feet_distance = feet_distance
        self.vertical_dist = 0.0
        self.lateral_dist = 0.0
        self.steering_angle =  0.0
        self.stop_steps = stop_steps
        # additional controlling parameters
        self.gaits_to_still = 0
        self.move_dir = "STILL"

        self.gait_process = 0.0
        self.policy_dt = policy_dt
        self.gait_frequency = gait_frequency
        self.is_gp_adaptive = is_gp_adaptive
        self.min_gp_delta = min_gp_delta
        self.max_gp_delta = max_gp_delta
        self.gp_off = self.policy_dt * self.gait_frequency

        self.swing_foot_idx = 0  # 0 for left, 1 for right
        self.sample_goal = False
        self.foot_offset = [
            np.array([0, self.feet_distance, 0.0]),
            np.array([1, 0, 0, 0]),
            np.array([0, -self.feet_distance, 0.0]),
            np.array([1, 0, 0, 0]),
        ]

        self.teleop = dict(
            # enabling movement stuff
            move_enabled=False,
            mov_dir="STILL",
            # veritacl steps
            vert_step=0.05,
            vert_min=-0.5,
            vert_max=0.5,
            # orientation stuff
            yaw_step=np.deg2rad(5.0),
            yaw_min=(-np.pi / 2.0),
            yaw_max=(np.pi / 2.0),
            # lateral steps stuff
            lat_step=0.05,
            lat_min=-0.3,
            lat_max=0.3,
        )

    def reset(self):
        self.vertical_dist = 0.0
        self.lateral_dist = 0.0
        self.steering_angle =  0.0

        self.gait_process = 0.0
        self.swing_foot_idx = 0  # 0 for left, 1 for right
        self.sample_goal = False
        self.foot_offset = [
            np.array([0, self.feet_distance, 0.0]),
            np.array([1, 0, 0, 0]),
            np.array([0, -self.feet_distance, 0.0]),
            np.array([1, 0, 0, 0]),
        ]
        self.gaits_to_still = 0
        self.move_dir = "STILL"

    def set_gp_offset(self, gp_offset: float):
        if self.is_gp_adaptive:
            raw_gp_clip = np.clip(gp_offset, -1.0, 1.0)
            self.gp_off = raw_gp_clip * (self.max_gp_delta - self.min_gp_delta) / 2.0 + \
                (self.max_gp_delta + self.min_gp_delta) / 2.0
        return self.gp_off

    def preprocess_gp_info(self):
        self.gait_process = (self.gait_process + self.gp_off) % 1.0
        swing_foot_idx = 0 if (self.gait_process < 0.5) else 1

        if self.swing_foot_idx != swing_foot_idx:
            self.sample_goal = True
            self.swing_foot_idx = swing_foot_idx
        else:
            self.sample_goal = False

        # Keyboard-controlled movement direction
        mov_dir = self.teleop["mov_dir"] if self.teleop["move_enabled"] else "STILL"
        if mov_dir != self.move_dir:
            # When coming to a stop, let the gait generator settle for a couple half-steps.
            if mov_dir != "STILL":
                self.gaits_to_still = self.stop_steps
            self.move_dir = mov_dir

        if self.gaits_to_still > 0:
            gait_info = np.array([np.cos(2 * np.pi * self.gait_process), np.sin(2 * np.pi * self.gait_process)])
        else:
            gait_info = np.array([0.0, 0.0])
        return gait_info

    def query_cmd(self):
        gait_info = self.preprocess_gp_info()

        if self.sample_goal:
            err_msg = f"[GaitGenerator: query_cmd] Mode {self.move_dir} is not valid."
            assert self.move_dir in ["STILL", "FWD", "BWD", "LEFT", "RIGHT", "DIAG-L", "DIAG-R"], err_msg

            if self.move_dir == "STILL":
                self.foot_offset = self._gen_still_cmd()
                self.gaits_to_still = max(0, self.gaits_to_still - 1)
            elif self.move_dir in ["FWD", "BWD"]:
                self.foot_offset = self._gen_vertical_cmd()
            elif self.move_dir in ["LEFT", "RIGHT"]:
                self.foot_offset = self._gen_lateral_cmd()
            elif self.move_dir in ["DIAG-L", "DIAG-R"]:
                direction = 1 if self.move_dir == "DIAG-L" else -1
                self.foot_offset = self._gen_diag_cmd(direction=direction)
        return self.foot_offset, gait_info

    def _gen_still_cmd(self):
        # no changing targets
        zero_orn_offset = np.array([1, 0, 0, 0], dtype=np.float32)
        zero_pos_offset = np.zeros(3, dtype=np.float32)

        if self.gaits_to_still > 0:
            # pos gen
            l_pos_offset = np.array([0, self.feet_distance, 0.0]) if self.swing_foot_idx == 0 else zero_pos_offset
            r_pos_offset = np.array([0, -self.feet_distance, 0.0]) if self.swing_foot_idx == 1 else zero_pos_offset
            # orn gen
            l_orn_offset = zero_orn_offset
            r_orn_offset = zero_orn_offset
        else:
            # pos gen
            l_pos_offset = np.array([0, self.feet_distance, 0.0])
            r_pos_offset = np.array([0, -self.feet_distance, 0.0])
            # orn gen
            l_orn_offset = zero_orn_offset
            r_orn_offset = zero_orn_offset

        return l_pos_offset, l_orn_offset, r_pos_offset, r_orn_offset

    def _gen_vertical_cmd(self):  # -> tuple[NDArray[Any] | NDArray[floating[_32Bit]], Any | NDA...:
        # no changing targets
        zero_orn_offset = np.array([1, 0, 0, 0], dtype=np.float32)
        zero_pos_offset = np.zeros(3, dtype=np.float32)

        # adjust the steering angle
        steering_angle = np.clip(self.steering_angle, -np.pi, np.pi)
        steering_foot_idx = 0 if (steering_angle >= 0 and steering_angle <= np.pi) else 1
        steering_orn_offset = np_R.from_euler("z", steering_angle).as_quat(scalar_first=True)

        # pos gen
        l_pos_offset = (
            np.array([self.vertical_dist, self.feet_distance, 0]) if self.swing_foot_idx == 0 else zero_pos_offset
        )
        r_pos_offset = (
            np.array([self.vertical_dist, -self.feet_distance, 0]) if self.swing_foot_idx == 1 else zero_pos_offset
        )
        # orn gen
        l_orn_offset = steering_orn_offset if steering_foot_idx == 0 else zero_orn_offset
        r_orn_offset = steering_orn_offset if steering_foot_idx == 1 else zero_orn_offset

        return l_pos_offset, l_orn_offset, r_pos_offset, r_orn_offset

    def _gen_lateral_cmd(self):
        # NOTE: direction 1 means left, -1 right
        direction = 1 if self.lateral_dist >= 0 else -1
        direction = 0 if abs(self.lateral_dist) < 1e-4 else direction

        # no changing targets
        zero_orn_offset = np.array([1, 0, 0, 0], dtype=np.float32)
        zero_pos_offset = np.zeros(3, dtype=np.float32)

        # ocmpute the lateral distance considering the offset of the feet distance
        lat_dist = self.feet_distance * direction + self.lateral_dist

        # clip the movement for the "evil foot"
        max_evil_movement = -self.feet_distance * direction / 2.0

        if direction == 1:  # left movement
            l_pos_offset = np.array([0.0, lat_dist, 0.0]) if self.swing_foot_idx == 0 else zero_pos_offset
            r_pos_offset = np.array([0.0, max_evil_movement, 0.0]) if self.swing_foot_idx == 1 else zero_pos_offset
        elif direction == -1:  # right movement
            l_pos_offset = np.array([0.0, max_evil_movement, 0.0]) if self.swing_foot_idx == 0 else zero_pos_offset
            r_pos_offset = np.array([0.0, lat_dist, 0.0]) if self.swing_foot_idx == 1 else zero_pos_offset
        else:  # no movement
            l_pos_offset = np.array([0, self.feet_distance, 0])
            r_pos_offset = np.array([0, -self.feet_distance, 0])

        # orientation
        l_orn_offset = zero_orn_offset
        r_orn_offset = zero_orn_offset

        return l_pos_offset, l_orn_offset, r_pos_offset, r_orn_offset

    def _gen_diag_cmd(self, direction: int = 1):
        # NOTE: direction 1 means left, -1 right
        err_msg = f"[GaitGenerator: _gen_diag_cmd] Direction {direction} is not valid."
        assert direction in [-1, 1], err_msg

        # no changing targets
        zero_orn_offset = np.array([1, 0, 0, 0], dtype=np.float32)
        zero_pos_offset = np.zeros(3, dtype=np.float32)

        # clip the movement for the "evil foot"
        max_evil_movement = self.lateral_dist

        if direction == 1:  # left movement
            l_pos_offset = (
                np.array([self.lateral_dist, self.lateral_dist, 0.0]) if self.swing_foot_idx == 0 else zero_pos_offset
            )
            r_pos_offset = (
                np.array([-max_evil_movement, -max_evil_movement, 0.0]) if self.swing_foot_idx == 1 else zero_pos_offset
            )
            l_orn_offset = zero_orn_offset
            r_orn_offset = zero_orn_offset
        else:  # right movement
            l_pos_offset = (
                np.array([-max_evil_movement, max_evil_movement, 0.0]) if self.swing_foot_idx == 0 else zero_pos_offset
            )
            r_pos_offset = (
                np.array([self.lateral_dist, -self.lateral_dist, 0.0]) if self.swing_foot_idx == 1 else zero_pos_offset
            )
            l_orn_offset = zero_orn_offset
            r_orn_offset = zero_orn_offset

        return l_pos_offset, l_orn_offset, r_pos_offset, r_orn_offset

    def key_callback(self, keycode):
        # MuJoCo constants for arrow keys
        # These are standard GLFW keycodes often used by MuJoCo
        LEFT_ARROW = 263
        RIGHT_ARROW = 262
        UP_ARROW = 265
        DOWN_ARROW = 264

        # Handle movement toggling (Pause/Play) with 'P'
        if keycode == ord("P") or keycode == ord("p"):
            self.teleop["move_enabled"] = not self.teleop["move_enabled"]

            if not self.teleop["move_enabled"]:
                self.teleop["mov_dir"] = "STILL"

        # Arrow keys for Directional Movement
        elif keycode == UP_ARROW:
            self.vertical_dist = float(
                np.clip(self.vertical_dist + self.teleop["vert_step"], self.teleop["vert_min"], self.teleop["vert_max"])
            )
            self.teleop["mov_dir"] = "FWD"
            print(f"[teleop] Vertical Distance: {self.vertical_dist:.2f}")
        elif keycode == DOWN_ARROW:
            self.vertical_dist = float(
                np.clip(self.vertical_dist - self.teleop["vert_step"], self.teleop["vert_min"], self.teleop["vert_max"])
            )
            self.teleop["mov_dir"] = "FWD"
            print(f"[teleop] Vertical Distance: {self.vertical_dist:.2f}")

        elif keycode == LEFT_ARROW:
            self.lateral_dist = float(
                np.clip(self.lateral_dist + self.teleop["lat_step"], self.teleop["lat_min"], self.teleop["lat_max"])
            )
            self.teleop["mov_dir"] = "LEFT"
            print(f"[teleop] Lateral Distance: {self.lateral_dist:.2f}")

        elif keycode == RIGHT_ARROW:
            self.lateral_dist = float(
                np.clip(self.lateral_dist - self.teleop["lat_step"], self.teleop["lat_min"], self.teleop["lat_max"])
            )
            self.teleop["mov_dir"] = "RIGHT"
            print(f"[teleop] Lateral Distance: {self.lateral_dist:.2f}")

        # Steering Angle Adjustment using Brackets [ ]
        elif keycode == ord(","):
            self.steering_angle = float(
                np.clip(self.steering_angle + self.teleop["yaw_step"], self.teleop["yaw_min"], self.teleop["yaw_max"])
            )
            print(f"[teleop] Steering Angle (deg): {np.rad2deg(self.steering_angle):.2f}")

        elif keycode == ord("."):
            self.steering_angle = float(
                np.clip(self.steering_angle - self.teleop["yaw_step"], self.teleop["yaw_min"], self.teleop["yaw_max"])
            )
            print(f"[teleop] Steering Angle (deg): {np.rad2deg(self.steering_angle):.2f}")

        elif keycode == ord("|") or keycode == ord("\\"):
            self.steering_angle = 0.0
            self.lateral_dist = 0.0
            self.vertical_dist = 0.0
            self.teleop["mov_dir"] = "FWD" if self.teleop["move_enabled"] else "STILL"
            print(f"[teleop] RESET.")

    def print_instruction(self):
        # ===============================================PRINT INSTRUCTIONS===============================================
        print("\n" + "=" * 60)
        print("          HUMANOID SIM2SIM TELEOPERATION CONTROLS          ")
        print("=" * 60)
        print("  [P]           : Toggle Movement ON/OFF (Defaults to STILL)")
        print("-" * 60)
        print("  [Arrow UP]    : Increase Forward Step Length (Accel)")
        print("  [Arrow DOWN]  : Decrease Forward Step Length (Decel)")
        print("  [Arrow LEFT]  : Step Left (Increase Lateral Dist)")
        print("  [Arrow RIGHT] : Step Right (Decrease Lateral Dist)")
        print("-" * 60)
        print("  [ ',' ]       : Steer Left  (Yaw +)")
        print("  [ '.' ]       : Steer Right (Yaw -)")
        print("-" * 60)
        print("  [ '\\' ] or [|]: EMERGENCY RESET (Zero all commands)")
        print("=" * 60 + "\n")


class GoalReachingGaitGenerator(GaitGenerator):
    def __init__(self, model, data, max_angle, gait_frequency, policy_dt, feet_distance: float = 0.2, stop_steps: int = 2,
                 is_gp_adaptive: bool = False, min_gp_delta: float = 0.01, max_gp_delta: float = 0.04):
        super().__init__(feet_distance=feet_distance, stop_steps=stop_steps, gait_frequency=gait_frequency, policy_dt=policy_dt, is_gp_adaptive=is_gp_adaptive, min_gp_delta=min_gp_delta, max_gp_delta=max_gp_delta)
        self.model = copy.deepcopy(model)
        self.data = copy.deepcopy(data)

        self.left_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        self.right_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

        self.first_step = True
        self.gp_shift = 0.0

        self.max_angle = max_angle
        self.turning_threshold = self.max_angle * 2

    def query_cmd(self, goal_pos, q_pos):
        self.data.qpos[:] = q_pos
        mujoco.mj_fwdPosition(self.model, self.data)

        target_dir = goal_pos - q_pos[:3]
        target_dist = np.linalg.norm(target_dir[:2])


        if self.move_dir == "STILL":
            self.first_step = True
        if self.move_dir in ["FWD", "BWD"] and self.first_step:
            self.first_step = False
            swing_foot_idx = 0 if target_dir[1] >= 0.0 else 1
            self.swing_foot_idx = 1 - swing_foot_idx
            desired_gp = 0.25 if swing_foot_idx == 0 else 0.75
            # self.gp_shift = (desired_gp - (gp % 1.0)) % 1.0

        # gp = (gp + self.gp_shift) % 1.0
        gait_info = self.preprocess_gp_info()

        stance_foot = self.left_foot_id if self.swing_foot_idx == 1 else self.right_foot_id

        stance_foot_pos = self.data.site_xpos[stance_foot]
        stance_foot_xmat = self.data.site_xmat[stance_foot].reshape(3, 3)

        target_to_stance = goal_pos - stance_foot_pos
        base_st_to_stance = q_pos[:3] - stance_foot_pos

        goal_st = stance_foot_xmat.T @ target_to_stance
        base_st = stance_foot_xmat.T @ base_st_to_stance
        desired_yaw = np.clip(self._compute_desired_foot_yaw(goal_st, base_st), -self.max_angle, self.max_angle)

        if self.sample_goal:
            if self.move_dir == "STILL":
                self.foot_offset = self._gen_still_cmd()
                self.gaits_to_still = max(0, self.gaits_to_still - 1)
            elif self.move_dir in ["FWD", "BWD"]:
                if target_dist < 0.2:
                    self.foot_offset = self._gen_still_cmd()
                    self.gaits_to_still = max(0, self.gaits_to_still - 1)
                    if self.gaits_to_still == 0:
                        self.first_step = True
                else:
                    self.gaits_to_still = self.stop_steps
                    self.foot_offset = self._gen_goal_reaching_cmd(
                        target_st=goal_st[:2],
                        desired_yaw=desired_yaw,
                    )

        return self.foot_offset, gait_info

    def _compute_desired_foot_yaw(self, end_point, start_point):
        end_to_start = end_point - start_point
        end_to_start_dir = end_to_start / np.linalg.norm(end_to_start) + 1e-6
        yaw = np.arctan2(end_to_start_dir[1], end_to_start_dir[0]) + np.pi * (np.sign(self.vertical_dist) < 0)
        return yaw

    def _gen_goal_reaching_cmd(self, target_st: np.ndarray, desired_yaw: float):
        zero_orn_offset = np.array([1, 0, 0, 0], dtype=np.float32)
        zero_pos_offset = np.zeros(3, dtype=np.float32)

        swing_orn_offset = np_R.from_euler("z", float(desired_yaw)).as_quat(scalar_first=True).astype(np.float32)

        y_min, y_max = self.feet_distance, 1.5 * self.feet_distance
        sign = +1.0 if self.swing_foot_idx == 0 else -1.0

        if sign * float(target_st[1]) >= 0.0:
            y_mag = float(np.clip(abs(float(target_st[1])), y_min, y_max))
        else:
            y_mag = float(y_min)

        y = float(sign * y_mag)
        x = target_st[0]

        turn = abs(desired_yaw) / self.max_angle
        rad = self.feet_distance ** 2 + ((1 - 0.2 * turn) * self.vertical_dist) ** 2 - y ** 2
        x_max = np.sqrt(max(rad, 0.0))
        x = np.clip(x, -x_max, x_max)

        swing_pos = np.array([x, y, 0.0], dtype=np.float32)

        l_pos_offset = swing_pos if self.swing_foot_idx == 0 else zero_pos_offset
        r_pos_offset = swing_pos if self.swing_foot_idx == 1 else zero_pos_offset
        l_orn_offset = swing_orn_offset if self.swing_foot_idx == 0 else zero_orn_offset
        r_orn_offset = swing_orn_offset if self.swing_foot_idx == 1 else zero_orn_offset

        return l_pos_offset, l_orn_offset, r_pos_offset, r_orn_offset

    @staticmethod
    def _wrap_to_pi(a: float) -> float:
        return (a + np.pi) % (2 * np.pi) - np.pi


class NarrowPathGaitGenerator(GoalReachingGaitGenerator):

    def __init__(
        self,
        model,
        data,
        max_angle,
        path_pts,
        gait_frequency,
        policy_dt,
        feet_distance: float = 0.2,
        stop_steps: int = 2,
        is_gp_adaptive: bool = False,
        min_gp_delta: float = 0.01,
        max_gp_delta: float = 0.04,
    ):
        super().__init__(model=model, data=data, max_angle=max_angle, feet_distance=feet_distance, stop_steps=stop_steps, gait_frequency=gait_frequency, policy_dt=policy_dt, is_gp_adaptive=is_gp_adaptive, min_gp_delta=min_gp_delta, max_gp_delta=max_gp_delta)
        assert path_pts is not None and len(path_pts) >= 3, "Need path_pts (>=3)."
        self.path_pts = [np.asarray(p, dtype=np.float32) for p in path_pts]


    def query_cmd(self, goal_pos, q_pos, goal_stage: int):
        self.data.qpos[:] = q_pos
        mujoco.mj_fwdPosition(self.model, self.data)

        target_dir = goal_pos - q_pos[:3]
        target_dist = np.linalg.norm(target_dir[:2])

        if self.move_dir == "STILL":
            self.first_step = True
        if self.move_dir in ["FWD", "BWD"] and self.first_step:
            self.first_step = False
            swing_foot_idx = 0 if target_dir[1] >= 0.0 else 1
            self.swing_foot_idx = 1 - swing_foot_idx
            desired_gp = 0.25 if swing_foot_idx == 0 else 0.75
            # self.gp_shift = (desired_gp - (gp % 1.0)) % 1.0

        # gp = (gp + self.gp_shift) % 1.0
        gait_info = self.preprocess_gp_info()

        stance_foot = self.left_foot_id if self.swing_foot_idx == 1 else self.right_foot_id
        stance_foot_pos = self.data.site_xpos[stance_foot]
        stance_foot_xmat = self.data.site_xmat[stance_foot].reshape(3, 3)

        target_to_stance = goal_pos - stance_foot_pos
        base_st_to_stance = q_pos[:3] - stance_foot_pos
        prev_waypoint_to_stance = np.array(
            [self.path_pts[goal_stage - 1][0], self.path_pts[goal_stage - 1][1], 0.0]) - stance_foot_pos

        goal_st = stance_foot_xmat.T @ target_to_stance
        base_st = stance_foot_xmat.T @ base_st_to_stance
        prev_waypoint_st = stance_foot_xmat.T @ prev_waypoint_to_stance

        if (goal_stage >= 1) and (goal_stage <= len(self.path_pts) - 2):
            yaw =self._compute_desired_foot_yaw(goal_st, prev_waypoint_st)
            desired_yaw = np.clip(self._choose_perpendicular_yaw(yaw), -self.max_angle, self.max_angle)

        else:
            desired_yaw = np.clip(self._compute_desired_foot_yaw(goal_st, base_st), -self.max_angle, self.max_angle)

        if self.sample_goal:
            if self.move_dir == "STILL":
                self.foot_offset = self._gen_still_cmd()
                self.gaits_to_still = max(0, self.gaits_to_still - 1)

            elif self.move_dir in ["FWD", "BWD"]:
                if target_dist < 0.2:
                    self.foot_offset = self._gen_still_cmd()
                    self.gaits_to_still = max(0, self.gaits_to_still - 1)
                    if self.gaits_to_still == 0:
                        self.first_step = True
                else:
                    self.gaits_to_still = self.stop_steps
                    if (goal_stage > 1) and (goal_stage <= len(self.path_pts) - 2):
                        self.foot_offset = self._gen_side_step_cmd(
                            desired_yaw=desired_yaw,
                            target_st=goal_st[:2]
                        )
                    else:
                        self.foot_offset = self._gen_goal_reaching_cmd(
                            desired_yaw=desired_yaw,
                            target_st=goal_st[:2]
                        )
        return self.foot_offset, gait_info


    def _gen_side_step_cmd(self, desired_yaw, target_st: np.ndarray) -> np.ndarray:
        zero_orn_offset = np.array([1, 0, 0, 0], dtype=np.float32)
        zero_pos_offset = np.zeros(3, dtype=np.float32)

        sign = +1.0 if self.swing_foot_idx == 0 else -1.0
        if sign * float(target_st[1]) >= 0.0:
            y = self.feet_distance * sign + sign * self.vertical_dist

        else:
            y = -self.feet_distance * sign / 2.0

        x = target_st[0]
        turn = abs(desired_yaw) / self.max_angle
        rad = self.feet_distance ** 2 + ((1 - 0.2 * turn) * self.vertical_dist) ** 2 - y ** 2
        x_max = np.sqrt(max(rad, 0.0))
        x = np.clip(x, -x_max, x_max)

        swing_pos = np.array([x, y, 0.0], dtype=np.float32)

        swing_orn_offset = np_R.from_euler("z", float(desired_yaw)).as_quat(scalar_first=True).astype(
            np.float32)

        l_pos_offset = swing_pos if self.swing_foot_idx == 0 else zero_pos_offset
        r_pos_offset = swing_pos if self.swing_foot_idx == 1 else zero_pos_offset
        l_orn_offset = swing_orn_offset if self.swing_foot_idx == 0 else zero_orn_offset
        r_orn_offset = swing_orn_offset if self.swing_foot_idx == 1 else zero_orn_offset

        return l_pos_offset, l_orn_offset, r_pos_offset, r_orn_offset

    def _choose_perpendicular_yaw(self, hear_yaw: float) -> float:
        cand1 = self._wrap_to_pi(hear_yaw + np.pi / 2)
        cand2 = self._wrap_to_pi(hear_yaw - np.pi / 2)

        return cand1 if abs(cand1) < abs(cand2) else cand2
