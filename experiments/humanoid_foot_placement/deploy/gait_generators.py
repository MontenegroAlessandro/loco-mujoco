import numpy as np
from scipy.spatial.transform import Rotation as np_R
import copy
import mujoco
import cv2
import os
import pyrealsense2 as rs


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


class VisualGaitGenerator(GaitGenerator):
    def __init__(
        self,
        robot_model,
        robot_data,
        cam_width,
        cam_height,
        gait_frequency: float = 1.0,
        policy_dt: float = 0.02,
        feet_distance: float = 0.2,
        stop_steps: int = 2,
        is_gp_adaptive: bool = False,
        min_gp_delta: float = 0.01,
        max_gp_delta: float = 0.04,
        max_gp_pause_steps: int = 5,
        img_delay_steps: int = 0,
        debug_vis: bool = False,
    ):
        super().__init__(feet_distance=feet_distance, stop_steps=stop_steps, gait_frequency=gait_frequency, policy_dt=policy_dt, is_gp_adaptive=is_gp_adaptive, min_gp_delta=min_gp_delta, max_gp_delta=max_gp_delta)

        self.model = copy.deepcopy(robot_model)
        self.data = copy.deepcopy(robot_data)

        self.left_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        self.right_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")

        self.img_delay_steps = img_delay_steps
        self.remaining_delay_steps = 0
        self.gait_process_forward = True
        self.swing_foot_on_ground = True
        self.gp_paused_steps = 0
        self.max_pause_steps = max_gp_pause_steps

        self.debug_vis = debug_vis
        self.img_count = 0

        self.rs_intrinsics = rs.intrinsics()
        self.rs_intrinsics.width = cam_width
        self.rs_intrinsics.height = cam_height
        self.rs_intrinsics.ppx = cam_width / 2.0
        self.rs_intrinsics.ppy = cam_height / 2.0
        self.rs_intrinsics.fx = cam_height / (2.0 * np.tan(np.deg2rad(self.model.cam_fovy[self.cam_id]) / 2.0))
        self.rs_intrinsics.fy = cam_height / (2.0 * np.tan(np.deg2rad(self.model.cam_fovy[self.cam_id]) / 2.0))
        self.rs_intrinsics.model = rs.distortion.none
        self.rs_intrinsics.coeffs = [0, 0, 0, 0, 0]

        if debug_vis:
            os.makedirs("./outputs/vis", exist_ok=True)

    def update_camera_info(self, cam_info):
        self.rs_intrinsics.width = cam_info.width
        self.rs_intrinsics.height = cam_info.height
        self.rs_intrinsics.ppx = cam_info.k[2]
        self.rs_intrinsics.ppy = cam_info.k[5]
        self.rs_intrinsics.fx = cam_info.k[0]
        self.rs_intrinsics.fy = cam_info.k[4]
        self.rs_intrinsics.model = self.get_rs_distortion_model(cam_info.distortion_model)
        self.rs_intrinsics.coeffs = [i for i in cam_info.d]

    @staticmethod
    def get_rs_distortion_model(ros_model_str):
        """
        Maps ROS distortion_model strings to pyrealsense2 distortion enums.
        """
        mapping = {
            "plumb_bob": rs.distortion.brown_conrady,
            "equidistant": rs.distortion.kannala_brandt4,
            # RealSense sometimes uses 'inverse_brown_conrady' for specific modules
            "inverse_brown_conrady": rs.distortion.inverse_brown_conrady,
        }
        # Default to 'none' if unknown or empty
        return mapping.get(ros_model_str, rs.distortion.none)

    def preprocess_gp_info(self):
        self.sample_goal = False

        if self.gait_process_forward:
            next_gp = (self.gait_process + self.gp_off) % 1.0

            next_swing_foot_idx = 0 if (next_gp < 0.5) else 1

            if next_swing_foot_idx != self.swing_foot_idx:
                self.gait_process_forward = False
                self.remaining_delay_steps = self.img_delay_steps
            else:
                self.gait_process = next_gp

        if not self.gait_process_forward:

            if self.remaining_delay_steps > 0:
                self.remaining_delay_steps -= 1

            if (self.remaining_delay_steps == 0 and self.swing_foot_on_ground) or (self.gp_paused_steps >= self.max_pause_steps):
                print(f"Gait Paused Steps: {self.gp_paused_steps}")
                self.sample_goal = True
                self.gait_process_forward = True
                self.gait_process = (self.gait_process + self.gp_off) % 1.0
                self.swing_foot_idx = 0 if (self.gait_process < 0.5) else 1
                self.gp_paused_steps = 0

            self.gp_paused_steps += 1

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

    def query_cmd(self, rgb_image, depth_image, joint_pos, image_updated: bool):
        # Update foot contact status first because preprocess_gp_info depends on it
        self.detect_foot_on_ground(joint_pos)

        gait_info = self.preprocess_gp_info()
        targets = []

        if self.sample_goal:
            targets, rgb_image, mask_image = self.detect_foot_target(rgb_image, depth_image)
            if self.move_dir == "STILL":
                self.foot_offset = self._gen_still_cmd()
                self.gaits_to_still = max(0, self.gaits_to_still - 1)
            elif self.move_dir in ["FWD", "BWD"]:
                # self.detect_foot_on_ground(joint_pos)
                # if self.swing_foot_on_ground:
                # if self.remaining_delay_steps == 0:
                depth_image = depth_image * (mask_image > 0)
                # targets, rgb_image, mask_image = self.detect_foot_target(rgb_image, depth_image)
                print(f"#################### STEP {self.img_count} # SWING FOOT {self.swing_foot_idx} ######################")
                if image_updated and len(targets) > 0:
                    self.foot_offset = self._gen_visual_cmd(targets, joint_pos)
                    # set z offset to zero to avoid unexpected behaviors
                    self.foot_offset[0][2] = 0.0
                    self.foot_offset[2][2] = 0.0
                else:
                    print(f"No valid targets detected")
                    self.foot_offset = self._gen_vertical_cmd()
                print(
                    f"Left foot offset from vision:  {np.array(self.foot_offset[0])}, \n"
                    f"Right foot offset from vision: {np.array(self.foot_offset[2])}"
                )

            if self.debug_vis and self.move_dir in ["FWD", "BWD"]:
                mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2RGB)
                # Merge RGB and mask images side by side
                img = np.concatenate((rgb_image, mask_image), axis=1)

                cv2.imwrite(f"./outputs/vis/rgb_mask_{self.img_count}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                self.img_count += 1

                # cv2.imshow("rgb_image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                # cv2.waitKey(1)

            # # TODO remove
            # if self.remaining_delay_steps == 0:
            #     print(
            #         f"#################### STEP {self.img_count} # SWING FOOT {self.swing_foot_idx} ######################"
            #     )
            #     self.foot_offset = self._gen_visual_cmd(targets, joint_pos)
            #     print(
            #         f"Left foot offset from vision: {np.array(self.foot_offset[0])}, "
            #         f"Right foot offset from vision: {np.array(self.foot_offset[2])}"
            #     )

        return self.foot_offset, gait_info, rgb_image, depth_image, targets

    def _gen_visual_cmd(self, targets, joint_pos):
        l_rel_pos, l_rel_xmat, r_rel_pos, r_rel_xmat = self._get_foot_to_cam(joint_pos)
        zero_orn_offset = np.array([1, 0, 0, 0], dtype=np.float32)
        zero_pos_offset = np.zeros(3, dtype=np.float32)

        # Defaut offsets when no feasible target is found
        l_pos_offset = np.array([self.vertical_dist, self.feet_distance, 0.0]) if self.swing_foot_idx == 0 else zero_pos_offset
        r_pos_offset = np.array([self.vertical_dist, -self.feet_distance, 0.0]) if self.swing_foot_idx == 1 else zero_pos_offset
        l_orn_offset = zero_orn_offset
        r_orn_offset = zero_orn_offset

        # Get the zero position of the swing foot w.r.t the stance foot
        if self.swing_foot_idx == 0:
            swing_zero_pos = np.array([0., self.feet_distance, 0.0])
            targets_to_stance_pos = (r_rel_xmat @ targets.T).T + r_rel_pos
        else:
            swing_zero_pos = np.array([0.0, -self.feet_distance, 0.0])
            targets_to_stance_pos = (l_rel_xmat @ targets.T).T + l_rel_pos

        target_ids = np.arange(len(targets))

        # Filter targets too close to the stance foot
        mask = np.linalg.norm(targets_to_stance_pos[:, :2], axis=1) > 0.15
        targets_to_stance_pos = targets_to_stance_pos[mask]
        targets = targets[mask]
        target_ids = target_ids[mask]

        # Sort the targets by distance to the swing foot zero position
        targets_to_zero = targets_to_stance_pos - swing_zero_pos
        sort_inds = np.argsort(np.linalg.norm(targets_to_zero[:, :2], axis=1))

        # for target_idx, target in enumerate(targets_to_stance_pos):
        for idx in sort_inds:
            feasible = True
            is_swing_left = 1 if self.swing_foot_idx == 0 else -1
            current_target_to_stance = targets_to_stance_pos[idx]

            target_dist = np.linalg.norm(current_target_to_stance[:2])
            if target_dist > 0.5 or target_dist < 0.2:
                print(
                    f"Target Idx: {target_ids[idx]}, Target Offset: {current_target_to_stance}, Distance: {target_dist} out of range."
                )
                feasible = False
            else:
                if current_target_to_stance[1] * is_swing_left < (self.feet_distance / 2.0):
                    print(f"Target Idx: {target_ids[idx]}, y range violated {current_target_to_stance[1]}, and feet distance {self.feet_distance / 2.0}.")
                    feasible = False
                elif current_target_to_stance[1] * is_swing_left < (self.feet_distance / 2.0):
                    current_target_to_stance[1] = np.maximum(current_target_to_stance[1] * is_swing_left, self.feet_distance / 2.0) * is_swing_left
                current_target_to_stance[0] = np.minimum(current_target_to_stance[0], 0.45)

            if feasible:
                print(f"Selected Target Idx: {target_ids[idx]}, Target Offset: {current_target_to_stance}")
                l_pos_offset = current_target_to_stance if self.swing_foot_idx == 0 else zero_pos_offset
                r_pos_offset = current_target_to_stance if self.swing_foot_idx == 1 else zero_pos_offset
                l_orn_offset = zero_orn_offset
                r_orn_offset = zero_orn_offset

                mask_current_target = np.ones(len(targets), dtype=bool)
                mask_current_target[idx] = False
                masked_targets_to_stance_pos = targets_to_stance_pos[mask_current_target]
                target_ids = target_ids[mask_current_target]
                if len(masked_targets_to_stance_pos) > 0:
                    sort_ids_next = np.argsort(np.linalg.norm(masked_targets_to_stance_pos[:, :2], axis=1))
                    if len(masked_targets_to_stance_pos) > 1:
                        next_target_to_stance = masked_targets_to_stance_pos[sort_ids_next[1]]
                        target_dir = next_target_to_stance - current_target_to_stance
                        print(
                            f"NNext Target Idx: {target_ids[sort_ids_next[1]]}, Next Target Offset: {next_target_to_stance}"
                        )
                        yaw = np.clip(np.arctan2(target_dir[1], target_dir[0]), np.deg2rad(-45), np.deg2rad(45))
                    elif len(masked_targets_to_stance_pos) > 0:
                        next_target_to_stance = masked_targets_to_stance_pos[sort_ids_next[0]]
                        print(f"Next Target Idx: {target_ids[sort_ids_next[0]]}, Next Target Offset: {next_target_to_stance}")

                        target_dir = next_target_to_stance - current_target_to_stance

                        desired_y_offset = self.feet_distance + 0.05
                        foot_sign = 1 if self.swing_foot_idx == 0 else -1
                        yaw_offset = np.arcsin(
                            -foot_sign * desired_y_offset / (np.maximum(np.linalg.norm(target_dir[:2]), desired_y_offset) + 1e-6)
                        )
                        yaw = np.clip(np.arctan2(target_dir[1], target_dir[0]) - yaw_offset, np.deg2rad(-45), np.deg2rad(45))
                    l_orn_offset = np_R.from_euler("z", yaw).as_quat(scalar_first=True) if self.swing_foot_idx == 0 else l_orn_offset
                    r_orn_offset = np_R.from_euler("z", yaw).as_quat(scalar_first=True) if self.swing_foot_idx == 1 else r_orn_offset

                # if self.swing_foot_idx == 0:

                #     yaw = np.arctan2(target_dir[1], target_dir[0]) - yaw_offset
                #     yaw = np.clip(yaw, np.deg2rad(-30), np.deg2rad(90))
                #     l_orn_offset = np_R.from_euler("z", yaw).as_quat(scalar_first=True)
                # else:
                #     next_l_to_target = l_rel_xmat @ next_target + l_rel_pos
                #     target_dir = next_l_to_target - target
                #     desired_y_offset = self.feet_distance + 0.05
                #     yaw_offset = np.arcsin(
                #         desired_y_offset / (np.maximum(np.linalg.norm(target_dir[:2]), desired_y_offset) + 1e-6)
                #     )
                #     yaw = np.arctan2(target_dir[1], target_dir[0]) - yaw_offset
                #     yaw = np.clip(yaw, np.deg2rad(-90), np.deg2rad(30))
                #     r_orn_offset = np_R.from_euler("z", yaw).as_quat(scalar_first=True)

                # if target_idx < len(targets) - 1:
                #     next_target = targets[target_idx + 1]
                #     if self.swing_foot_idx == 0:
                #         next_r_to_target = r_rel_xmat @ next_target + r_rel_pos
                #         target_dir = next_r_to_target - target
                #         desired_y_offset = self.feet_distance + 0.02
                #         yaw_offset = np.arcsin(
                #             -desired_y_offset / (np.maximum(np.linalg.norm(target_dir[:2]), desired_y_offset) + 1e-6)
                #         )
                #         yaw = np.arctan2(target_dir[1], target_dir[0]) - yaw_offset
                #         yaw = np.clip(yaw, np.deg2rad(-30), np.deg2rad(90))
                #         l_orn_offset = np_R.from_euler("z", yaw).as_quat(scalar_first=True)
                #     else:
                #         next_l_to_target = l_rel_xmat @ next_target + l_rel_pos
                #         target_dir = next_l_to_target - target
                #         desired_y_offset = self.feet_distance + 0.05
                #         yaw_offset = np.arcsin(
                #             desired_y_offset / (np.maximum(np.linalg.norm(target_dir[:2]), desired_y_offset) + 1e-6)
                #         )
                #         yaw = np.arctan2(target_dir[1], target_dir[0]) - yaw_offset
                #         yaw = np.clip(yaw, np.deg2rad(-90), np.deg2rad(30))
                #         r_orn_offset = np_R.from_euler("z", yaw).as_quat(scalar_first=True)

                break

            # for target_idx, target in enumerate(targets):
            # l_to_target = l_rel_xmat @ target + l_rel_pos
            # r_to_target = r_rel_xmat @ target + r_rel_pos
            # l_dist = np.linalg.norm(l_to_target[:2])
            # r_dist = np.linalg.norm(r_to_target[:2])

            # break if both feet are too far
            # if min(l_dist, r_dist) > 0.5:
            #     break

            # Check feasibility for the swing foot
            # feasible = True
            # if self.swing_foot_idx == 0:
            #     stance_target_offset = r_to_target
            #     stance_dist = r_dist
            #     if stance_dist > 0.5 or stance_dist < 0.1:
            #         print(f"Stance Foot Offset: {stance_target_offset}, Distance: {stance_dist} out of range.")
            #         feasible = False
            #     else:
            #         if stance_target_offset[1] < self.feet_distance / 2.0:
            #             print(f"y range violated {stance_target_offset[1]} < {self.feet_distance / 2.0}.")
            #             feasible = False
            #             # Wait for one step
            #             l_pos_offset = np.array([0.0, self.feet_distance, 0.0])
            # else:
            #     stance_target_offset = l_to_target
            #     stance_dist = l_dist
            #     if stance_dist > 0.5 or stance_dist < 0.2:
            #         print(f"Stance Foot Offset: {stance_target_offset}, Distance: {stance_dist} out of range.")
            #         feasible = False
            #     else:
            #         if stance_target_offset[1] > -self.feet_distance / 2.0:
            #             print(f"y range violated {stance_target_offset[1]} > {-self.feet_distance / 2.0}.")
            #             feasible = False
            #             # Wait for one step
            #             r_pos_offset = np.array([0.0, -self.feet_distance, 0.0])

            # # TODO remove
            # feasible = True

            # if feasible:
            #     l_pos_offset = r_to_target if self.swing_foot_idx == 0 else zero_pos_offset
            #     r_pos_offset = l_to_target if self.swing_foot_idx == 1 else zero_pos_offset
            #     l_orn_offset = zero_orn_offset
            #     r_orn_offset = zero_orn_offset

            #     # Determine orientation based on the next target
            #     if target_idx < len(targets) - 1:
            #         next_target = targets[target_idx + 1]
            #         if self.swing_foot_idx == 0:
            #             next_r_to_target = r_rel_xmat @ next_target + r_rel_pos
            #             target_dir = next_r_to_target - r_to_target
            #             yaw_offset = np.arcsin(
            #                 -self.feet_distance / (np.maximum(np.linalg.norm(target_dir[:2]), self.feet_distance) + 1e-6)
            #             )
            #             yaw = np.arctan2(target_dir[1], target_dir[0]) - yaw_offset
            #             yaw = np.clip(yaw, np.deg2rad(-30), np.deg2rad(90))
            #             l_orn_offset = np_R.from_euler("z", yaw).as_quat(scalar_first=True)
            #         else:
            #             next_l_to_target = l_rel_xmat @ next_target + l_rel_pos
            #             target_dir = next_l_to_target - l_to_target
            #             yaw_offset = np.arcsin(
            #                 self.feet_distance / (np.maximum(np.linalg.norm(target_dir[:2]), self.feet_distance) + 1e-6)
            #             )
            #             yaw = np.arctan2(target_dir[1], target_dir[0]) - yaw_offset
            #             yaw = np.clip(yaw, np.deg2rad(-90), np.deg2rad(30))
            #             r_orn_offset = np_R.from_euler("z", yaw).as_quat(scalar_first=True)

            #     break

        return l_pos_offset, l_orn_offset, r_pos_offset, r_orn_offset

    def _get_foot_to_cam(self, joint_pos):
        # set the robot joint positions
        self.data.qpos[7:] = joint_pos
        # self.data.qpos[3:7] = quat
        self.data.qpos[:3] = np.array([0.0, 0.0, 1.0])  # set a fixed height for the base
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # no rotation for the base

        mujoco.mj_fwdPosition(self.model, self.data)

        # Get camera and foot positions
        cam_pos = self.data.cam_xpos[self.cam_id]
        cam_xmat = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        l_foot_pos = self.data.site_xpos[self.left_foot_id]
        l_foot_xmat = self.data.site_xmat[self.left_foot_id].reshape(3, 3)
        l_rel_pos, l_rel_xmat = self.compute_relative_transformation(l_foot_pos, l_foot_xmat, cam_pos, cam_xmat)

        r_foot_pos = self.data.site_xpos[self.right_foot_id]
        r_foot_xmat = self.data.site_xmat[self.right_foot_id].reshape(3, 3)
        r_rel_pos, r_rel_xmat = self.compute_relative_transformation(r_foot_pos, r_foot_xmat, cam_pos, cam_xmat)
        return l_rel_pos, l_rel_xmat, r_rel_pos, r_rel_xmat

    def detect_foot_on_ground(self, joint_pos):
        if self.swing_foot_idx == 0:
            rel_pos, rel_mat = self.fk_foot_to_foot(joint_pos, left_to_right=False)
        else:
            rel_pos, rel_mat = self.fk_foot_to_foot(joint_pos, left_to_right=True)
        if not self.gait_process_forward and rel_pos[2] < -0.02:
            self.swing_foot_on_ground = True
            print("Swing to Stance Foot Rel Pos: ", rel_pos)
        else:
            self.swing_foot_on_ground = False

    def fk_foot_to_foot(self, joint_pos, left_to_right: bool, only_yaw: bool = False):
        if left_to_right:
            from_id = self.left_foot_id
            to_id = self.right_foot_id
        else:
            from_id = self.right_foot_id
            to_id = self.left_foot_id

        self.data.qpos[7:] = joint_pos
        self.data.qpos[:3] = np.array([0.0, 0.0, 1.0])  # set a fixed height for the base
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # no rotation for the base
        mujoco.mj_fwdPosition(self.model, self.data)

        # Get foot positions
        from_pos = self.data.site_xpos[from_id]
        from_xmat = self.data.site_xmat[from_id].reshape(3, 3)
        to_pos = self.data.site_xpos[to_id]
        to_xmat = self.data.site_xmat[to_id].reshape(3, 3)

        if only_yaw:
            from_yaw = np.arctan2(from_xmat[1, 0], from_xmat[0, 0])
            to_yaw = np.arctan2(to_xmat[1, 0], to_xmat[0, 0])
            from_xmat = np_R.from_euler("z", from_yaw).as_matrix()
            to_xmat = np_R.from_euler("z", to_yaw).as_matrix()
        rel_pos, rel_xmat = self.compute_relative_transformation(from_pos, from_xmat, to_pos, to_xmat)
        return rel_pos, rel_xmat

    @staticmethod
    def compute_relative_transformation(origin_pos, origin_xmat, target_pos, target_xmat):
        relative_pos = origin_xmat.T @ (target_pos - origin_pos)
        relative_xmat = origin_xmat.T @ target_xmat
        return relative_pos, relative_xmat

    def detect_foot_target(self, rgb_image, depth_image):
        """
        Detect circular or elliptical targets using shape (not color).
        Returns a list of 3D points (camera frame) for detected targets, sorted by area desc.
        """
        # Convert to grayscale and denoise
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

        # mask = cv2.inRange(hsv, (107, 100, 0), (120, 255, 255)) # Range for simulation
        mask = cv2.inRange(hsv, (90, 88, 100), (120, 255, 255)) # Range for real world

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours for visualization
        contours = contours[0] if len(contours) == 2 else contours[1]

        targets = []
        N = 3
        h, w = depth_image.shape

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:  # Minimum area threshold to filter noise
                continue

            # fitEllipse requires at least 5 points
            if len(cnt) < 5:
                continue

            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)

            if MA / ma < 0.5:  # Filter out non-elliptical shapes
                continue

            # 1. Calculate the ROI (Region of Interest) bounds with safety clipping
            x_start = int(max(0, x - N))
            x_end = int(min(w, x + N))
            y_start = int(max(0, y - N))
            y_end = int(min(h, y + N))

            # 2. Extract the neighborhood depth slice
            depth_roi = depth_image[y_start:y_end, x_start:x_end]
            valid_mask = (depth_roi > 0.1) & (depth_roi < 5.0)
            valid_depths = depth_roi[valid_mask]

            if valid_depths.size == 0:
                continue # Skip if no pixels in the window meet the criteria

            # 4. Use Median to calculate the representative depth
            target_depth = np.median(valid_depths)

            targets.append((x, y, target_depth, MA, ma, angle))
        if len(targets) == 0:
            return [], rgb_image, mask

        # Distance sorting to bottom center of the image
        # targets_to_sort = targets[:, :2] - np.array([self.rs_intrinsics.width / 2, self.rs_intrinsics.height])
        # sort_ids = np.argsort(np.linalg.norm(targets_to_sort, axis=1))
        # targets = targets[sort_ids]

        # Assuming targets is a numpy array of [u, v, depth]
        points_3d = [rs.rs2_deproject_pixel_to_point(self.rs_intrinsics, [t[0], t[1]], t[2]) for t in targets]

        # Convert Camera frame to MuJoCo frame
        targets_cam = np.array(points_3d)
        targets_cam[:, 1] = -targets_cam[:, 1]
        targets_cam[:, 2] = -targets_cam[:, 2]

        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # for i, target in enumerate(targets[sorted_indices][:3]):
        for i, target in enumerate(targets):
            x, y, depth, MA, ma, angle = target
            bgr = cv2.ellipse(bgr, (int(x), int(y)), (int(MA / 2), int(ma / 2)), angle, 0, 360, 
                              (0, 0, 255), 2)
            bgr = cv2.circle(bgr, (int(x), int(y)), 3, (0, 255, 255), -1)
            bgr = cv2.putText(bgr, f"ID:{i}", (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, (0, 0, 255), 1, cv2.LINE_AA)
            
            # cv2.imshow("Detection", bgr)
            # cv2.waitKey(1)
        return targets_cam, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mask

    def reset(self):
        super().reset()
        self.gait_process_forward = True
        self.swing_foot_on_ground = True
        self.gp_paused_steps = 0
        self.remaining_delay_steps = 0


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
