import numpy as np
from scipy.spatial.transform import Rotation as np_R
import copy
import mujoco
import cv2


class GaitGenerator:
    def __init__(
        self,
        feet_distance: float = 0.2,
        stop_steps: int = 2,
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

    def preprocess_gp_info(self, gp: float):
        swing_foot_idx = 0 if (gp < 0.5) else 1

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
            gait_info = np.array([np.cos(2 * np.pi * gp), np.sin(2 * np.pi * gp)])
        else:
            gait_info = np.array([0.0, 0.0])
        return gait_info

    def query_cmd(self, gp: float = 0.0):
        gait_info = self.preprocess_gp_info(gp=gp)

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
        max_evil_movement = self.lateral_dist / 2.0

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
        feet_distance: float = 0.2,
        stop_steps: int = 2,
    ):
        super().__init__(feet_distance=feet_distance, stop_steps=stop_steps)

        self.model = copy.deepcopy(robot_model)
        self.data = copy.deepcopy(robot_data)

        self.left_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        self.right_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")

        self.cam_info = {
            "width": cam_width,
            "height": cam_height,
            "focal": cam_height / (2.0 * np.tan(np.deg2rad(self.model.cam_fovy[self.cam_id]) / 2.0)),
            "principal_point": (cam_width / 2.0, cam_height / 2.0),
        }
        self.cam_intrinsics = np.array(
            [
                [self.cam_info["focal"], 0, self.cam_info["principal_point"][0]],
                [0, self.cam_info["focal"], self.cam_info["principal_point"][1]],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

    def query_cmd(self, rgb_image, depth_image, joint_pos, gp: float):
        gait_info = self.preprocess_gp_info(gp=gp)

        if self.sample_goal:
            if self.move_dir == "STILL":
                self.foot_offset = self._gen_still_cmd()
                self.gaits_to_still = max(0, self.gaits_to_still - 1)
            elif self.move_dir in ["FWD", "BWD"]:
                # detect the foot target
                targets, rgb_image = self.detect_foot_target(rgb_image, depth_image)
                if len(targets) > 0:
                    self.foot_offset = self._gen_visual_cmd(targets, joint_pos)
                else:
                    self.foot_offset = self._gen_vertical_cmd()

        return self.foot_offset, gait_info, rgb_image

    def _gen_visual_cmd(self, targets, joint_pos):
        l_rel_pos, l_rel_xmat, r_rel_pos, r_rel_xmat = self._get_foot_to_cam(joint_pos)
        zero_orn_offset = np.array([1, 0, 0, 0], dtype=np.float32)
        zero_pos_offset = np.zeros(3, dtype=np.float32)

        l_pos_offset = np.array([0, self.feet_distance, 0.0]) if self.swing_foot_idx == 0 else zero_pos_offset
        r_pos_offset = np.array([0, -self.feet_distance, 0.0]) if self.swing_foot_idx == 1 else zero_pos_offset
        l_orn_offset = zero_orn_offset
        r_orn_offset = zero_orn_offset

        for target_idx, target in enumerate(targets):
            l_to_target = l_rel_xmat @ target + l_rel_pos
            r_to_target = r_rel_xmat @ target + r_rel_pos
            l_dist = np.linalg.norm(l_to_target)
            r_dist = np.linalg.norm(r_to_target)

            # break if both feet are too far
            if min(l_dist, r_dist) > 0.5:
                break

            # Check feasibility for the swing foot
            feasible = True
            if self.swing_foot_idx == 0:
                stance_target_offset = r_to_target
                stance_dist = r_dist
                if stance_target_offset[1] < self.feet_distance / 2.0 or stance_dist > 0.5 or stance_dist < 0.2:
                    feasible = False
            else:
                stance_target_offset = l_to_target
                stance_dist = l_dist
                if stance_target_offset[1] > -self.feet_distance / 2.0 or stance_dist > 0.5 or stance_dist < 0.2:
                    feasible = False

            if feasible:
                l_pos_offset = r_to_target if self.swing_foot_idx == 0 else zero_pos_offset
                r_pos_offset = l_to_target if self.swing_foot_idx == 1 else zero_pos_offset
                l_orn_offset = zero_orn_offset
                r_orn_offset = zero_orn_offset

                # Determine orientation based on the next target
                if target_idx < len(targets) - 1:
                    next_target = targets[target_idx + 1]
                    if self.swing_foot_idx == 0:
                        next_r_to_target = r_rel_xmat @ next_target + r_rel_pos
                        target_dir = next_r_to_target - r_to_target
                        yaw_offset = np.arcsin(-self.feet_distance / (np.linalg.norm(target_dir[:2]) + 1e-6))
                        yaw = np.arctan2(target_dir[1], target_dir[0]) - yaw_offset
                        yaw = np.clip(yaw, np.deg2rad(-30), np.deg2rad(90))
                        l_orn_offset = np_R.from_euler("z", yaw).as_quat(scalar_first=True)
                    else:
                        next_l_to_target = l_rel_xmat @ next_target + l_rel_pos
                        target_dir = next_l_to_target - l_to_target
                        yaw_offset = np.arcsin(self.feet_distance / (np.linalg.norm(target_dir[:2]) + 1e-6))
                        yaw = np.arctan2(target_dir[1], target_dir[0]) - yaw_offset
                        yaw = np.clip(yaw, np.deg2rad(-90), np.deg2rad(30))
                        r_orn_offset = np_R.from_euler("z", yaw).as_quat(scalar_first=True)

                break

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

        mask = cv2.inRange(hsv, (107, 100, 0), (120, 255, 255))

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours for visualization
        contours = contours[0] if len(contours) == 2 else contours[1]
        targets = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:  # Minimum area threshold to filter noise
                continue

            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)

            if MA / ma < 0.5:  # Filter out non-elliptical shapes
                continue

            targets.append((x, y, MA, ma, angle))

        if len(targets) == 0:
            return [], rgb_image

        targets = np.array(targets, dtype=np.float32)

        # Sort the targets by distance to the bottom center of the image
        x_pixel = targets[:, 0]
        y_pixel = targets[:, 1]
        z = depth_image[
            np.clip(y_pixel.round().astype(int), 0, depth_image.shape[0] - 1),
            np.clip(x_pixel.round().astype(int), 0, depth_image.shape[1] - 1),
        ]

        x_cam = (x_pixel - self.cam_info["principal_point"][0]) * z / self.cam_info["focal"]
        y_cam = (y_pixel - self.cam_info["principal_point"][1]) * z / self.cam_info["focal"]
        targets_cam = np.stack([x_cam, -y_cam, -z], axis=1)
        distances = np.linalg.norm(targets_cam, axis=1)
        sorted_indices = np.argsort(distances)
        targets_cam = targets_cam[sorted_indices]

        bgr = cv2.cvtColor(rgb_imageGaitGenerator, cv2.COLOR_RGB2BGR)
        for target in targets[sorted_indices][:3]:
            x, y, MA, ma, angle = target
            bgr = cv2.ellipse(bgr, (int(x), int(y)), (int(MA / 2), int(ma / 2)), angle, 0, 360, (0, 255, 0), 1)
        return targets_cam, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class GoalReachingGaitGenerator(GaitGenerator):
    def __init__(self, model, data, max_angle, feet_distance: float = 0.2, stop_steps: int = 2, z_off: float = 0.0):
        super().__init__(feet_distance=feet_distance, stop_steps=stop_steps)
        self.model = copy.deepcopy(model)
        self.data = copy.deepcopy(data)

        self.left_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        self.right_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

        self.first_step = True
        self.gp_shift = 0.0

        self.max_angle = max_angle
        self.turning_threshold = self.max_angle * 2

        self.z_off = z_off

    def query_cmd(self, goal_pos, q_pos, gp: float):
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
            self.gp_shift = (desired_gp - (gp % 1.0)) % 1.0

        gp = (gp + self.gp_shift) % 1.0
        gait_info = self.preprocess_gp_info(gp=gp)

        stance_foot = self.left_foot_id if self.swing_foot_idx == 1 else self.right_foot_id

        stance_foot_pos = self.data.site_xpos[stance_foot]
        stance_foot_xmat = self.data.site_xmat[stance_foot].reshape(3, 3)

        target_to_stance = goal_pos - stance_foot_pos
        base_st_to_stance = q_pos[:3] - stance_foot_pos

        goal_st = stance_foot_xmat.T @ target_to_stance
        base_st = stance_foot_xmat.T @ base_st_to_stance

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
                    self.z_off = goal_pos[2]
                    self.gaits_to_still = self.stop_steps
                    self.foot_offset = self._gen_goal_reaching_cmd(
                        base_st = base_st[:2],
                        target_st=goal_st[:2],
                    )

        return self.foot_offset, gait_info

    def _gen_goal_reaching_cmd(self, base_st: np.ndarray, target_st: np.ndarray):
        zero_orn_offset = np.array([1, 0, 0, 0], dtype=np.float32)
        zero_pos_offset = np.zeros(3, dtype=np.float32)

        base_to_target_st = target_st - base_st
        head_dir = base_to_target_st / np.linalg.norm(base_to_target_st) + 1e-6
        head_yaw = np.arctan2(head_dir[1], head_dir[0]) + np.pi * (np.sign(self.vertical_dist) < 0)
        desired_yaw = np.clip(self._wrap_to_pi(head_yaw), -self.max_angle, self.max_angle)
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

        swing_pos = np.array([x, y, self.z_off], dtype=np.float32)


        l_pos_offset = swing_pos if self.swing_foot_idx == 0 else zero_pos_offset
        r_pos_offset = swing_pos if self.swing_foot_idx == 1 else zero_pos_offset
        l_orn_offset = swing_orn_offset if self.swing_foot_idx == 0 else zero_orn_offset
        r_orn_offset = swing_orn_offset if self.swing_foot_idx == 1 else zero_orn_offset

        return l_pos_offset, l_orn_offset, r_pos_offset, r_orn_offset

    @staticmethod
    def _wrap_to_pi(a: float) -> float:
        return (a + np.pi) % (2 * np.pi) - np.pi