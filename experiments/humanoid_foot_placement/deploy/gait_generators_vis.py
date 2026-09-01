from gait_generators import GaitGenerator
import mujoco
import copy
import numpy as np
import pyrealsense2 as rs
import os
import cv2
from scipy.spatial.transform import Rotation as np_R


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
        clahe_enhance: bool = False,
        debug_vis: bool = False,
    ):
        super().__init__(
            feet_distance=feet_distance,
            stop_steps=stop_steps,
            gait_frequency=gait_frequency,
            policy_dt=policy_dt,
            is_gp_adaptive=is_gp_adaptive,
            min_gp_delta=min_gp_delta,
            max_gp_delta=max_gp_delta,
        )

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
        self.clahe_enhance = clahe_enhance

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

            if (self.remaining_delay_steps == 0 and self.swing_foot_on_ground) or (
                self.gp_paused_steps >= self.max_pause_steps
            ):
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
                print(
                    f"#################### STEP {self.img_count} # SWING FOOT {self.swing_foot_idx} ######################"
                )
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

        return self.foot_offset, gait_info, rgb_image, depth_image, targets

    def _gen_visual_cmd(self, targets, joint_pos):
        l_rel_pos, l_rel_xmat, r_rel_pos, r_rel_xmat = self._get_foot_to_cam(joint_pos)
        zero_orn_offset = np.array([1, 0, 0, 0], dtype=np.float32)
        zero_pos_offset = np.zeros(3, dtype=np.float32)

        # Defaut offsets when no feasible target is found
        l_pos_offset = (
            np.array([self.vertical_dist, self.feet_distance, 0.0]) if self.swing_foot_idx == 0 else zero_pos_offset
        )
        r_pos_offset = (
            np.array([self.vertical_dist, -self.feet_distance, 0.0]) if self.swing_foot_idx == 1 else zero_pos_offset
        )
        l_orn_offset = zero_orn_offset
        r_orn_offset = zero_orn_offset

        # Get the zero position of the swing foot w.r.t the stance foot
        if self.swing_foot_idx == 0:
            swing_zero_pos = np.array([0.0, self.feet_distance, 0.0])
            targets_to_stance_pos = (r_rel_xmat @ targets.T).T + r_rel_pos
        else:
            swing_zero_pos = np.array([0.0, -self.feet_distance, 0.0])
            targets_to_stance_pos = (l_rel_xmat @ targets.T).T + l_rel_pos

        target_ids = np.arange(len(targets))

        # Filter targets too close to the stance foot
        mask = np.linalg.norm(targets_to_stance_pos[:, :2], axis=1) > 0.15
        # Mask out target too high or too far
        mask = np.logical_and(mask, targets_to_stance_pos[:, 2] < 0.15)
        mask = np.logical_and(mask, np.linalg.norm(targets_to_stance_pos[:, :2], axis=1) < 1.0)
        targets_to_stance_pos = targets_to_stance_pos[mask]
        targets = targets[mask]
        target_ids = target_ids[mask]

        if len(targets) == 0:
            print("No valid targets after filtering too close to stance foot.")
            return l_pos_offset, l_orn_offset, r_pos_offset, r_orn_offset

        # Sort the targets by distance to the swing foot zero position
        targets_to_zero = targets_to_stance_pos - swing_zero_pos
        sort_inds = np.argsort(np.linalg.norm(targets_to_zero[:, :2], axis=1))
        mask_current_target = np.ones(len(targets), dtype=bool)

        # for target_idx, target in enumerate(targets_to_stance_pos):
        for idx in sort_inds:
            feasible = True
            is_swing_left = 1 if self.swing_foot_idx == 0 else -1
            current_target_to_stance = targets_to_stance_pos[idx]

            target_dist = np.linalg.norm(current_target_to_stance[:2])
            if target_dist > 0.6 or target_dist < 0.2:
                print(
                    f"Target Idx: {target_ids[idx]}, Target Offset: {current_target_to_stance}, Distance: {target_dist} out of range."
                )
                feasible = False

            # if current_target_to_stance[1] * is_swing_left < (self.feet_distance / 2.0):
            if current_target_to_stance[1] * is_swing_left < 0:
                print(
                    f"Target Idx: {target_ids[idx]}, y range violated {current_target_to_stance[1]}, and feet distance {self.feet_distance / 2.0}."
                )
                feasible = False

            mask_current_target[idx] = False
            if feasible:
                current_target_to_stance[0] = np.minimum(current_target_to_stance[0], 0.45)
                current_target_to_stance[1] = (
                    np.maximum(current_target_to_stance[1] * is_swing_left, self.feet_distance / 2.0) * is_swing_left
                )
                print(f"Selected Target Idx: {target_ids[idx]}, Target Offset: {current_target_to_stance}")
                l_pos_offset = current_target_to_stance if self.swing_foot_idx == 0 else zero_pos_offset
                r_pos_offset = current_target_to_stance if self.swing_foot_idx == 1 else zero_pos_offset

                break

        # Determine the angle of the orientation offset
        yaw = 0.0
        masked_targets_to_stance_pos = targets_to_stance_pos[mask_current_target]
        masked_target_ids = target_ids[mask_current_target]

        if self.swing_foot_idx == 0:
            masked_targets_to_swing_target = masked_targets_to_stance_pos - l_pos_offset
        else:
            masked_targets_to_swing_target = masked_targets_to_stance_pos - r_pos_offset

        if len(masked_targets_to_stance_pos) > 1:
            # sort_ids_next = np.argsort(np.linalg.norm(masked_targets_to_swing_target[:, :2], axis=1))
            # sort_ids_next = np.argsort(masked_targets_to_stance_pos[:, 0])
            # next_target_to_stance = masked_targets_to_stance_pos[sort_ids_next[1]]
            swing_foot_yaw = np.arctan2(masked_targets_to_swing_target[:3, 1], masked_targets_to_swing_target[:3, 0])
            sort_ids_next = np.argsort(np.abs(swing_foot_yaw))
            next_target_id = sort_ids_next[0]
            next_target_to_stance = masked_targets_to_stance_pos[next_target_id]

            target_dir = next_target_to_stance - current_target_to_stance
            print(f"NNext Target Idx: {masked_target_ids[next_target_id]}, Next Target Offset: {next_target_to_stance}")
            yaw = np.clip(np.arctan2(target_dir[1], target_dir[0]), np.deg2rad(-45), np.deg2rad(45))
        elif len(masked_targets_to_stance_pos) > 0:
            next_target_to_stance = masked_targets_to_stance_pos[0]
            print(f"Next Target Idx: {masked_target_ids[0]}, Next Target Offset: {next_target_to_stance}")

            target_dir = next_target_to_stance - current_target_to_stance

            desired_y_offset = self.feet_distance + 0.05
            foot_sign = 1 if self.swing_foot_idx == 0 else -1
            yaw_offset = np.arcsin(
                -foot_sign * desired_y_offset / (np.maximum(np.linalg.norm(target_dir[:2]), desired_y_offset) + 1e-6)
            )
            yaw = np.clip(np.arctan2(target_dir[1], target_dir[0]) - yaw_offset, np.deg2rad(-45), np.deg2rad(45))

        l_orn_offset = (
            np_R.from_euler("z", yaw).as_quat(scalar_first=True) if self.swing_foot_idx == 0 else l_orn_offset
        )
        r_orn_offset = (
            np_R.from_euler("z", yaw).as_quat(scalar_first=True) if self.swing_foot_idx == 1 else r_orn_offset
        )


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
        # if not self.gait_process_forward and rel_pos[2] < 0.02:
        if rel_pos[2] < 0.02:
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

        if self.clahe_enhance:
            # 1. Convert to Lab to isolate "Blueness" from "Brightness"
            lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            # 2. Apply CLAHE to the L channel to normalize lighting
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)

            # 3. Merge back and convert to HSV for easier color intuitive masking
            # OR mask directly on the 'b' channel from Lab.
            enhanced_rgb = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2RGB)
            hsv = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2HSV)
        else:
            # Convert to grayscale and denoise
            hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

        # mask = cv2.inRange(hsv, (107, 100, 0), (120, 255, 255)) # Range for simulation
        mask = cv2.inRange(hsv, (90, 40, 80), (130, 255, 255))  # Range for real world

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Close small holes inside the circle
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

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
            if len(cnt) < 20:
                continue

            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)

            if MA / ma < 0.6:  # Filter out non-elliptical shapes
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
                continue  # Skip if no pixels in the window meet the criteria

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
            bgr = cv2.ellipse(bgr, (int(x), int(y)), (int(MA / 2), int(ma / 2)), angle, 0, 360, (0, 0, 255), 2)
            bgr = cv2.circle(bgr, (int(x), int(y)), 3, (0, 255, 255), -1)
            bgr = cv2.putText(
                bgr, f"ID:{i}", (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
            )

        # cv2.imshow("Detection", bgr)
        # cv2.imshow("Mask", mask)
        # cv2.waitKey(1)
        return targets_cam, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mask

    def reset(self):
        super().reset()
        self.gait_process_forward = True
        self.swing_foot_on_ground = True
        self.gp_paused_steps = 0
        self.remaining_delay_steps = 0
