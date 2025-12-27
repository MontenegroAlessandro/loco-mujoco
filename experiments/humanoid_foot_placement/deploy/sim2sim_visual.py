from shutil import move
import time
import os
import sys

# Add parent directory to import path to find lmj and other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

import mujoco.viewer
import mujoco
import numpy as np
import yaml
import hydra
from omegaconf import DictConfig  # Using omegaconf.DictConfig for hydra config type
import pickle  # Still needed for PPOJax.load_agent
import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation as np_R
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast
import copy
import cv2

from loco_mujoco.algorithms import PPOJax


class LMJPolicy:
    def __init__(self, policy_path: str) -> None:  # Removed control_func_path
        # Load agent configuration and state from the policy checkpoint
        agent_conf, agent_state = PPOJax.load_agent(policy_path)

        # Removed: self.control_func = pickle.load(fileobj) as it's no longer used

        # Set policy to be deterministic by setting log_std to negative infinity
        train_state = agent_state.train_state
        train_state.params["log_std"] = np.ones_like(train_state.params["log_std"]) * -np.inf

        key = jax.random.key(0)
        key, _rng = jax.random.split(key)

        self.agent_conf = agent_conf
        self.train_state = train_state
        self._rng = _rng

        # Precompute the network apply function to avoid passing non-hashable objects
        self.network_apply = agent_conf.network.apply

        # Define the JIT-compiled function once for performance
        self._jit_sample_action = jax.jit(self._sample_action, static_argnames=["network_apply"])
        print("Policy loaded and JIT function compiled.")

    @staticmethod
    def _sample_action(network_apply, params, run_stats, rng, obs):  # MODIFIED: Removed batch_stats
        """This function is JIT-compiled for speed."""
        # MODIFIED: Removed batch_stats from the dictionary and mutable list
        y, updates = network_apply({"params": params, "run_stats": run_stats}, obs, mutable=["run_stats"])
        pi, _ = y
        a = pi.mode()  # Get the deterministic action
        a = jnp.atleast_2d(a)
        return a

    def predict_action(self, obs):
        """Uses the precompiled JIT function to get the action."""
        # MODIFIED: Removed self.train_state.batch_stats from the call
        a = self._jit_sample_action(
            self.network_apply, self.train_state.params, self.train_state.run_stats, self._rng, obs
        )
        return a


def quat_rotate_inverse(q, v):
    """Rotates a vector by the inverse of a quaternion. MuJoCo quaternions are [w, x, y, z]."""
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates PD control torques."""
    return (target_q - q) * kp + (target_dq - dq) * kd


class GaitGenerator:
    def __init__(
        self,
        feet_distance: float = 0.2,
        vertical_dist: float = 0.1,
        lateral_dist: float = 0.3,
        steering_angle: float = 0.0,
        stop_steps: int = 2,
    ):
        # map the parameters
        self.feet_distance = feet_distance
        self.vertical_dist = vertical_dist
        self.lateral_dist = lateral_dist
        self.steering_angle = steering_angle
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
                self.foot_offset = self._gen_still_cmd(gp=gp)
                self.gaits_to_still = max(0, self.gaits_to_still - 1)
            elif self.move_dir in ["FWD", "BWD"]:
                self.foot_offset = self._gen_vertical_cmd(gp=gp)
            elif self.move_dir in ["LEFT", "RIGHT"]:
                self.foot_offset = self._gen_lateral_cmd(gp=gp)
            elif self.move_dir in ["DIAG-L", "DIAG-R"]:
                direction = 1 if self.move_dir == "DIAG-L" else -1
                self.foot_offset = self._gen_diag_cmd(gp=gp, direction=direction)
        return self.foot_offset, gait_info
    
    def _gen_still_cmd(self, gp: float = 0.0):
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

    def _gen_vertical_cmd(self, gp: float = 0.0, direction: int = 1):# -> tuple[NDArray[Any] | NDArray[floating[_32Bit]], Any | NDA...:
        # no changing targets
        zero_orn_offset = np.array([1, 0, 0, 0], dtype=np.float32)
        zero_pos_offset = np.zeros(3, dtype=np.float32)

        # adjust the steering angle
        steering_angle = np.clip(self.steering_angle, -np.pi, np.pi)
        steering_foot_idx = 0 if (steering_angle >= 0 and steering_angle <= np.pi) else 1
        steering_orn_offset = np_R.from_euler("z", steering_angle).as_quat(scalar_first=True)

        # pos gen
        l_pos_offset = np.array([self.vertical_dist, self.feet_distance, 0]) if self.swing_foot_idx == 0 else zero_pos_offset
        r_pos_offset = (
            np.array([self.vertical_dist, -self.feet_distance, 0]) if self.swing_foot_idx == 1 else zero_pos_offset
        )
        # orn gen
        l_orn_offset = steering_orn_offset if steering_foot_idx == 0 else zero_orn_offset
        r_orn_offset = steering_orn_offset if steering_foot_idx == 1 else zero_orn_offset

        return l_pos_offset, l_orn_offset, r_pos_offset, r_orn_offset

    def _gen_lateral_cmd(self, gp: float = 0.0):
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

    def _gen_diag_cmd(self, gp: float = 0.0, direction: int = 1):
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
            self.lateral_dist = float(np.clip(self.lateral_dist + self.teleop["lat_step"], self.teleop["lat_min"], self.teleop["lat_max"]))
            self.teleop["mov_dir"] = "LEFT"
            print(f"[teleop] Lateral Distance: {self.lateral_dist:.2f}")

        elif keycode == RIGHT_ARROW:
            self.lateral_dist = float(np.clip(self.lateral_dist - self.teleop["lat_step"], self.teleop["lat_min"], self.teleop["lat_max"]))
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
    def __init__(self, robot_model, robot_data, cam_width, cam_height,
                 feet_distance: float = 0.2, vertical_dist: float = 0.1, lateral_dist: float = 0.3, 
                 steering_angle: float = 0.0, stop_steps: int = 2):
        super().__init__(feet_distance = feet_distance, vertical_dist = vertical_dist, lateral_dist = lateral_dist, 
                         steering_angle = steering_angle, stop_steps = stop_steps)

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
            [[self.cam_info["focal"], 0, self.cam_info["principal_point"][0]],
             [0, self.cam_info["focal"], self.cam_info["principal_point"][1]],
             [0, 0, 1]], dtype=np.float32
             )

    def query_cmd(self, rgb_image, depth_image, joint_pos, gp: float):
        gait_info = self.preprocess_gp_info(gp=gp)

        if self.sample_goal:
            if self.move_dir == "STILL":
                self.foot_offset = self._gen_still_cmd(gp=gp)
                self.gaits_to_still = max(0, self.gaits_to_still - 1)
            elif self.move_dir in ["FWD", "BWD"]:
                # detect the foot target
                targets, rgb_image = self.detect_foot_target(rgb_image, depth_image)
                if len(targets) > 0:
                    self.foot_offset = self._gen_visual_cmd(targets, joint_pos, gp)
                else:
                    self.foot_offset = self._gen_vertical_cmd(gp=gp)

        return self.foot_offset, gait_info, rgb_image

    def _gen_visual_cmd(self, targets, joint_pos, gp):
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
                        yaw_offset = np.arcsin(-self.feet_distance/(np.linalg.norm(target_dir[:2]) + 1e-6))
                        yaw = np.arctan2(target_dir[1], target_dir[0]) - yaw_offset
                        yaw = np.clip(yaw, np.deg2rad(-30), np.deg2rad(90))
                        l_orn_offset = np_R.from_euler("z", yaw).as_quat(scalar_first=True)
                    else:
                        next_l_to_target = l_rel_xmat @ next_target + l_rel_pos
                        target_dir = next_l_to_target - l_to_target
                        yaw_offset = np.arcsin(self.feet_distance/(np.linalg.norm(target_dir[:2]) + 1e-6))
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

        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        for target in targets[sorted_indices][:3]:
            x, y, MA, ma, angle = target
            bgr = cv2.ellipse(bgr, (int(x), int(y)), (int(MA / 2), int(ma / 2)), angle, 0, 360, (0, 255, 0), 1)
        return targets_cam, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@hydra.main(config_name="config_sim2sim_visual.yaml")
def main(config: DictConfig):

    xml_path = config["xml_path"]
    simulation_duration = config["simulation_duration"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    agent_path = config["agent_path"]

    kps = np.array(config["lmj_kps"], dtype=np.float32)
    kds = np.array(config["lmj_kds"], dtype=np.float32)

    default_angles = np.array(config["default_angles"], dtype=np.float32)
    min_angles = np.array(config["min_angles"], dtype=np.float32)
    max_angles = np.array(config["max_angles"], dtype=np.float32)

    num_qj = len(default_angles)  # Number of actuated joints (23)
    base_num_actions = config["num_actions"]  # This is also 23

    cmd_params = config["command"]

    # --- Load Policy ---
    # Initialize Hydra to access environment config used during training
    # The config_path should point to the directory containing your hydra config files
    # hydra.initialize(config_path="./") # Adjust path if your hydra config is elsewhere
    lmj_hydra_config = hydra.compose(config_name="conf_t1")
    policy = LMJPolicy(policy_path=agent_path)

    # Determine actual num_actions and observation size based on policy's environment config
    num_actions = base_num_actions

    # Calculate the total observation size for policy warmup and runtime
    # obs = [projected_gravity (3), qj (num_qj), base_ang_vel (3), dqj (num_qj), action (num_actions), cmd (6)]
    num_obs = 3 + num_qj + 3 + num_qj + num_actions + 6

    print(f"Policy expects an observation size of: {num_obs}")
    print("Warming up the policy network for JIT compilation...")
    total_obs = max(policy.agent_conf.network.actor_obs_ind.max(), policy.agent_conf.network.critic_obs_ind.max()) + 1
    for _ in range(500):  # Reduced warmup steps from 1000 to 500
        dummy_obs = jnp.zeros((1, total_obs), dtype=np.float32)
        _ = policy.predict_action(dummy_obs)
    print("Warmup complete.")

    # Load robot model
    spec = mujoco.MjSpec.from_file(xml_path)
    wb = spec.worldbody

    # ==================================================OBSTACLEs==================================================
    # this part is only needed for initialization, then the boxes will be added when doing the reset
    wb.add_site(
        name=f"foot_0",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.1, 0.04, 0.001),  # *
        pos=(0.1, 0.0, 0.0),  # **
        quat=(0, 0, 0, 1),
        group=1,
        rgba=(1.0, 0.5, 0.0, 0.5),
    )

    wb.add_site(
        name=f"foot_1",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.1, 0.04, 0.001),  # *
        pos=(0.1, 0.0, 0.0),  # **
        quat=(0, 0, 0, 1),
        group=1,
        rgba=(1.0, 1.0, 0.0, 0.5),
    )

    # Add visual sites as foot targets
    target_dist = 0.25
    target_angle_range = 45  # degrees
    target_site_pos = np.zeros(3)
    angle = 0
    site_idx = 0
    for i in range(10):
        if i > 5:
            angle += np.random.uniform(-target_angle_range, target_angle_range)
        target_site_pos += target_dist * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0.0])
        feet_pos = target_site_pos + cmd_params["feet_distance"] / 2 * np.array(
            [np.cos(np.deg2rad(angle + (-1) ** i * 90)), np.sin(np.deg2rad(angle + (-1) ** i * 90)), 0.0]
        )
        wb.add_site(
            name=f"target_{i}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=(0.08, 0.005, 0.0),  # *
            pos=feet_pos,
            quat=(0, 0, 0, 1),
            group=0,
            rgba=(0.0, 0.0, 1.0, 1.0),
        )

    # get model spec
    # delete all geoms whose names end in "_col" from spec
    for geom in spec.geoms:
        if geom.name.endswith("_col"):
            geom.delete()

    m = spec.compile()
    d = mujoco.MjData(m)
    left_foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    right_foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    # # --- Initialize Simulation ---
    m.opt.timestep = simulation_dt

    cam_width = 320
    cam_height = 200
    rgb_renderer = mujoco.Renderer(m, width=cam_width, height=cam_height)
    depth_renderer = mujoco.Renderer(m, width=cam_width, height=cam_height)
    depth_renderer.enable_depth_rendering()
    rgb_renderer._scene_option.sitegroup[1] = 0
    depth_renderer._scene_option.sitegroup[1] = 0

    rgb_viewport = mujoco.MjrRect(920, 0, cam_width, cam_height)
    depth_viewport = mujoco.MjrRect(1240, 0, cam_width, cam_height)
    # Set initial robot state from policy's environment config
    try:
        initial_qpos = np.array(lmj_hydra_config.experiment.env_params.init_state_params.qpos_init, dtype=np.float32)
        initial_qvel = np.array(lmj_hydra_config.experiment.env_params.init_state_params.qvel_init, dtype=np.float32)
        if len(initial_qpos) == m.nq and len(initial_qvel) == m.nv:
            d.qpos[:] = initial_qpos
            d.qvel[:] = initial_qvel
            print(initial_qpos)
            print("Initial state (qpos, qvel) loaded from policy's environment config.")
        else:
            print(
                f"Warning: Initial qpos/qvel length mismatch with model. "
                f"Config qpos: {len(initial_qpos)} (expected {m.nq}), "
                f"qvel: {len(initial_qvel)} (expected {m.nv}). Using default MuJoCo init."
            )
    except (AttributeError, KeyError) as e:
        print(f"Warning: Could not load initial state from policy config ({e}). Using default MuJoCo initialization.")

    # Update physics state after setting qpos/qvel (important for correct sensor readings etc.)
    mujoco.mj_forward(m, d)

    # Initialize context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    target_dof_kps = kps.copy()
    target_dof_kds = kds.copy()

    # command parameters
    cmd = np.zeros(16, dtype=np.float32)
    counter = 1
    gait_frequency = cmd_params["gait_frequency"]
    feet_dist = cmd_params["feet_distance"]

    # init the gait generator
    # GG = GaitGenerator(feet_distance=feet_dist, vertical_dist=0.0, lateral_dist=0.0, steering_angle=0.0)
    GG = VisualGaitGenerator(
        feet_distance=feet_dist, vertical_dist=0.0, lateral_dist=0.0, steering_angle=0.0, stop_steps=2,
        robot_model=m, robot_data=d, cam_width=cam_width, cam_height=cam_height)
    GG.print_instruction()

    # ===========================================TELEOPERATION via KEYBOARD===========================================
    # --- Start Simulation and Viewer ---
    with mujoco.viewer.launch_passive(m, d, key_callback=GG.key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            for i in range(control_decimation):
                counter += 1
                # Step the simulation forward. The PD controller runs at the physics rate.
                tau = pd_control(
                    target_dof_pos, d.qpos[7:], target_dof_kps, np.zeros_like(kds), d.qvel[6:], target_dof_kds
                )
                d.ctrl[:] = tau

                mujoco.mj_step(m, d)

            # get rgb and depth images from head camera
            rgb_renderer.update_scene(d, camera="head_camera")
            depth_renderer.update_scene(d, camera="head_camera")
            rgb_array = rgb_renderer.render()
            depth_array = depth_renderer.render()
            depth_rgb = np.clip((depth_array - 0.2) / 1.8 * 255.0, 0, 255)
            depth_rgb = depth_rgb[:, :, None].repeat(3, axis=2).astype(np.uint8)

            # --- Prepare Observations ---
            qj = d.qpos[7:]
            dqj = d.qvel[6:]
            quat = d.qpos[3:7]  # Pelvis orientation [w, x, y, z] from free joint
            # base_ang_vel = d.sensor("angular-velocity").data.astype(np.float32) # Assuming sensor name "angular-velocity"
            base_ang_vel = d.qvel[3:6]
            projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))

            # --- Create Command Vector `cmd` ---
            gait_process = (counter * simulation_dt * gait_frequency) % 1.0

            # foot_offset, gait_info = GG.query_cmd(gp=gait_process)
            foot_offset, gait_info, rgb_array = GG.query_cmd(
                rgb_image=rgb_array, depth_image=depth_array, joint_pos=qj, gp=gait_process)
            l_offset, l_orn_offset, r_offset, r_orn_offset = foot_offset

            if GG.sample_goal:
                if GG.swing_foot_idx == 0:
                    # left foot swing
                    # get stance foot rotation
                    rot_stance = np_R.from_matrix(d.site("right_foot").xmat.reshape(3, 3))
                    stance_yaw = rot_stance.as_euler("xyz")[2]
                    rot_stance_flat = np_R.from_euler("z", stance_yaw)
                    # compute the quat
                    cmd_quat = np.array([l_orn_offset[1], l_orn_offset[2], l_orn_offset[3], l_orn_offset[0]])
                    rot_cmd = np_R.from_quat(cmd_quat)
                    # compute target rot
                    target_rot = rot_stance_flat * rot_cmd
                    # update site
                    m.site("foot_1").pos = d.site_xpos[right_foot_id] + rot_stance_flat.apply(l_offset)
                    m.site("foot_1").pos[2] = 0
                    m.site("foot_1").quat = target_rot.as_quat(scalar_first=True)
                else:
                    # right foot swing
                    # get stance foot rotation
                    rot_stance = np_R.from_matrix(d.site("left_foot").xmat.reshape(3, 3))
                    stance_yaw = rot_stance.as_euler("xyz")[2]
                    rot_stance_flat = np_R.from_euler("z", stance_yaw)
                    # compute the quat
                    cmd_quat = np.array([r_orn_offset[1], r_orn_offset[2], r_orn_offset[3], r_orn_offset[0]])
                    rot_cmd = np_R.from_quat(cmd_quat)
                    # compute the quat
                    target_rot = rot_stance_flat * rot_cmd
                    # update site
                    m.site("foot_0").pos = d.site_xpos[left_foot_id] + rot_stance_flat.apply(r_offset)
                    m.site("foot_0").pos[2] = 0
                    m.site("foot_0").quat = target_rot.as_quat(scalar_first=True)

                # move the visual targets
                target_has_passed = False
                target_pos = m.site("target_{}".format(site_idx)).pos
                robot_pos = d.qpos[:3]
                robot_orn = np_R.from_quat(d.qpos[3:7], scalar_first=True).as_matrix()
                target_pos_in_robot = robot_orn.T @ (target_pos - robot_pos)
                if target_pos_in_robot[0] < -0.05:
                    target_has_passed = True

                if target_has_passed:
                    angle += np.random.uniform(-target_angle_range, target_angle_range)
                    target_site_pos += target_dist * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0.0])
                    feet_pos = target_site_pos + cmd_params["feet_distance"] / 2 * np.array(
                        [
                            np.cos(np.deg2rad(angle + (-1) ** site_idx * 90)),
                            np.sin(np.deg2rad(angle + (-1) ** site_idx * 90)),
                            0.0,
                        ]
                    )
                    m.site("target_{}".format(site_idx)).pos = feet_pos
                    site_idx = (site_idx + 1) % 10

                mujoco.mj_fwdPosition(m, d)

            cmd = np.concatenate([l_offset, l_orn_offset, r_offset, r_orn_offset, gait_info], dtype=np.float32)

            # --- Construct Observation Vector ---
            obs_list = []
            obs_list += projected_gravity.flatten().tolist()
            obs_list += qj.flatten().tolist()
            obs_list += (base_ang_vel * 1.0).flatten().tolist()
            obs_list += (dqj * 0.1).flatten().tolist()
            obs_list += action.flatten().tolist()  # 'action' here is the previous action
            obs_list += cmd.flatten().tolist()

            obs = [0.0] * (78) + obs_list

            obs = np.array(obs, dtype=np.float32).reshape(1, -1)

            # Override Head Pitch Angle in Observation
            obs[0, 81] = 0.0  # Head Yaw Angle
            obs[0, 82] = 0.0  # Head Pitch joint position

            # --- Policy Inference ---
            emitted_action = np.asarray(policy.predict_action(obs)).flatten()

            emitted_action = np.clip(emitted_action, -1.0, 1.0)

            # Apply smoothing/filtering to the action
            action = action * 0.0 + emitted_action * 1.0

            # Deconstruct action vector into control commands
            target_dof_pos = (
                action[:num_qj] + default_angles[:num_qj]
            )  # Use num_qj here as it's the base action for positions

            # Override head joint for control
            target_dof_pos[0] = 0.0
            target_dof_pos[1] = 1.0

            # Clip target_dof_pos to joint limits
            target_dof_pos = np.clip(target_dof_pos, min_angles, max_angles)

            target_dof_kps = kps.copy()
            target_dof_kds = kds.copy()

            # Sync viewer and maintain real-time simulation speed
            viewer.sync()
            viewer.set_images([(rgb_viewport, rgb_array), (depth_viewport, depth_rgb)])

            time_until_next_step = m.opt.timestep * control_decimation - (time.time() - step_start)
            # time.sleep(0.1)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
