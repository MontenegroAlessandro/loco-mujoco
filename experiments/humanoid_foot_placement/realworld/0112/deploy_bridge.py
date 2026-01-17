import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict
from scipy.spatial.transform import Rotation as np_R
import hydra
import mujoco 

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from message_filters import Subscriber, ApproximateTimeSynchronizer

import cv_bridge
import cv2

from robot_bridge_py.robot_client import RobotClient
import loco_mujoco
from loco_mujoco.algorithms import PPOJax


LMJ_PATH = loco_mujoco.__path__[0]
deploy_path = os.path.join(LMJ_PATH, "..", "experiments/humanoid_foot_placement/deploy")
sys.path.insert(0, deploy_path)
from gait_generators import GaitGenerator, VisualGaitGenerator


class JAXPolicy:
    """A wrapper for loading and running a JAX-based PPO policy."""

    def __init__(self, policy_path: str) -> None:
        # Load agent configuration and state from the policy checkpoint
        agent_conf, agent_state = PPOJax.load_agent(policy_path)

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
    def _sample_action(network_apply, params, run_stats, rng, obs):
        y, _ = network_apply({'params': params, 'run_stats': run_stats}, obs, mutable=["run_stats"])
        pi, _ = y
        a = pi.mode()
        a = jnp.atleast_2d(a)
        return a

    def predict_action(self, obs: np.ndarray) -> np.ndarray:
        action = self._jit_sample_action(
            self.network_apply,
            self.train_state.params,
            self.train_state.run_stats,
            self._rng,
            obs,
        )
        return action


class GaitGenerator1:
    def __init__(
        self,
        feet_distance: float = 0.2,
        vertical_dist: float = 0.1,
        lateral_dist: float = 0.3,
        steering_angle: float = 0.0,
    ):
        self.feet_distance = feet_distance
        self.vertical_dist = vertical_dist
        self.lateral_dist = lateral_dist
        self.steering_angle = steering_angle
        self.gaits_to_still = 0

    def query_cmd(self, mov_dir: str = "STILL", reset: bool = False, gp: float = 0.0):
        err_msg = f"[GaitGenerator: query_cmd] Mode {mov_dir} is not valid."
        assert mov_dir in ["STILL", "FWD", "BWD", "LEFT", "RIGHT", "DIAG-L", "DIAG-R"], err_msg

        if mov_dir == "STILL":
            cmd = self._gen_still_cmd(reset=reset, gp=gp)
        elif mov_dir in ["FWD", "BWD"]:
            # direction = 1 if mov_dir == "FWD" else -1
            direction = 1
            cmd = self._gen_vertical_cmd(gp=gp, direction=direction)
        elif mov_dir in ["LEFT", "RIGHT"]:
            direction = 1 if mov_dir == "LEFT" else -1
            cmd = self._gen_lateral_cmd(gp=gp, direction=direction)
        elif mov_dir in ["DIAG-L", "DIAG-R"]:
            direction = 1 if mov_dir == "DIAG-L" else -1
            cmd = self._gen_diag_cmd(gp=gp, direction=direction)

        return cmd

    def _gen_still_cmd(self, reset: bool = False, gp: float = 0.0):
        swing_foot_idx = 0 if (gp < 0.5) else 1

        zero_orn_offset = np.array([1, 0, 0, 0], dtype=np.float32)
        zero_pos_offset = np.zeros(3, dtype=np.float32)

        if self.gaits_to_still > 0:
            gait_info = np.array([np.cos(2 * np.pi * gp), np.sin(2 * np.pi * gp)])
            l_pos_offset = np.array([0, self.feet_distance, 0]) if swing_foot_idx == 0 else zero_pos_offset
            r_pos_offset = np.array([0, -self.feet_distance, 0]) if swing_foot_idx == 1 else zero_pos_offset
            l_orn_offset = zero_orn_offset
            r_orn_offset = zero_orn_offset
        else:
            gait_info = np.zeros(2, dtype=np.float32)
            l_pos_offset = np.array([0, self.feet_distance, 0])
            r_pos_offset = np.array([0, -self.feet_distance, 0])
            l_orn_offset = zero_orn_offset
            r_orn_offset = zero_orn_offset

        return l_pos_offset, l_orn_offset, r_pos_offset, r_orn_offset, gait_info

    def _gen_vertical_cmd(self, gp: float = 0.0, direction: int = 1):
        err_msg = f"[GaitGenerator: _gen_vertical_cmd] Direction {direction} is not valid."
        assert direction in [-1, 1], err_msg

        swing_foot_idx = 0 if (gp < 0.5) else 1

        zero_orn_offset = np.array([1, 0, 0, 0], dtype=np.float32)
        zero_pos_offset = np.zeros(3, dtype=np.float32)

        steering_angle = np.clip(self.steering_angle, -np.pi, np.pi) if direction == 1 else 0.0
        steering_foot_idx = 0 if (steering_angle >= 0 and steering_angle <= np.pi) else 1
        steering_orn_offset = np_R.from_euler("z", steering_angle).as_quat(scalar_first=True)

        gait_info = np.array([np.cos(2 * np.pi * gp), np.sin(2 * np.pi * gp)])
        l_pos_offset = np.array([direction * self.vertical_dist, self.feet_distance, 0]) if swing_foot_idx == 0 else zero_pos_offset
        r_pos_offset = np.array([direction * self.vertical_dist, -self.feet_distance, 0]) if swing_foot_idx == 1 else zero_pos_offset
        l_orn_offset = steering_orn_offset if steering_foot_idx == 0 else zero_orn_offset
        r_orn_offset = steering_orn_offset if steering_foot_idx == 1 else zero_orn_offset

        return l_pos_offset, l_orn_offset, r_pos_offset, r_orn_offset, gait_info

    def _gen_lateral_cmd(self, gp: float = 0.0, direction: int = 1):
        err_msg = f"[GaitGenerator: _gen_lateral_cmd] Direction {direction} is not valid."
        # assert direction in [-1, 1], err_msg
        direction = 1 if self.lateral_dist >= 0 else -1

        swing_foot_idx = 0 if (gp < 0.5) else 1

        zero_orn_offset = np.array([1, 0, 0, 0], dtype=np.float32)
        zero_pos_offset = np.zeros(3, dtype=np.float32)

        lat_dist = self.feet_distance * direction + self.lateral_dist
        max_evil_movement = - lat_dist / 2.0

        gait_info = np.array([np.cos(2 * np.pi * gp), np.sin(2 * np.pi * gp)])

        if direction == 1:  # left
            l_pos_offset = np.array([0.0, lat_dist, 0.0]) if swing_foot_idx == 0 else zero_pos_offset
            r_pos_offset = np.array([0.0, max_evil_movement, 0.0]) if swing_foot_idx == 1 else zero_pos_offset
        else:  # right
            l_pos_offset = np.array([0.0, max_evil_movement, 0.0]) if swing_foot_idx == 0 else zero_pos_offset
            r_pos_offset = np.array([0.0, lat_dist, 0.0]) if swing_foot_idx == 1 else zero_pos_offset

        l_orn_offset = zero_orn_offset
        r_orn_offset = zero_orn_offset
        return l_pos_offset, l_orn_offset, r_pos_offset, r_orn_offset, gait_info

    def _gen_diag_cmd(self, gp: float = 0.0, direction: int = 1):
        err_msg = f"[GaitGenerator: _gen_diag_cmd] Direction {direction} is not valid."
        assert direction in [-1, 1], err_msg

        swing_foot_idx = 0 if (gp < 0.5) else 1

        zero_orn_offset = np.array([1, 0, 0, 0], dtype=np.float32)
        zero_pos_offset = np.zeros(3, dtype=np.float32)

        gait_info = np.array([np.cos(2 * np.pi * gp), np.sin(2 * np.pi * gp)])
        max_evil_movement = self.lateral_dist / 2.0

        if direction == 1:  # diag-left
            l_pos_offset = np.array([self.lateral_dist, self.lateral_dist, 0.0]) if swing_foot_idx == 0 else zero_pos_offset
            r_pos_offset = np.array([-max_evil_movement, -max_evil_movement, 0.0]) if swing_foot_idx == 1 else zero_pos_offset
        else:  # diag-right
            l_pos_offset = np.array([-max_evil_movement, max_evil_movement, 0.0]) if swing_foot_idx == 0 else zero_pos_offset
            r_pos_offset = np.array([self.lateral_dist, -self.lateral_dist, 0.0]) if swing_foot_idx == 1 else zero_pos_offset

        l_orn_offset = zero_orn_offset
        r_orn_offset = zero_orn_offset
        return l_pos_offset, l_orn_offset, r_pos_offset, r_orn_offset, gait_info


class RobotController:
    def __init__(self, node, cfg):
        self.node: Node = node
        self.robot = RobotClient(node=self.node, robot_type="T1", num_dof=23, control_frequency=50.0, interpolation_order=0.)

        policy_path = cfg["agent_path"]
        self.policy = JAXPolicy(policy_path=policy_path)

        self.color_image: Image = None
        self.depth_image: Image = None
        self.cam_info: CameraInfo = None

        self.depth_sub = Subscriber(self.node, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        self.color_sub = Subscriber(self.node, Image, '/camera/camera/color/image_raw')
        self.info_sub = Subscriber(self.node, CameraInfo, '/camera/camera/color/camera_info')

        # Publish TF from base_link to camera frame
        self.base_to_cam_tf_pub = self.node.create_publisher(TFMessage, "/tf", 10)

        self.ts = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub, self.info_sub], queue_size=5, slop=0.1)
        self.ts.registerCallback(self.synchronized_callback)

        self.cv_bridge = cv_bridge.CvBridge()

        self.node.get_logger().info("Waiting for camera messages...")
        while self.color_image is None or self.cam_info is None or self.depth_image is None:
            rclpy.spin_once(self.node)

        self.policy_dt = 0.02

        self.kps: np.ndarray = np.array(cfg["lmj_kps"])
        self.kds: np.ndarray = np.array(cfg["lmj_kds"])

        self.default_angles: np.ndarray = np.array(cfg["default_angles"], dtype=np.float32)
        self.min_angles: np.ndarray = np.array(cfg["min_angles"], dtype=np.float32)
        self.max_angles: np.ndarray = np.array(cfg["max_angles"], dtype=np.float32)

        self.num_actions: int = cfg["num_actions"]
        self.command: Dict[str, float] = cfg["command"]

        self.robot.set_default_cmd(default_pos=self.default_angles, default_kp=self.kps, default_kd=self.kds)

        self.agent_started = False
        self.prev_action = np.zeros(self.num_actions, dtype=np.float32)

        # Gait / foot placement
        self.counter = 0
        self.gait_frequency = self.command["gait_frequency"]

        # self.GG = GaitGenerator(feet_distance=self.command.feet_distance, stop_steps=self.command.stop_steps)

        spec = mujoco.MjSpec.from_file(cfg.xml_path)
        self._model = spec.compile()
        self._data = mujoco.MjData(self._model)

        self.last_img_time = self.cam_info.header.stamp.sec + self.cam_info.header.stamp.nanosec * 1e-9

        self.GG = VisualGaitGenerator(robot_model=self._model, robot_data=self._data, 
                                             cam_width=self.cam_info.width, cam_height=self.cam_info.height,
                                             feet_distance=self.command.feet_distance, stop_steps=self.command.stop_steps,
                                             debug_vis=True)

        self.swing_foot_idx = 0 if ((self.counter * self.policy_dt * self.gait_frequency) % 1.0 < 0.5) else 1

        print(
            "Controller bindings:\n"
            "  1) LT + RT + A : init (agent_started) in STILL mode\n"
            "  2) A           : toggle stand/move (STILL <-> FWD)\n"
            "  3) UP          : increase step length, set FWD\n"
            "  4) DOWN        : increase step length, set BWD\n"
            "  5) RIGHT       : strafe RIGHT\n"
            "  6) LEFT        : strafe LEFT\n"
            "  7) LT + RIGHT  : increase steering\n"
            "  8) LT + LEFT   : decrease steering\n"
            "  9) B   : reset\n"
        )

        self.timer = self.node.create_timer(self.policy_dt, self.step)

    # ---------------- Camera Callbacks ----------------
    def synchronized_callback(self, color_msg, depth_msg, info_msg):
        self.depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        self.depth_image = self.depth_image.astype(np.float32) / 1000.0  # Convert from mm to meters
        self.color_image = self.cv_bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
        self.cam_info = info_msg

    # ---------------- Teleop actions (controller -> gait params) ----------------
    def _teleop_toggle_move(self):
        self.GG.teleop["move_enabled"] = not self.GG.teleop["move_enabled"]
        if not self.GG.teleop["move_enabled"]:
            self.GG.teleop["mov_dir"] = "STILL"

        if self.GG.teleop["mov_dir"] == "STILL":
            self.GG.gaits_to_still = 2

        self.node.get_logger().info(
            f"[teleop] move_enabled={self.GG.teleop['move_enabled']} mov_dir={self.GG.teleop['mov_dir']}"
        )

    def _teleop_w(self):
        self.GG.vertical_dist = float(
            np.clip(self.GG.vertical_dist + self.GG.teleop["vert_step"], self.GG.teleop["vert_min"], self.GG.teleop["vert_max"])
            )
        # self.vert_dist = self.GG.vertical_dist

        self.GG.teleop["mov_dir"] = "FWD"
        self.node.get_logger().info(f"[teleop] vertical_dist={self.GG.vertical_dist:.2f} (FWD)")

    def _teleop_s(self):
        # mirror your sim2sim: DOWN increases step length, sets BWD
        self.GG.vertical_dist = float(
            np.clip(self.GG.vertical_dist - self.GG.teleop["vert_step"], self.GG.teleop["vert_min"], self.GG.teleop["vert_max"])
            )
        # self.vert_dist = self.GG.vertical_dist

        self.GG.teleop["mov_dir"] = "FWD"
        self.node.get_logger().info(f"[teleop] vertical_dist={self.GG.vertical_dist:.2f} (BWD)")

    def _teleop_left(self):
        self.GG.lateral_dist = float(
            np.clip(self.GG.lateral_dist + self.GG.teleop["lat_step"], self.GG.teleop["lat_min"], self.GG.teleop["lat_max"])
            )
        # if self.GG.teleop["move_enabled"]:
        #     self.GG.teleop["mov_dir"] = "LEFT"
        #     self.node.get_logger().info("[teleop] mov_dir=LEFT")
        self.GG.teleop["mov_dir"] = "LEFT"
        self.node.get_logger().info(f"[teleop] lat_dist={self.GG.lateral_dist:.2f} (LEFT)")

    def _teleop_right(self):
        self.GG.lateral_dist = float(
            np.clip(self.GG.lateral_dist - self.GG.teleop["lat_step"], self.GG.teleop["lat_min"], self.GG.teleop["lat_max"])
            )
        # if self.GG.teleop["move_enabled"]:
        #     self.GG.teleop["mov_dir"] = "RIGHT"
        #     self.node.get_logger().info("[teleop] mov_dir=RIGHT")
        self.GG.teleop["mov_dir"] = "RIGHT"
        self.node.get_logger().info(f"[teleop] lat_dist={self.GG.lateral_dist:.2f} (RIGHT)")

    def _teleop_q(self):
        self.GG.steering_angle = float(
            np.clip(self.GG.steering_angle + self.GG.teleop["yaw_step"], self.GG.teleop["yaw_min"], self.GG.teleop["yaw_max"])
            )
        self.steering_angle = self.GG.steering_angle
        self.node.get_logger().info(f"[teleop] steering_angle(deg)={np.rad2deg(self.GG.steering_angle):.2f}")

    def _teleop_e(self):
        self.GG.steering_angle = float(
            np.clip(self.GG.steering_angle - self.GG.teleop["yaw_step"], self.GG.teleop["yaw_min"], self.GG.teleop["yaw_max"])
        )
        self.steering_angle = self.GG.steering_angle
        self.node.get_logger().info(f"[teleop] steering_angle(deg)={np.rad2deg(self.GG.steering_angle):.2f}")

    def _teleop_reset(self):
        self.GG.steering_angle = 0.0
        self.GG.vertical_dist = 0.0
        self.GG.lateral_dist = 0.0
        self.GG.teleop["mov_dir"] = "FWD" if self.GG.teleop["move_enabled"] else "STILL"
        self.node.get_logger().info("[teleop] RESET gait parameters")

    # ---------------- Main loop ----------------
    def step(self):
        self.check_state()

        # targets, rgb_image, mask_image = self.GG.detect_foot_target(self.color_image, self.depth_image)
        # joint_pos = self.robot.q_pos
        # l_rel_pos, l_rel_xmat, r_rel_pos, r_rel_xmat = self.GG._get_foot_to_cam(joint_pos)
        # target_in_l_foot = (l_rel_xmat @ targets.T).T + l_rel_pos

        # cv2.imshow("rgb_image", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        # cv2.imshow("mask_image", cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR))
        # print(f"Detected targets in left foot frame: {target_in_l_foot}")
        # cv2.waitKey(0)

        if self.robot.control_started and self.agent_started:
            self.policy_step()

    def policy_step(self):
        projected_gravity = self._quat_to_projected_gravity(
            self.robot.quat, np.array([0, 0, -1], dtype=np.float32)
        )
        dof_pos = self.robot.q_pos
        dof_vel = self.robot.q_vel
        base_ang_vel = self.robot.angular_velocity

        # gait phase
        gait_process = (self.counter * self.policy_dt * self.gait_frequency) % 1.0

        # build command vector
        new_img_time = self.cam_info.header.stamp.sec + self.cam_info.header.stamp.nanosec * 1e-9
        if new_img_time > self.last_img_time:
            image_updated = True
        else:
            image_updated = False
            # self.GG_visual.update_camera_info(width=self.cam_info.width, height=self.cam_info.height,
            #                                  focal_x=self.cam_info.k[0], focal_y=self.cam_info.k[4],
            #                                  pp_x=self.cam_info.k[2], pp_y=self.cam_info.k[5])

        foot_offset, gait_info, _ = self.GG.query_cmd(self.color_image, self.depth_image, dof_pos, gp=gait_process, image_updated=image_updated)

        if self.GG.sample_goal and image_updated:
            self.last_img_time = new_img_time

        # foot_offset, gait_info = self.GG.query_cmd(gp=gait_process)
        l_offset, l_orn_offset, r_offset, r_orn_offset = foot_offset

        cmd = np.concatenate(
            [l_offset, l_orn_offset, r_offset, r_orn_offset, gait_info], dtype=np.float32
        )

        obs_list = []
        obs_list.extend(projected_gravity.flatten())
        obs_list.extend(dof_pos.flatten())
        obs_list.extend(base_ang_vel.flatten())
        obs_list.extend((dof_vel * 0.1).flatten())
        obs_list.extend(self.prev_action.flatten())
        obs_list.extend(cmd.flatten())

        padded_obs_list = [0.] * 78 + obs_list
        obs = np.array(padded_obs_list, dtype=np.float32).reshape(1, -1)
        obs[0, 81] = 0.0
        obs[0, 82] = 0.0

        emitted_action = np.asarray(self.policy.predict_action(obs)).flatten()
        self.prev_action = emitted_action.copy()
        emitted_action = np.clip(emitted_action, -1.0, 1.0)

        q_des = emitted_action + self.default_angles
        q_des[1] = 1.0
        q_des = np.clip(q_des, self.min_angles, self.max_angles)

        self.robot.send_cmd(q_target_pos=q_des, target_kp=self.kps, target_kd=self.kds)
        self.counter += 1

    # ---------------- Utility ----------------
    def convert_quat_to_rot_mat(self, quat):
        w, x, y, z = quat
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
        ])
        return R

    def _quat_to_projected_gravity(self, quat, vector):
        rot_mat = self.convert_quat_to_rot_mat(quat)
        return rot_mat.T @ vector

    # ---------------- Controller state machine ----------------
    def check_state(self):
        self.robot.update_robot_state()

        self._publish_base_to_camera_tf()

        if self.robot.joy_key is not None:
            # 1) LT + RT + A : initialization in still mode
            if (self.robot.joy_key.lt and self.robot.joy_key.rt and self.robot.joy_key.a and self.robot.key_count == 3):
                if self.robot.control_started:
                    self.agent_started = True

                    # reset teleop to STILL
                    self.GG.teleop["move_enabled"] = False
                    self.GG.teleop["mov_dir"] = "STILL"
                    self.GG.teleop["last_mov_dir"] = "STILL"
                    self.GG.gaits_to_still = 0

                    # optional: reset counters for clean phase
                    self.counter = 0
                    self.swing_foot_idx = 0

                    # reset
                    self.GG.vertical_dist = 0.0
                    self.GG.steering_angle = 0.0
                    self.vert_dist = self.GG.vertical_dist
                    self.steering_angle = self.GG.steering_angle
                    self.GG.lateral_dist = 0.0
                    self.lat_dist = self.GG.lateral_dist

                    self.node.get_logger().info("Initialized agent in STILL mode (LT+RT+A).")
                else:
                    self.node.get_logger().warn("Please start the control first (robot control not started).")

            # Teleop commands only if agent started
            if self.agent_started:
                # 2) A : toggle stand/move
                if self.robot.joy_key.a and self.robot.key_count == 1:
                    self._teleop_toggle_move()

                # 3) UP
                elif self.robot.joy_key.hat_u and self.robot.key_count == 1:
                    self._teleop_w()

                # 4) DOWN
                elif self.robot.joy_key.hat_d and self.robot.key_count == 1:
                    self._teleop_s()

                # 5) RIGHT
                elif self.robot.joy_key.hat_r and self.robot.key_count == 1:
                    # self.counter = int(0.5 / (self.policy_dt * self.gait_frequency))
                    # self.GG.lateral_dist = self.lat_dist
                    self._teleop_right()

                # 6) LEFT
                elif self.robot.joy_key.hat_l and self.robot.key_count == 1:
                    # self.counter = 0
                    # self.GG.lateral_dist = self.lat_dist
                    self._teleop_left()

                # 7) RB
                elif self.robot.joy_key.lt and self.robot.joy_key.hat_r and self.robot.key_count == 2:
                    self._teleop_e()

                # 8) LB
                elif self.robot.joy_key.lt and self.robot.joy_key.hat_l and self.robot.key_count == 2:
                    self._teleop_q()

                elif self.robot.joy_key.b and self.robot.key_count == 1:
                    self.GG.vertical_dist = 0.0
                    self.GG.steering_angle = 0.0
                    self.GG.lateral_dist = 0.0
                    self.vert_dist = self.GG.vertical_dist
                    self.steering_angle = self.GG.steering_angle
                    self.lat_dist = self.GG.lateral_dist
                    print("RESET")

            # Reset inputs after processing
            self.robot.joy_key = None
            self.robot.key_count = 0
            self.robot.joy_axes = np.zeros(6, dtype=np.float32)

        if not self.robot.control_started:
            self.agent_started = False

    def _publish_base_to_camera_tf(self):
        tf = TFMessage()
        stamp = self.node.get_clock().now().to_msg()

        self._data.qpos[0:3] = np.array([0.0, 0.0, 1.0])
        self._data.qpos[3:7] = self.robot.quat
        self._data.qpos[7:30] = self.robot.q_pos
        mujoco.mj_fwdKinematics(self._model, self._data)

        l_feet_id = self._model.site("left_foot").id
        r_feet_id = self._model.site("right_foot").id
        l_hand_id = self._model.site("left_hand").id
        r_hand_id = self._model.site("right_hand").id
        base_id = self._model.body("Trunk").id
        waist_id = self._model.body("Waist").id
        cam_id = self._model.cam("head_camera").id

        base_pos = self._data.xpos[base_id]
        base_mat = self._data.xmat[base_id].reshape(3, 3)

        def prepare_target_tf(base_pos, base_mat, target_pos, target_mat, child_frame_id, stamp):
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = "Trunk"
            transform.child_frame_id = child_frame_id
            target_in_base_pos = base_mat.T @ (target_pos - base_pos)
            target_in_base_mat = base_mat.T @ target_mat
            transform.transform.translation.x = target_in_base_pos[0]
            transform.transform.translation.y = target_in_base_pos[1]
            transform.transform.translation.z = target_in_base_pos[2]
            rot = np_R.from_matrix(target_in_base_mat)
            quat = rot.as_quat()  # x, y, z, w
            transform.transform.rotation.x = quat[0]
            transform.transform.rotation.y = quat[1]
            transform.transform.rotation.z = quat[2]
            transform.transform.rotation.w = quat[3]

            return transform

        cam_pos = self._data.cam_xpos[cam_id]
        cam_mat = self._data.cam_xmat[cam_id].reshape(3, 3)
        transform = prepare_target_tf(base_pos, base_mat, cam_pos, cam_mat, "head_camera", stamp)
        tf.transforms.append(transform)

        l_foot_pos = self._data.site_xpos[l_feet_id]
        l_foot_mat = self._data.site_xmat[l_feet_id].reshape(3, 3)
        transform = prepare_target_tf(base_pos, base_mat, l_foot_pos, l_foot_mat, "left_foot_site", stamp)
        tf.transforms.append(transform)

        r_foot_pos = self._data.site_xpos[r_feet_id]
        r_foot_mat = self._data.site_xmat[r_feet_id].reshape(3, 3)
        transform = prepare_target_tf(base_pos, base_mat, r_foot_pos, r_foot_mat, "right_foot_site", stamp)
        tf.transforms.append(transform)

        l_hand_pos = self._data.site_xpos[l_hand_id]
        l_hand_mat = self._data.site_xmat[l_hand_id].reshape(3, 3)
        transform = prepare_target_tf(base_pos, base_mat, l_hand_pos, l_hand_mat, "left_hand_site", stamp)
        tf.transforms.append(transform)

        r_hand_pos = self._data.site_xpos[r_hand_id]
        r_hand_mat = self._data.site_xmat[r_hand_id].reshape(3, 3)
        transform = prepare_target_tf(base_pos, base_mat, r_hand_pos, r_hand_mat, "right_hand_site", stamp)
        tf.transforms.append(transform)

        waist_pos = self._data.xpos[waist_id]
        waist_mat = self._data.xmat[waist_id].reshape(3, 3)
        transform = prepare_target_tf(base_pos, base_mat, waist_pos, waist_mat, "Waist", stamp)
        tf.transforms.append(transform)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = "head_camera"
        transform.child_frame_id = "camera_link"

        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0
        rot = np_R.from_euler("xyz", [0.0, np.pi / 2.0, np.pi / 2.0])
        quat = rot.as_quat()  # x, y, z, w
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        tf.transforms.append(transform)

        self.base_to_cam_tf_pub.publish(tf)


if __name__ == "__main__":
    rclpy.init()
    cfg_file = "config_sim2sim.yaml"

    print("wrf")
    
    hydra.initialize(config_path="./")
    cfg = hydra.compose(config_name=cfg_file)

    node = rclpy.create_node('robot_client_node')
    controller = RobotController(node, cfg)

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()
