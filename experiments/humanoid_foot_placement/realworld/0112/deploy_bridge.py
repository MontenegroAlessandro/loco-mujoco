import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict
from scipy.spatial.transform import Rotation as np_R
import hydra
import mujoco

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
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
from gait_generators_vis import VisualGaitGenerator


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
        y, _ = network_apply({"params": params, "run_stats": run_stats}, obs, mutable=["run_stats"])
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


class RobotController:
    def __init__(self, node, cfg):
        self.node: Node = node
        self.robot = RobotClient(
            node=self.node, robot_type="T1", num_dof=23, control_frequency=50.0, interpolation_order=0.0
        )

        policy_path = cfg["agent_path"]
        self.policy = JAXPolicy(policy_path=policy_path)

        self.color_image: Image = None
        self.depth_image: Image = None
        self.cam_info: CameraInfo = None

        qos_profile = QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT, durability=rclpy.qos.DurabilityPolicy.VOLATILE, depth=1
        )

        self.depth_sub = Subscriber(
            self.node, Image, "/camera/camera/aligned_depth_to_color/image_raw", qos_profile=qos_profile
        )
        self.color_sub = Subscriber(self.node, Image, "/camera/camera/color/image_raw", qos_profile=qos_profile)
        self.info_sub = Subscriber(
            self.node, CameraInfo, "/camera/camera/aligned_depth_to_color/camera_info", qos_profile=qos_profile
        )

        self.color_local_pub = self.node.create_publisher(Image, "/deploy_bridge/color/image_new", 1)
        self.depth_local_pub = self.node.create_publisher(Image, "/deploy_bridge/depth/image_new", 1)
        self.depth_caminfo_local_pub = self.node.create_publisher(CameraInfo, "/deploy_bridge/depth/camera_info", 1)
        self.pub_img_flag = False

        # Publish TF from base_link to camera frame
        self.base_to_cam_tf_pub = self.node.create_publisher(TFMessage, "/tf", 10)

        self.ts = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub, self.info_sub], queue_size=5, slop=0.05)
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

        self.num_qj = len(self.default_angles)

        self.cmd_params: Dict[str, float] = cfg["command"]
        self.num_actions = cfg["num_actions"]
        self.num_actions += 1 if self.cmd_params.is_gp_adaptive else 0

        self.robot.set_default_cmd(default_pos=self.default_angles, default_kp=self.kps, default_kd=self.kds)

        self.agent_started = False
        self.prev_action = np.zeros(self.num_actions, dtype=np.float32)

        # self.GG = GaitGenerator(feet_distance=self.command.feet_distance, stop_steps=self.command.stop_steps)

        spec = mujoco.MjSpec.from_file(cfg.xml_path)
        self._model = spec.compile()
        self._data = mujoco.MjData(self._model)

        self.last_img_time = self.cam_info.header.stamp.sec + self.cam_info.header.stamp.nanosec * 1e-9

        self.GG = VisualGaitGenerator(
            robot_model=self._model,
            robot_data=self._data,
            gait_frequency=self.cmd_params.gait_frequency,
            policy_dt=self.policy_dt,
            cam_width=self.cam_info.width,
            cam_height=self.cam_info.height,
            feet_distance=self.cmd_params.feet_distance,
            stop_steps=self.cmd_params.stop_steps,
            img_delay_steps=self.cmd_params.img_delay_steps,
            is_gp_adaptive=self.cmd_params.is_gp_adaptive,
            min_gp_delta=self.cmd_params.min_gp_delta,
            max_gp_delta=self.cmd_params.max_gp_delta,
            max_gp_pause_steps=self.cmd_params.max_pause_steps,
            clahe_enhance=self.cmd_params.clahe_enhance,
            debug_vis=True,
        )

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
        depth_array = np.frombuffer(depth_msg.data, dtype=np.uint16)
        self.depth_image = depth_array.reshape((depth_msg.height, depth_msg.width))
        self.depth_image = self.depth_image.astype(np.float32) / 1000.0  # Convert from mm to meters
        # TODO
        self.depth_image = np.maximum(self.depth_image - 0.04, 0.0)

        color_array = np.frombuffer(color_msg.data, dtype=np.uint8)
        self.color_image = color_array.reshape((color_msg.height, color_msg.width, 3))
        # Match the 'rgb8' requirement
        # If the camera sends BGR (standard for ROS), swap it to RGB
        if color_msg.encoding == "bgr8":
            self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

        self.img_q_pos = self.robot.q_pos.copy()

        # self.color_image = self.cv_bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
        self.cam_info = info_msg
        self.pub_img_flag = True

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
            np.clip(
                self.GG.vertical_dist + self.GG.teleop["vert_step"],
                self.GG.teleop["vert_min"],
                self.GG.teleop["vert_max"],
            )
        )
        # self.vert_dist = self.GG.vertical_dist

        self.GG.teleop["mov_dir"] = "FWD"
        self.node.get_logger().info(f"[teleop] vertical_dist={self.GG.vertical_dist:.2f} (FWD)")

    def _teleop_s(self):
        # mirror your sim2sim: DOWN increases step length, sets BWD
        self.GG.vertical_dist = float(
            np.clip(
                self.GG.vertical_dist - self.GG.teleop["vert_step"],
                self.GG.teleop["vert_min"],
                self.GG.teleop["vert_max"],
            )
        )
        # self.vert_dist = self.GG.vertical_dist

        self.GG.teleop["mov_dir"] = "FWD"
        self.node.get_logger().info(f"[teleop] vertical_dist={self.GG.vertical_dist:.2f} (BWD)")

    def _teleop_left(self):
        self.GG.lateral_dist = float(
            np.clip(
                self.GG.lateral_dist + self.GG.teleop["lat_step"], self.GG.teleop["lat_min"], self.GG.teleop["lat_max"]
            )
        )
        # if self.GG.teleop["move_enabled"]:
        #     self.GG.teleop["mov_dir"] = "LEFT"
        #     self.node.get_logger().info("[teleop] mov_dir=LEFT")
        self.GG.teleop["mov_dir"] = "LEFT"
        self.node.get_logger().info(f"[teleop] lat_dist={self.GG.lateral_dist:.2f} (LEFT)")

    def _teleop_right(self):
        self.GG.lateral_dist = float(
            np.clip(
                self.GG.lateral_dist - self.GG.teleop["lat_step"], self.GG.teleop["lat_min"], self.GG.teleop["lat_max"]
            )
        )
        # if self.GG.teleop["move_enabled"]:
        #     self.GG.teleop["mov_dir"] = "RIGHT"
        #     self.node.get_logger().info("[teleop] mov_dir=RIGHT")
        self.GG.teleop["mov_dir"] = "RIGHT"
        self.node.get_logger().info(f"[teleop] lat_dist={self.GG.lateral_dist:.2f} (RIGHT)")

    def _teleop_q(self):
        self.GG.steering_angle = float(
            np.clip(
                self.GG.steering_angle + self.GG.teleop["yaw_step"],
                self.GG.teleop["yaw_min"],
                self.GG.teleop["yaw_max"],
            )
        )
        self.steering_angle = self.GG.steering_angle
        self.node.get_logger().info(f"[teleop] steering_angle(deg)={np.rad2deg(self.GG.steering_angle):.2f}")

    def _teleop_e(self):
        self.GG.steering_angle = float(
            np.clip(
                self.GG.steering_angle - self.GG.teleop["yaw_step"],
                self.GG.teleop["yaw_min"],
                self.GG.teleop["yaw_max"],
            )
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
        # target_in_r_foot = (r_rel_xmat @ targets.T).T + r_rel_pos

        # cv2.imshow("rgb_image", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        # cv2.imshow("mask_image", cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(1)

        # TODO
        # dof_pos = self.robot.q_pos.copy()
        # foot_offset, gait_info, rgb_image, depth_image, targets = self.GG.query_cmd(
        #     self.color_image.copy(), self.depth_image.copy(), dof_pos, image_updated=True
        # )
        # if len(targets) > 0:
        #     print(np.linalg.norm(targets, axis=-1, keepdims=True))
        #     self.publish_cam_target_tf(targets)

        if self.robot.control_started and self.agent_started:
            self.policy_step()

    def policy_step(self):
        projected_gravity = self._quat_to_projected_gravity(self.robot.quat, np.array([0, 0, -1], dtype=np.float32))
        dof_pos = self.robot.q_pos
        dof_vel = self.robot.q_vel
        base_ang_vel = self.robot.angular_velocity

        # build command vector
        new_img_time = self.cam_info.header.stamp.sec + self.cam_info.header.stamp.nanosec * 1e-9
        if new_img_time > self.last_img_time:
            image_updated = True
        else:
            image_updated = False

        foot_offset, gait_info, rgb_image, depth_image, targets = self.GG.query_cmd(
            self.color_image.copy(), self.depth_image.copy(), self.img_q_pos, image_updated=image_updated
        )

        if self.GG.sample_goal and self.GG.remaining_delay_steps == 0 and image_updated:
            self.last_img_time = new_img_time

        if self.GG.sample_goal:
            self.publish_cam_target_tf(targets)
            self.publish_foot_target_tf(foot_offset, self.GG.swing_foot_idx)

        # foot_offset, gait_info = self.GG.query_cmd(gp=gait_process)
        l_offset, l_orn_offset, r_offset, r_orn_offset = foot_offset

        # # TODO remove
        # l_offset = np.array([0, self.GG.feet_distance, 0.0])
        # l_orn_offset = np.array([1, 0, 0, 0])
        # r_offset = np.array([0, -self.GG.feet_distance, 0.0])
        # r_orn_offset = np.array([1, 0, 0, 0])
        # gait_info = np.array([0.0, 0.0])

        cmd = np.concatenate([l_offset, l_orn_offset, r_offset, r_orn_offset, gait_info], dtype=np.float32)

        obs_list = []
        obs_list.extend(projected_gravity.flatten())
        obs_list.extend(dof_pos.flatten())
        obs_list.extend(base_ang_vel.flatten())
        obs_list.extend((dof_vel * 0.1).flatten())
        obs_list.extend(self.prev_action.flatten())
        obs_list.extend(cmd.flatten())

        critic_n_obs = 78 if not self.GG.is_gp_adaptive else 79
        padded_obs_list = [0.0] * critic_n_obs + obs_list
        obs = np.array(padded_obs_list, dtype=np.float32).reshape(1, -1)
        obs[0, critic_n_obs + 3] = 0.0  # Head Yaw Angle
        obs[0, critic_n_obs + 4] = 0.0  # Head Pitch joint position

        emitted_action = np.asarray(self.policy.predict_action(obs)).flatten()
        if self.GG.is_gp_adaptive:
            self.GG.set_gp_offset(emitted_action[-1])

        self.prev_action = emitted_action.copy()
        clipped_action = np.clip(emitted_action, -1.0, 1.0)

        q_des = clipped_action[:self.num_qj] + self.default_angles[:self.num_qj]
        q_des[0] = 0.0
        q_des[1] = 1.0
        q_des = np.clip(q_des, self.min_angles, self.max_angles)

        assert np.isfinite(q_des).all(), "Non-infinite values in q_des!"
        assert np.isfinite(self.kps).all(), "Non-infinite values in kps!"
        assert np.isfinite(self.kds).all(), "Non-infinite values in kds!"

        self.robot.send_cmd(q_target_pos=q_des, target_kp=self.kps, target_kd=self.kds)

    # ---------------- Utility ----------------
    def convert_quat_to_rot_mat(self, quat):
        w, x, y, z = quat
        R = np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
            ]
        )
        return R

    def _quat_to_projected_gravity(self, quat, vector):
        rot_mat = self.convert_quat_to_rot_mat(quat)
        return rot_mat.T @ vector

    # ---------------- Controller state machine ----------------
    def check_state(self):
        self.robot.update_robot_state()

        self._publish_base_to_camera_tf()
        self.publish_images()

        self.GG.update_camera_info(cam_info=self.cam_info)

        self.agent_started = self.agent_started and self.robot.control_started

        if self.robot.joy_key is not None:
            # 1) LT + RT + A : initialization in still mode
            if self.robot.joy_key.lt and self.robot.joy_key.rt and self.robot.joy_key.a and self.robot.key_count == 3:
                if self.robot.control_started:
                    self.agent_started = True

                    # reset teleop to STILL
                    self.GG.teleop["move_enabled"] = False
                    self.GG.teleop["mov_dir"] = "STILL"
                    self.GG.teleop["last_mov_dir"] = "STILL"
                    self.GG.gaits_to_still = 0

                    # optional: reset gait process for clean phase
                    self.GG.gait_process = 0.0
                    self.GG.swing_foot_idx = 0

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
            elif self.robot.joy_key.lt and self.robot.joy_key.back and self.robot.key_count == 2:
                self.robot.stop_control()
                self.node.get_logger().info("Stopped robot control.")

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
                    self._teleop_right()

                # 6) LEFT
                elif self.robot.joy_key.hat_l and self.robot.key_count == 1:
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

                elif (
                    self.robot.joy_key.lt
                    and self.robot.joy_key.rt
                    and self.robot.joy_key.x
                    and self.robot.key_count == 3
                ):
                    self.agent_started = False
                    self.robot.init_control()
                    self.GG.gait_process = 0.0
                    print("STOPPED")

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
        transform.child_frame_id = "camera_color_optical_frame"

        transform.transform.translation.x = 0.0115
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0
        rot = np_R.from_euler("yzx", [np.pi, np.pi, 0.06])
        quat = rot.as_quat()  # x, y, z, w
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        tf.transforms.append(transform)

        self.base_to_cam_tf_pub.publish(tf)

    def publish_cam_target_tf(self, targets):
        tf = TFMessage()
        stamp = self.node.get_clock().now().to_msg()

        for i, target in enumerate(targets):
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = "camera_color_optical_frame"
            transform.child_frame_id = f"foot_target_{i}"
            transform.transform.translation.x = float(target[0])
            transform.transform.translation.y = -float(target[1])
            transform.transform.translation.z = -float(target[2])
            transform.transform.rotation.w = 1.0
            transform.transform.rotation.x = 0.0
            transform.transform.rotation.y = 0.0
            transform.transform.rotation.z = 0.0

            tf.transforms.append(transform)
        self.base_to_cam_tf_pub.publish(tf)

    def publish_foot_target_tf(self, foot_offset, swing_foot_idx):
        tf = TFMessage()
        stamp = self.node.get_clock().now().to_msg()

        l_offset, l_orn_offset, r_offset, r_orn_offset = foot_offset

        if swing_foot_idx == 0:
            offset = l_offset
            orn_offset = l_orn_offset
            parent_frame_id = "right_foot_site"
        else:
            offset = r_offset
            orn_offset = r_orn_offset
            parent_frame_id = "left_foot_site"

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = parent_frame_id
        transform.child_frame_id = "foot_target"
        transform.transform.translation.x = float(offset[0])
        transform.transform.translation.y = float(offset[1])
        transform.transform.translation.z = float(offset[2])
        transform.transform.rotation.w = float(orn_offset[0])
        transform.transform.rotation.x = float(orn_offset[1])
        transform.transform.rotation.y = float(orn_offset[2])
        transform.transform.rotation.z = float(orn_offset[3])

        tf.transforms.append(transform)
        self.base_to_cam_tf_pub.publish(tf)

    def publish_images(self):
        if self.pub_img_flag:
            color_msg = self.cv_bridge.cv2_to_imgmsg(self.color_image, encoding="rgb8")
            depth_msg = self.cv_bridge.cv2_to_imgmsg(
                (self.depth_image * 1000.0).astype(np.uint16), encoding="16UC1"
            )  # Convert back to mm
            color_msg.header = self.cam_info.header
            depth_msg.header = self.cam_info.header
            self.color_local_pub.publish(color_msg)
            self.depth_local_pub.publish(depth_msg)
            self.depth_caminfo_local_pub.publish(self.cam_info)
            self.pub_img_flag = False


if __name__ == "__main__":
    rclpy.init()
    cfg_file = "config_realworld.yaml"

    hydra.initialize(config_path="./")
    cfg = hydra.compose(config_name=cfg_file)

    node = rclpy.create_node("robot_client_node")
    controller = RobotController(node, cfg)

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()
