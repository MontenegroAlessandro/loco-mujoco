import rclpy
import time
import numpy as np
# from utils.remote_control_service import RemoteControlService
# from utils.policy import Policy
import yaml
from robot_bridge_py.robot_client import RobotClient
from enum import Enum
from loco_mujoco.algorithms import PPOJax
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from typing import Dict, Any
from scipy.spatial.transform import Rotation as np_R

class JAXPolicy:
    """A wrapper for loading and running a JAX-based PPO policy."""

    def __init__(self, policy_path: str) -> None:
        """
        Loads a trained policy from the specified path.

        Args:
            policy_path: Path to the directory containing the saved agent state.
        """
        agent_conf, agent_state = PPOJax.load_agent(policy_path)

        # Set standard deviation to a very small value for deterministic actions
        agent_state.train_state.params["log_std"] = np.ones_like(
            agent_state.train_state.params["log_std"]
        ) * -np.inf

        self.train_state = agent_state.train_state
        self._rng = jax.random.key(0)

        # Pre-compile the action sampling function for performance
        ### MODIFIED ###: Matched the function signature and call to the MuJoCo script
        self._jit_sample_action = jax.jit(self._sample_action, static_argnames=["network_apply"])
        self.network_apply = agent_conf.network.apply

    @staticmethod
    def _sample_action(network_apply, params, run_stats, rng, obs):
        """Static method to be JIT-compiled for sampling actions."""
        y, _ = network_apply(
            {'params': params, 'run_stats': run_stats},
            obs,
            mutable=["run_stats"],
        )
        pi, _ = y
        action = pi.mode()
        return jnp.atleast_2d(action)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """
        Predicts a deterministic action for a given observation.

        Args:
            obs: The observation vector from the environment.

        Returns:
            The computed action vector.
        """
        ### MODIFIED ###: Removed `self.train_state.batch_stats` from the call
        action = self._jit_sample_action(
            self.network_apply,
            self.train_state.params,
            self.train_state.run_stats,
            self._rng,
            obs,
        )
        return np.asarray(action).flatten()

class RobotController:   
    def __init__(self, node, cfg):
        self.node = node  
        self.robot = RobotClient(node=self.node, robot_type="T1", num_dof=23, control_frequency=50.0, interpolation_order=0.)
        
        # Initialize components
        # self.remoteControlService = RemoteControlService()
        # self.policy = Policy(cfg=cfg)
        policy_path = cfg["agent_path"]

        self.policy = JAXPolicy(policy_path=policy_path)
        self.policy_dt = 0.02
        self.timer = self.node.create_timer(self.policy_dt, self.step)  # 50 Hz control frequency

        self.kps: np.ndarray = np.array(cfg["lmj_kps"])

        self.kds: np.ndarray = np.array(cfg["lmj_kds"])

        self.default_angles: np.ndarray = np.array(cfg["default_angles"], dtype=np.float32)
        self.min_angles: np.ndarray = np.array(cfg["min_angles"], dtype=np.float32)
        self.max_angles: np.ndarray = np.array(cfg["max_angles"], dtype=np.float32)

        self.num_actions: int = cfg["num_actions"]
        ### ADDED ###: Load command parameters from the config file for consistency
        self.command: Dict[str, float] = cfg["command"]

        self.robot.set_default_cmd(default_pos=self.default_angles, default_kp=self.kps, default_kd=self.kds)
        
        self.agent_started = False
        self.vx_cmd = 0.0
        self.vy_cmd = 0.0
        self.vyaw_cmd = 0.0

        self.prev_action = np.zeros(self.num_actions, dtype=np.float32)

        # new for foot placement
        self.counter = 0
        self.des_dist = self.command["distance"]
        # define angles for movements
        self.fwd_angle = np.deg2rad(30)
        self.bwd_angle = np.deg2rad(120)
        self.hold_angle = np.deg2rad(90)
        # initialize the angle to be still
        self.alpha = self.hold_angle
        self.swing_foot_idx = 0
        self.gait_frequency = self.command["gait_frequency"]

        print("Please press\n\t \"LT + START\" to start control, \n\t \"LT + A\" to start inferrence, \n\t \"BACK\" to stop control, \n\t \"LB\" for ready position, \n\t \"RB\" for zero position, \n\t \"LT + BACK\" for emergency stop.")

    def step(self):
        self.check_state()
        if self.robot.control_started and self.agent_started:
            self.policy_step()  
            
    def policy_step(self):
        
        projected_gravity = self._quat_to_projected_gravity(self.robot.quat, np.array([0, 0, -1], dtype=np.float32))
        dof_pos = self.robot.q_pos
        dof_vel = self.robot.q_vel
        base_ang_vel = self.robot.angular_velocity

        # 2. CREATE COMMAND VECTOR
        # This vector is part of the observation
        cmd = np.zeros(16, dtype=np.float32)

        # Calculate gait phase based on time
        gait_process = (self.counter * self.policy_dt * self.gait_frequency) % 1.0 
        sign = np.where(self.swing_foot_idx == 0, 1, -1)
        # dist  = self.des_dist / np.sin(self.alpha) if np.sin(self.alpha) != 0 else self.des_dist
        if self.alpha != np.pi / 2:
            if self.alpha < np.pi / 2:
                dist = self.des_dist / np.sin(self.alpha)
            else:
                dist = self.des_dist / np.sin(self.alpha - np.pi / 2)
        else:
            dist = self.des_dist
        swing_pos_offset = np.array([dist * np.cos(self.alpha), dist * np.sin(self.alpha) * sign, 0], dtype=np.float32)
        stance_pos_offset = np.zeros(3, np.float32)
        swing_orn_offset = np_R.from_euler('z', np.deg2rad(0)).as_quat(scalar_first=True)
        stance_orn_offset = np_R.from_euler('z', 0).as_quat(scalar_first=True)
        
        # SET GOALS
        if self.swing_foot_idx == 0 and (gait_process >= 0.5 and gait_process < 1):
            self.swing_foot_idx = 1
        elif self.swing_foot_idx == 1 and (gait_process < 0.5 and gait_process >= 0):
            self.swing_foot_idx = 0
        # MANAGE OBS
        if self.swing_foot_idx == 0:
            l_offset = swing_pos_offset
            l_orn_offset = swing_orn_offset
            r_offset = stance_pos_offset
            r_orn_offset = stance_orn_offset
        else:
            l_offset = stance_pos_offset
            l_orn_offset = stance_orn_offset
            r_offset = swing_pos_offset
            r_orn_offset = swing_orn_offset

        # pack gait information
        gait_cos = np.cos(2 * np.pi * gait_process) 
        gait_sin = np.sin(2 * np.pi * gait_process)
        gait_info = np.array([gait_cos, gait_sin])
                    
        cmd = np.concatenate(
            [l_offset, l_orn_offset, r_offset, r_orn_offset, gait_info], dtype=np.float32
        )

        obs_list = []
        obs_list.extend(projected_gravity.flatten())
        obs_list.extend(dof_pos.flatten())
        obs_list.extend(base_ang_vel.flatten())
        obs_list.extend((dof_vel * 0.1).flatten()) # Crucial scaling factor
        obs_list.extend(self.prev_action.flatten())
        obs_list.extend(cmd.flatten())

        # Prepend zeros to match the padded observation space from loco-mujoco
        padded_obs_list = [0.] * 78 + obs_list
        obs = np.array(padded_obs_list, dtype=np.float32).reshape(1, -1)

        # obs = np.clip(obs, -50.0, 50.0) # Clip observation values

        # 4. POLICY INFERENCE
        emitted_action = self.policy.predict(obs)

        self.prev_action = emitted_action.copy()

        emitted_action = np.clip(emitted_action, -1.0, 1.0)

        # 5. PROCESS AND STORE ACTION
        # The action is stored for the next observatio

        # The policy outputs a residual to be added to the default standing pose
        dof_target_residual = emitted_action
        q_des = dof_target_residual + self.default_angles

        # Clip the final target angles to be within the robot's joint limits
        q_des = np.clip(q_des, self.min_angles, self.max_angles)
        q_des[3] = -1.2 # arms contraint
        q_des[7] = 1.2 # arms contraint

        # 6. SEND COMMAND TO ROBOT
        self.robot.send_cmd(q_target_pos=q_des, target_kp=self.kps, target_kd=self.kds)

        # 7. INCREMENT COUNTER
        self.counter += 1
    
    def convert_quat_to_rot_mat(self, quat):
        """
        Convert a quaternion to a rotation matrix.

        Parameters:
        quat (np.ndarray): A 4-element array representing the quaternion (w, x, y, z).

        Returns:
        np.ndarray: A 3x3 rotation matrix.
        """
        w, x, y, z = quat
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
        ])
        return R
    
    def _quat_to_projected_gravity(self, quat, vector):
        """
        Rotate a vector by the inverse of the given roll, pitch, and yaw angles.

        Parameters:
        roll (float): The roll angle in radians.
        pitch (float): The pitch angle in radians.
        yaw (float): The yaw angle in radians.
        vector (np.ndarray): The 3D vector to be rotated.

        Returns:
        np.ndarray: The rotated 3D vector.
        """
        rot_mat = self.convert_quat_to_rot_mat(quat)
        # R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        # R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        # R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        return rot_mat.T @ vector
        
    def check_state(self):
        self.robot.update_robot_state()

        if self.robot.joy_key is not None:
            if self.robot.joy_key.lt and self.robot.joy_key.a and self.robot.key_count == 2:  # start: LT + A
                if self.robot.control_started:
                    self.agent_started = True
                    # self.vx_cmd = 0.0
                    # self.vy_cmd = 0.0
                    # self.vyaw_cmd = 0.0
                    self.alpha = self.hold_angle
                    self.node.get_logger().info("Agent started.")
                else:
                    self.node.get_logger().warn("Please start the control first by pressing LT + START.")

            if self.agent_started:
                if self.robot.joy_key.hat_u and self.robot.key_count == 1:
                    # up key
                    # self.vx_cmd += 0.1
                    self.alpha = self.fwd_angle
                elif self.robot.joy_key.hat_d and self.robot.key_count == 1:
                    # down key
                    self.alpha = self.bwd_angle
                    # self.vx_cmd -= 0.1
                elif self.robot.joy_key.hat_l and self.robot.key_count == 1:
                    # left key
                    # self.vy_cmd += 0.1
                    self.alpha = self.hold_angle
                elif self.robot.joy_key.hat_r and self.robot.key_count == 1:
                    # self.vy_cmd -= 0.1
                    self.alpha = self.hold_angle
                elif self.robot.joy_key.rx * -1 >= 1.0:
                    # self.vyaw_cmd += 0.1
                    self.alpha = self.hold_angle
                elif self.robot.joy_key.rx * -1 <= -1.0:
                    # self.vyaw_cmd -= 0.1
                    self.alpha = self.hold_angle
                if (self.robot.joy_key.ls or self.robot.joy_key.rs) and self.robot.key_count == 1:
                    # self.vx_cmd = 0.0
                    # self.vy_cmd = 0.0
                    # self.vyaw_cmd = 0.0
                    self.alpha = self.hold_angle

                # self.vx_cmd = np.clip(self.vx_cmd, -0.5, 0.5)
                # self.vy_cmd = np.clip(self.vy_cmd, -0.5, 0.5)
                # self.vyaw_cmd = np.clip(self.vyaw_cmd, -2.0, 2.0)
                # print(f"Velocity commands - vx: {self.vx_cmd}, vy: {self.vy_cmd}, vyaw: {self.vyaw_cmd}")
            else:
                # self.vx_cmd = 0.0
                # self.vy_cmd = 0.0
                # self.vyaw_cmd = 0.0
                self.alpha = self.hold_angle

            self.robot.joy_key = None  # Reset joy_key
            self.robot.key_count = 0 # Reset true_count after processing
            self.robot.joy_axes = np.zeros(6, dtype=np.float32)

        if not self.robot.control_started:
            self.agent_started = False


if __name__ == "__main__":
    rclpy.init()
    # cfg_file = "src/humanoid_bridge/robot_bridge/example/T1/configs/T1.yaml"
    cfg_file = "config_lmj.yaml"
    with open(cfg_file, "r", encoding="utf-8") as f:
        policy_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    node = rclpy.create_node('robot_client_node')
    controller = RobotController(node, policy_cfg)
    
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()