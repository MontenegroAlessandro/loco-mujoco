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

class GaitGenerator:
    def __init__(
        self, 
        feet_distance: float = 0.2,
        vertical_dist: float = 0.1,
        lateral_dist: float = 0.3,
        steering_angle: float = 0.0,
    ):
        # map the parameters
        self.feet_distance = feet_distance
        self.vertical_dist = vertical_dist
        self.lateral_dist = lateral_dist
        self.steering_angle = steering_angle
    
    def query_cmd(self, mov_dir: str = "STILL", reset: bool = False, gp: float = 0.0):
        err_msg = f"[GaitGenerator: query_cmd] Mode {mov_dir} is not valid."
        assert mov_dir in ["STILL", "FWD", "BWD", "LEFT", "RIGHT", "DIAG-L", "DIAG-R"], err_msg
        
        if mov_dir == "STILL":
            cmd = self._gen_still_cmd(reset=reset, gp=gp)
        elif mov_dir in ["FWD", "BWD"]:
            direction = 1 if mov_dir == "FWD" else -1
            cmd = self._gen_vertical_cmd(gp=gp, direction=direction)
        elif mov_dir in ["LEFT", "RIGHT"]:
            direction = 1 if mov_dir == "LEFT" else -1
            cmd = self._gen_lateral_cmd(gp=gp, direction=direction)
        elif mov_dir in ["DIAG-L", "DIAG-R"]:
            direction = 1 if mov_dir == "DIAG-L" else -1
            cmd = self._gen_diag_cmd(gp=gp, direction=direction)
        
        return cmd
    
    def _gen_still_cmd(self, reset: bool = False, gp: float = 0.0):
        # get the swing foot index
        swing_foot_idx = 0 if (gp < 0.5) else 1
        
        # no changing targets
        zero_orn_offset = np.array([1, 0, 0, 0], dtype=np.float32)
        zero_pos_offset = np.zeros(3, dtype=np.float32)
        
        if reset:
            # gait info gen
            gait_info = np.array([np.cos(2 * np.pi * gp), np.sin(2 * np.pi * gp)])
            # pos gen
            l_pos_offset = np.array([0, self.feet_distance, 0]) if swing_foot_idx == 0 else zero_pos_offset
            r_pos_offset = np.array([0, -self.feet_distance, 0]) if swing_foot_idx == 1 else zero_pos_offset
            # orn gen
            l_orn_offset = zero_orn_offset
            r_orn_offset = zero_orn_offset
        else:
            # gait info gen
            gait_info = np.zeros(2, dtype=np.float32)
            # pos gen
            l_pos_offset = np.array([0, self.feet_distance, 0])
            r_pos_offset = np.array([0, -self.feet_distance, 0])
            # orn gen
            l_orn_offset = zero_orn_offset
            r_orn_offset = zero_orn_offset
        
        return l_pos_offset, l_orn_offset, r_pos_offset, r_orn_offset, gait_info
    
    def _gen_vertical_cmd(self, gp: float = 0.0, direction: int = 1):
        # NOTE: direction 1 means forward, -1 backward 
        err_msg = f"[GaitGenerator: _gen_vertical_cmd] Direction {direction} is not valid."
        assert direction in [-1, 1], err_msg
        
        # get the swing foot index
        swing_foot_idx = 0 if (gp < 0.5) else 1
        
        # no changing targets
        zero_orn_offset = np.array([1, 0, 0, 0], dtype=np.float32)
        zero_pos_offset = np.zeros(3, dtype=np.float32)
        
        # adjust the steering angle
        steering_angle = np.clip(self.steering_angle, -np.pi, np.pi) if direction == 1 else 0.0
        steering_foot_idx = 0 if (steering_angle >= 0 and steering_angle <= np.pi) else 1
        steering_orn_offset = np_R.from_euler("z", steering_angle).as_quat(scalar_first=True)
        
        # gait info gen
        gait_info = np.array([np.cos(2 * np.pi * gp), np.sin(2 * np.pi * gp)])
        # pos gen
        l_pos_offset = np.array([direction * self.vertical_dist, self.feet_distance, 0]) if swing_foot_idx == 0 else zero_pos_offset
        r_pos_offset = np.array([direction * self.vertical_dist, -self.feet_distance, 0]) if swing_foot_idx == 1 else zero_pos_offset
        # orn gen
        l_orn_offset = steering_orn_offset if steering_foot_idx == 0 else zero_orn_offset
        r_orn_offset = steering_orn_offset if steering_foot_idx == 1 else zero_orn_offset

        return l_pos_offset, l_orn_offset, r_pos_offset, r_orn_offset, gait_info
    
    def _gen_lateral_cmd(self, gp: float = 0.0, direction: int = 1):
        # NOTE: direction 1 means left, -1 right 
        err_msg = f"[GaitGenerator: _gen_lateral_cmd] Direction {direction} is not valid."
        assert direction in [-1, 1], err_msg
        
        # get the swing foot index
        swing_foot_idx = 0 if (gp < 0.5) else 1
        
        # no changing targets
        zero_orn_offset = np.array([1, 0, 0, 0], dtype=np.float32)
        zero_pos_offset = np.zeros(3, dtype=np.float32)
        
        # clip the movement for the "evil foot"
        max_evil_movement = self.lateral_dist / 2.0 # np.clip(self.lateral_dist, 0, self.feet_distance)
        
        # craft gait_info
        gait_info = np.array([np.cos(2 * np.pi * gp), np.sin(2 * np.pi * gp)])
        
        if direction == 1: # left movement
            l_pos_offset = np.array([0.0, self.lateral_dist, 0.0]) if swing_foot_idx == 0 else zero_pos_offset
            r_pos_offset = np.array([0.0, -max_evil_movement, 0.0]) if swing_foot_idx == 1 else zero_pos_offset
            l_orn_offset = zero_orn_offset
            r_orn_offset = zero_orn_offset
        else: # right movement
            l_pos_offset = np.array([0.0, max_evil_movement, 0.0]) if swing_foot_idx == 0 else zero_pos_offset
            r_pos_offset = np.array([0.0, -self.lateral_dist, 0.0]) if swing_foot_idx == 1 else zero_pos_offset
            l_orn_offset = zero_orn_offset
            r_orn_offset = zero_orn_offset
        
        return l_pos_offset, l_orn_offset, r_pos_offset, r_orn_offset, gait_info

    def _gen_diag_cmd(self, gp: float = 0.0, direction: int = 1):
        # NOTE: direction 1 means left, -1 right 
        err_msg = f"[GaitGenerator: _gen_diag_cmd] Direction {direction} is not valid."
        assert direction in [-1, 1], err_msg
        
         # get the swing foot index
        swing_foot_idx = 0 if (gp < 0.5) else 1
        
        # no changing targets
        zero_orn_offset = np.array([1, 0, 0, 0], dtype=np.float32)
        zero_pos_offset = np.zeros(3, dtype=np.float32)
        
        # gait info gen
        gait_info = np.array([np.cos(2 * np.pi * gp), np.sin(2 * np.pi * gp)])
        
        # clip the movement for the "evil foot"
        max_evil_movement = self.lateral_dist / 2.0
        
        if direction == 1: # left movement
            l_pos_offset = np.array([self.lateral_dist, self.lateral_dist, 0.0]) if swing_foot_idx == 0 else zero_pos_offset
            r_pos_offset = np.array([-max_evil_movement, -max_evil_movement, 0.0]) if swing_foot_idx == 1 else zero_pos_offset
            l_orn_offset = zero_orn_offset
            r_orn_offset = zero_orn_offset
        else: # right movement
            l_pos_offset = np.array([-max_evil_movement, max_evil_movement, 0.0]) if swing_foot_idx == 0 else zero_pos_offset
            r_pos_offset = np.array([self.lateral_dist, -self.lateral_dist, 0.0]) if swing_foot_idx == 1 else zero_pos_offset
            l_orn_offset = zero_orn_offset
            r_orn_offset = zero_orn_offset

        return l_pos_offset, l_orn_offset, r_pos_offset, r_orn_offset, gait_info

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
        # initialize the angle to be still
        self.swing_foot_idx = 0
        
        # retrieve the parameters
        self.gait_frequency = self.command["gait_frequency"]
        self.vert_dist = self.command["vertical_distance"]
        self.lat_dist = self.command["lateral_distance"]
        self.feet_dist = self.command["feet_distance"]
        self.steering_angle = np.deg2rad(self.command["steering_angle"])
        self.mode = "STILL"
        self.first_step = False
        self.GG = GaitGenerator(feet_distance=self.feet_dist, vertical_dist=self.vert_dist, lateral_dist=self.lat_dist, steering_angle=self.steering_angle)

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
        
        l_offset, l_orn_offset, r_offset, r_orn_offset, gait_info = self.GG.query_cmd(
            mov_dir=self.mode, reset=self.first_step, gp=gait_process
        )
        
        if self.first_step:
            self.first_step = False
                    
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
        # q_des[3] = -1.2 # arms contraint
        # q_des[7] = 1.2 # arms contraint

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
        return rot_mat.T @ vector
        
    def check_state(self):
        self.robot.update_robot_state()

        if self.robot.joy_key is not None:
            if self.robot.joy_key.lt and self.robot.joy_key.a and self.robot.key_count == 2:  # start: LT + A
                if self.robot.control_started:
                    self.agent_started = True
                    self.mode = "STILL"
                    self.first_step = False
                    self.node.get_logger().info("Agent started.")
                else:
                    self.node.get_logger().warn("Please start the control first by pressing LT + START.")

            if self.agent_started:
                if self.robot.joy_key.hat_u and self.robot.key_count == 1:
                    # up key
                    self.mode = "FWD"
                    self.first_step = False
                elif self.robot.joy_key.hat_d and self.robot.key_count == 1:
                    # down key
                    self.mode = "BWD"
                    self.first_step = False
                elif self.robot.joy_key.hat_l and self.robot.key_count == 1:
                    # left key
                    self.mode = "LEFT"
                    self.first_step = False
                    self.counter = 0.0
                elif self.robot.joy_key.hat_r and self.robot.key_count == 1:
                    # right key
                    self.mode = "RIGHT"
                    self.first_step = False
                    self.counter = 0.5 / (self.policy_dt * self.gait_frequency)
                elif self.robot.joy_key.a and self.robot.key_count == 1:
                    self.mode = "STILL"
                    self.first_step = True
                elif self.robot.joy_key.b and self.robot.key_count == 1:
                    self.mode = "DIAG-R"
                    self.first_step = False
                    self.counter = 0.5 / (self.policy_dt * self.gait_frequency)
                elif self.robot.joy_key.x and self.robot.key_count == 1:
                    self.mode = "DIAG-L"
                    self.first_step = False
                    self.counter = 0.0
                elif self.robot.joy_key.rx * -1 >= 1.0:
                    pass
                elif self.robot.joy_key.rx * -1 <= -1.0:
                    pass
                if (self.robot.joy_key.ls or self.robot.joy_key.rs) and self.robot.key_count == 1:
                    pass
            else:
                self.mode = "STILL"
                self.first_step = True

            self.robot.joy_key = None  # Reset joy_key
            self.robot.key_count = 0 # Reset true_count after processing
            self.robot.joy_axes = np.zeros(6, dtype=np.float32)

        if not self.robot.control_started:
            self.agent_started = False


if __name__ == "__main__":
    rclpy.init()
    # cfg_file = "src/humanoid_bridge/robot_bridge/example/T1/configs/T1.yaml"
    cfg_file = "config_sim2sim.yaml"
    with open(cfg_file, "r", encoding="utf-8") as f:
        policy_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    node = rclpy.create_node('robot_client_node')
    controller = RobotController(node, policy_cfg)
    
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()