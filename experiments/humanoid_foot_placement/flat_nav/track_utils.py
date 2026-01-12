import numpy as np, jax.numpy as jnp
from typing import Union, List
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as np_R
from jax._src.scipy.spatial.transform import Rotation as jnp_R
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast

def quat_rotate_inverse(q, v):
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

@dataclass
class Checkpoint:
    chk_pos: Union[np.array, jnp.array, List]
    next_pos: Union[np.array, jnp.array, List]
    mov_mode: str
    xy_max_offset: float
    z_offset: float

class GaitGenerator:
    def __init__(
        self,
        feet_distance: float = 0.2,
        stop_steps: int = 2,
        vertical_dist: float = 0.0,
        lateral_dist: float = 0.0,
        steering_angle: float = 0.0,
    ):
        self.feet_distance = feet_distance
        self.vertical_dist = vertical_dist
        self.lateral_dist = lateral_dist
        self.steering_angle = steering_angle
        self.stop_steps = stop_steps
        self.z_dist = 0.0
        
        self.gaits_to_still = 0
        self.n_gaits = 0
        self.move_dir = "STILL"
        self.swing_foot_idx = 0  
        self.sample_goal = False
        self.foot_offset = [
            np.array([0, self.feet_distance, 0.0]),
            np.array([1, 0, 0, 0]),
            np.array([0, -self.feet_distance, 0.0]),
            np.array([1, 0, 0, 0]),
        ]

        self.teleop = dict(
            move_enabled=False,
            mov_dir="STILL",
            vert_step=0.05,
            vert_min=-0.5,
            vert_max=0.5,
            yaw_step=np.deg2rad(5.0),
            yaw_min=(-np.pi / 2.0),
            yaw_max=(np.pi / 2.0),
            lat_step=0.05,
            lat_min=-0.3,
            lat_max=0.3,
        )

    def preprocess_gp_info(self, gp: float):
        swing_foot_idx = 0 if (gp < 0.5) else 1

        if self.swing_foot_idx != swing_foot_idx:
            self.sample_goal = True
            self.n_gaits += 1
            self.swing_foot_idx = swing_foot_idx
        else:
            self.sample_goal = False

        mov_dir = self.teleop["mov_dir"] if self.teleop["move_enabled"] else "STILL"
        
        if mov_dir != self.move_dir:
            if mov_dir != "STILL":
                self.gaits_to_still = self.stop_steps
            self.move_dir = mov_dir

        if self.gaits_to_still > 0:
            gait_info = np.array([np.cos(2 * np.pi * gp), np.sin(2 * np.pi * gp)])
        else:
            gait_info = np.array([0.0, 0.0])
        return gait_info

    def query_cmd(self, gp: float = 0.0, mov_dir=None, reset=False):
        gait_info = self.preprocess_gp_info(gp=gp)

        if mov_dir is not None:
            self.move_dir = mov_dir

        if self.sample_goal:
            if reset:
                self.move_dir = "STILL"
                
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

    def _get_steering_info(self):
        steering_angle = np.clip(self.steering_angle, -np.pi, np.pi)
        steering_foot_idx = 0 if (steering_angle >= 0 and steering_angle <= np.pi) else 1
        
        steering_orn_offset = np_R.from_euler("z", steering_angle).as_quat(scalar_first=True)
        zero_orn_offset = np.array([1, 0, 0, 0], dtype=np.float32)
        
        l_orn = steering_orn_offset # if steering_foot_idx == 0 else zero_orn_offset
        r_orn = steering_orn_offset # if steering_foot_idx == 1 else zero_orn_offset
        
        return l_orn, r_orn

    def _gen_still_cmd(self):
        zero_orn_offset = np.array([1, 0, 0, 0], dtype=np.float32)
        zero_pos_offset = np.zeros(3, dtype=np.float32)

        if self.gaits_to_still > 0:
            l_pos_offset = np.array([0, self.feet_distance, 0.0]) if self.swing_foot_idx == 0 else zero_pos_offset
            r_pos_offset = np.array([0, -self.feet_distance, 0.0]) if self.swing_foot_idx == 1 else zero_pos_offset
            l_orn_offset = zero_orn_offset
            r_orn_offset = zero_orn_offset
        else:
            l_pos_offset = np.array([0, self.feet_distance, 0.0])
            r_pos_offset = np.array([0, -self.feet_distance, 0.0])
            l_orn_offset = zero_orn_offset
            r_orn_offset = zero_orn_offset

        return l_pos_offset, l_orn_offset, r_pos_offset, r_orn_offset

    def _gen_vertical_cmd(self):
        zero_pos_offset = np.zeros(3, dtype=np.float32)
        l_orn_offset, r_orn_offset = self._get_steering_info()

        l_pos_offset = (
            np.array([self.vertical_dist, self.feet_distance, self.z_dist]) if self.swing_foot_idx == 0 else zero_pos_offset
        )
        r_pos_offset = (
            np.array([self.vertical_dist, -self.feet_distance, self.z_dist]) if self.swing_foot_idx == 1 else zero_pos_offset
        )

        return l_pos_offset, l_orn_offset, r_pos_offset, r_orn_offset

    def _gen_lateral_cmd(self):
        direction = 1 if self.lateral_dist >= 0 else -1
        if abs(self.lateral_dist) < 1e-4: direction = 0

        zero_pos_offset = np.zeros(3, dtype=np.float32)
        l_orn_offset, r_orn_offset = self._get_steering_info()

        total_x_offset = self.vertical_dist

        lat_dist = self.feet_distance * direction + self.lateral_dist
        max_evil_movement = -self.feet_distance * direction / 2.0

        if direction == 1:  # Left
            if self.swing_foot_idx == 0:
                l_pos_offset = np.array([total_x_offset, lat_dist, 0.0])
            else:
                l_pos_offset = zero_pos_offset
            
            if self.swing_foot_idx == 1:
                r_pos_offset = np.array([total_x_offset, max_evil_movement, 0.0])
            else:
                r_pos_offset = zero_pos_offset
                
        elif direction == -1:  # Right
            if self.swing_foot_idx == 0:
                l_pos_offset = np.array([total_x_offset, max_evil_movement, 0.0])
            else:
                l_pos_offset = zero_pos_offset
                
            if self.swing_foot_idx == 1:
                r_pos_offset = np.array([total_x_offset, lat_dist, 0.0])
            else:
                r_pos_offset = zero_pos_offset
                
        else: # Still 
            if abs(self.vertical_dist) > 0.01:
                l_pos_offset = np.array([total_x_offset, self.feet_distance, 0.0])
                r_pos_offset = np.array([total_x_offset, -self.feet_distance, 0.0])
            else:
                l_pos_offset = np.array([0, self.feet_distance, 0])
                r_pos_offset = np.array([0, -self.feet_distance, 0])

        return l_pos_offset, l_orn_offset, r_pos_offset, r_orn_offset

    def _gen_diag_cmd(self, direction: int = 1):
        assert direction in [-1, 1]
        zero_pos_offset = np.zeros(3, dtype=np.float32)
        l_orn_offset, r_orn_offset = self._get_steering_info()
        max_evil_movement = self.lateral_dist / 2.0
        
        dx = self.vertical_dist
        dy = self.lateral_dist
        
        if direction == 1: # Left Diag
            if self.swing_foot_idx == 0:
                l_pos_offset = np.array([dx, dy, 0.0])
            else:
                l_pos_offset = zero_pos_offset
            
            if self.swing_foot_idx == 1:
                r_pos_offset = np.array([-max_evil_movement, -max_evil_movement, 0.0])
            else:
                r_pos_offset = zero_pos_offset
        else: # Right Diag
            if self.swing_foot_idx == 0:
                l_pos_offset = np.array([-max_evil_movement, max_evil_movement, 0.0])
            else:
                l_pos_offset = zero_pos_offset
                
            if self.swing_foot_idx == 1:
                r_pos_offset = np.array([dx, -dy, 0.0])
            else:
                r_pos_offset = zero_pos_offset

        return l_pos_offset, l_orn_offset, r_pos_offset, r_orn_offset

    def key_callback(self, keycode):
        pass

class CheckpointController:
    def __init__(
            self, 
            gait_generator, 
            checkpoints: List, 
            dt: float = 0.01,
            kp_lin: float = 1.0,
            kp_ang: float = 2.0,
            max_step_len: float = 0.2,
            max_lat_len: float = 0.3,
            max_steer_deg: float = 45.0,
            dist_threshold: float = 0.3
        ):
        self.gg = gait_generator
        self.checkpoints = checkpoints
        self.dt = dt
        self.current_chk_idx = 0
        self.finished = False
        
        # Pause Logic
        self.pause = True
        self.pause_start_gait = 0

        self.kp_lin = kp_lin
        self.kp_ang = kp_ang
        
        self.max_step_len = max_step_len
        self.max_lat_len = max_lat_len
        self.max_steer_rad = np.deg2rad(max_steer_deg)
        self.dist_threshold = dist_threshold
        
        self.cmd_vert = 0.0
        self.cmd_lat = 0.0
        self.cmd_steer = 0.0
        
        self.accel_vert = 0.2 
        self.accel_lat = 0.5
        self.accel_ang = np.deg2rad(90)
        
        self.gg.teleop["move_enabled"] = True

    def _rate_limit(self, target, current, max_rate):
        delta = target - current
        limit = max_rate * self.dt
        delta_clamped = np.clip(delta, -limit, limit)
        return current + delta_clamped

    def get_command(self, robot_pos: np.ndarray, robot_quat: np.ndarray, gp: float, backend = np):
        R = np_R if backend == np else jnp_R

        if self.pause and (self.gg.n_gaits - self.pause_start_gait) >= 3:
            self.pause = False
        
        if self.finished or self.pause:
            self.cmd_vert = self._rate_limit(0.0, self.cmd_vert, self.accel_vert)
            self.cmd_lat = self._rate_limit(0.0, self.cmd_lat, self.accel_lat)
            self.cmd_steer = self._rate_limit(0.0, self.cmd_steer, self.accel_ang)
            
            self.gg.teleop["mov_dir"] = "FWD"
            self.gg.vertical_dist = float(self.cmd_vert)
            self.gg.lateral_dist = float(self.cmd_lat)
            self.gg.steering_angle = float(self.cmd_steer)
            return self.gg.query_cmd(gp)

        target_chk = self.checkpoints[self.current_chk_idx]
        target_pos = np.array(target_chk.chk_pos)
        self.max_step_len = target_chk.xy_max_offset
        self.gg.z_dist = target_chk.z_offset
        
        diff_vec = target_pos[:2] - robot_pos[:2]
        dist_to_target = np.linalg.norm(diff_vec)
        
        tolerance = self.dist_threshold
        if dist_to_target < tolerance:
            print(f"[Controller] Reached Checkpoint {self.current_chk_idx} at {target_pos}")
            if (self.current_chk_idx + 1) >= len(self.checkpoints):
                print("[Controller] Course Complete.")
                self.finished = True
            else:
                self.current_chk_idx += 1
                self.pause = True
                self.pause_start_gait = self.gg.n_gaits
                return self.gg.query_cmd(gp)

        if len(robot_quat) == 4:
            r_rot = R.from_quat(quat_scalarfirst2scalarlast(robot_quat))
        
        robot_yaw = r_rot.as_euler('zxy')[0]
        cos_yaw = np.cos(robot_yaw)
        sin_yaw = np.sin(robot_yaw)
        
        local_x = diff_vec[0] * cos_yaw + diff_vec[1] * sin_yaw
        local_y = -diff_vec[0] * sin_yaw + diff_vec[1] * cos_yaw
        
        mode = target_chk.mov_mode
        target_vert = 0.0
        target_lat = 0.0
        target_steer = 0.0
        z_cmd = self.gg.z_dist
        move_dir_str = "FWD"

        heading_err = np.arctan2(local_y, local_x)

        if mode == "LATERAL" and not self.pause:
            side_sign = 1 if local_y > 0 else -1
            target_heading = side_sign * (np.pi / 2.0)
            
            align_err = heading_err - target_heading
            
            ALIGN_THRESHOLD = np.deg2rad(10) # 10 degrees tolerance
            
            if abs(align_err) > ALIGN_THRESHOLD:
                # Turn in Place 
                move_dir_str = "FWD"
                target_lat = 0.0
                target_vert = 0.0
                target_steer = align_err * self.kp_ang
            else:
                # Step Sideways + Minor Alignment
                move_dir_str = "LEFT" if side_sign > 0 else "RIGHT"
                
                target_lat = np.clip(local_y * self.kp_lin, -self.max_lat_len, self.max_lat_len)
                target_vert = np.clip(local_x * self.kp_lin, -0.05, 0.05) # Tiny shuffle for X-alignment
                target_steer = align_err * self.kp_ang
            
            target_steer = np.clip(target_steer, -self.max_steer_rad, self.max_steer_rad)

        elif mode == "DIAGONAL" and not self.pause:
            target_vert = np.clip(local_x * self.kp_lin, -self.max_step_len, self.max_step_len)
            target_lat = np.clip(local_y * self.kp_lin, -self.max_lat_len, self.max_lat_len)
            target_steer = np.clip(heading_err * self.kp_ang, -self.max_steer_rad, self.max_steer_rad)
            
            if abs(target_lat) > 0.05:
                move_dir_str = "DIAG-L" if target_lat > 0 else "DIAG-R"
            else:
                move_dir_str = "FWD" if target_vert >= 0 else "BWD"

        elif mode == "FWD" and not self.pause:
            target_vert = np.clip(local_x * self.kp_lin, -self.max_step_len, self.max_step_len)
            target_steer = np.clip(heading_err * (self.kp_ang * 0.75), -self.max_steer_rad, self.max_steer_rad)
            target_lat = 0.0 
            
            if abs(heading_err) > np.deg2rad(30):
                target_vert = 0.0 
                z_cmd = 0.0
            
            move_dir_str = "FWD" if target_vert >= -0.01 else "BWD"
            if abs(target_vert) < 0.01 and abs(target_steer) < 0.01:
                move_dir_str = "STILL"

        # Apply Rate Limiting
        self.cmd_vert = self._rate_limit(target_vert, self.cmd_vert, self.accel_vert)
        self.cmd_lat = self._rate_limit(target_lat, self.cmd_lat, self.accel_lat)
        self.cmd_steer = self._rate_limit(target_steer, self.cmd_steer, self.accel_ang)

        self.gg.vertical_dist = float(self.cmd_vert)
        self.gg.lateral_dist = float(self.cmd_lat)
        self.gg.steering_angle = float(self.cmd_steer)
        self.gg.teleop["mov_dir"] = move_dir_str
        self.gg.z_dist = z_cmd
        
        return self.gg.query_cmd(gp)