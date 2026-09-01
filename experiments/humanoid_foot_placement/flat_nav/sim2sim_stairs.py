import time
import os
import sys
import mujoco
import mujoco.viewer
import numpy as np
import hydra
from omegaconf import DictConfig
import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation as np_R

from loco_mujoco.environments.utils import add_stair, add_stair_and_flat, add_stairs_platform_stairs
from loco_mujoco.algorithms import PPOJax

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

EPS = 0.15

# Collision geom names that can realistically contact a stair riser.
# Taken from booster_t1_original.xml.
FOOT_AND_CALF_GEOMS = frozenset([
    "left_foot_1", "left_foot_2",
    "right_foot_1", "right_foot_2",
    "left_calf", "right_calf",
])

# Global keyboard state
keyboard_state = {
    'paused': True,
    'reset_requested': False
}

def key_callback(keycode):
    """Callback for keyboard events"""
    if keycode == ord('P') or keycode == ord('p'):
        keyboard_state['paused'] = not keyboard_state['paused']
        print(f"{'Paused' if keyboard_state['paused'] else 'Running'}")
    elif keycode == ord('\\'):  # Backslash for reset
        keyboard_state['reset_requested'] = True
        print("Reset requested...")

# =====================================================================================================================
class StairPlanGenerator:
    def __init__(self, start_x, num_steps, step_len, step_height, step_width, feet_spacing, n_flat_steps=3, platform_length=None):
        self.start_x = start_x
        self.num_steps = num_steps
        self.step_len = step_len
        self.step_height = step_height
        self.step_width = step_width
        self.feet_spacing = feet_spacing
        self.n_flat_steps = n_flat_steps
        self.platform_length = platform_length if platform_length is not None else step_len * n_flat_steps

    def generate_plan(self):
        # generate a list of footholds for the left and right foot (shape is (N,3) for both)
        left_plan = []
        right_plan = []
        
        # initial stance
        left_plan.append([0.0, self.feet_spacing/2, 0.0])
        right_plan.append([0.0, -self.feet_spacing/2, 0.0])

        # flat plane
        n_approach = int(self.start_x / self.step_len)
        if abs(self.start_x - n_approach * self.step_len) < 1e-6:
            n_approach -= 1
        
        for i in range(1, n_approach + 1):
            x = i * self.step_len
            left_plan.append([x, self.feet_spacing/2, 0.0])
            right_plan.append([x, -self.feet_spacing/2, 0.0])

        last_x = n_approach * self.step_len

        # stairs up
        for i in range(self.num_steps):
            # height increases
            z = (i + 1) * self.step_height
            target_x = self.start_x + i * self.step_len
            
            left_plan.append([target_x, self.feet_spacing/2, z])
            right_plan.append([target_x, -self.feet_spacing/2, z])
            
            last_x = target_x
            last_z = z

        # platform part
        platform_start_x = last_x + self.step_len / 2.0
        platform_end_x = platform_start_x + self.platform_length + self.step_len / 2.0
        
        n_platform = max(1, int(self.platform_length / self.step_len))
        
        for i in range(n_platform):
            x = platform_start_x + (i + 0.5) * self.step_len
            left_plan.append([x, self.feet_spacing/2, last_z])
            right_plan.append([x, -self.feet_spacing/2, last_z])

        # stairs down
        for i in range(self.num_steps):
            # height decreases
            z = last_z - (i) * self.step_height
            target_x = platform_end_x + i * self.step_len
            
            left_plan.append([target_x, self.feet_spacing/2, z])
            right_plan.append([target_x, -self.feet_spacing/2, z])
            
            last_x = target_x
    
        # final plane
        for i in range(1, n_approach + 1):
            x = last_x + i * self.step_len
            left_plan.append([x, self.feet_spacing/2, 0.0])
            right_plan.append([x, -self.feet_spacing/2, 0.0])

        return np.array(left_plan, dtype=np.float32), np.array(right_plan, dtype=np.float32)

# =====================================================================================================================
class PlanFollowingGaitGenerator:
    def __init__(
            self, model, data, left_plan, right_plan, gait_freq, policy_dt, max_vert: float = 0.4, 
            feet_dist: float = 0.2, target_yaw: float = 0.0, mode: str = "fast", tolerance: float = 0.1, debug: bool = False):
        self.model = model
        self.data = data
        
        self.left_plan = left_plan
        self.right_plan = right_plan
        
        self.gait_freq = gait_freq
        self.policy_dt = policy_dt
        self.gait_phase = 0.0
        self.max_vert = max_vert
        self.feet_dist = feet_dist
        self.target_yaw = target_yaw
        self.l_idx = 0 # idx left plan
        self.r_idx = 0 # idx right plan
        half_yaw = self.target_yaw / 2.0
        self.target_quat = np.array([np.cos(half_yaw), 0, 0, np.sin(half_yaw)])  # foot orientation target
        assert mode in ["slow", "fast"]
        self.mode = mode
        self.tolerance = tolerance
        self.debug = debug
        self.prev_phase = 0.0 # to detect switches
        
        self.left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        self.right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

        # storing variables for giving the commands during the phase
        self.reset(data)

    def reset(self, data):
        """Reset the plan indices and gait phase."""
        self.l_idx = 0
        self.r_idx = 0
        self.placement_events = [] # for logging
        
        # retrieve the robot yaw
        robot_quat = data.qpos[3:7]  # wxyz format
        robot_yaw = np_R.from_quat([robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]]).as_euler('xyz')[2]
        
        # Calculate yaw difference relative to target direction
        yaw_diff = robot_yaw - self.target_yaw
        # Normalize to [-pi, pi]
        yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
        
        if self.debug:
            print(f"  Robot yaw: {np.rad2deg(robot_yaw):.2f}°, Target yaw: {np.rad2deg(self.target_yaw):.2f}°, Diff: {np.rad2deg(yaw_diff):.2f}°")
        
        # If robot is rotated left (positive yaw diff), move RIGHT foot first (phase 0.5)
        # If robot is rotated right (negative yaw diff), move LEFT foot first (phase 0.0)
        if yaw_diff < 0:
            # Robot leaning right -> move LEFT foot first
            self.gait_phase = 0.0
            self.r_off_phase_cmd = np.zeros(3)
            self.r_orn_phase_cmd = np.array([1, 0, 0, 0])
            self.l_off_phase_cmd, self.l_orn_phase_cmd, _, _, _ = self.get_observation_cmd(data)
            if self.debug:
                print(f"  Starting with LEFT foot swing (phase=0.0)")
        else:
            # Robot leaning left -> move RIGHT foot first
            self.gait_phase = 0.5
            self.l_off_phase_cmd = np.zeros(3)
            self.l_orn_phase_cmd = np.array([1, 0, 0, 0])
            _, _, self.r_off_phase_cmd, self.r_orn_phase_cmd, _ = self.get_observation_cmd(data)
            if self.debug:
                print(f"  Starting with RIGHT foot swing (phase=0.5)")
        
        self.prev_phase = self.gait_phase

        # Clip foot offsets
        self.l_off_phase_cmd[0] = np.clip(self.l_off_phase_cmd[0], -self.max_vert, self.max_vert) 
        self.l_off_phase_cmd[1] = np.clip(self.l_off_phase_cmd[1], 0, self.feet_dist)
        self.r_off_phase_cmd[0] = np.clip(self.r_off_phase_cmd[0], -self.max_vert, self.max_vert) 
        self.r_off_phase_cmd[1] = np.clip(self.r_off_phase_cmd[1], -self.feet_dist, 0)
        
        if self.debug:
            print(f"  Initial commands: L={self.l_off_phase_cmd}, R={self.r_off_phase_cmd}")

        self.gp_off = self.policy_dt * self.gait_freq

    def _check_foot_reached_target(self, foot_pos, target_pos):  
        """Check if foot is close enough to its target (X distance)"""
        dist = np.linalg.norm(foot_pos[0] - target_pos[0])
        reached = dist <= self.tolerance
        return reached

    def update(self, data):
        self.prev_phase = self.gait_phase
        self.gait_phase = (self.gait_phase + self.policy_dt * self.gait_freq) % 1.0
        
        # switching phase from right swing foot to left
        if (self.prev_phase > 0.5 and self.gait_phase < 0.5):
            # Right foot just finished swinging and becomes stance
            right_foot_pos = data.site_xpos[self.right_foot_id]
            right_target = self.right_plan[self.r_idx]

            # Record foot placement error
            err_3d = float(np.linalg.norm(right_foot_pos - right_target))
            err_2d = float(np.linalg.norm(right_foot_pos[:2] - right_target[:2]))
            self.placement_events.append({
                'side': 'right', 'step_idx': self.r_idx,
                'target': right_target.copy(), 'actual': right_foot_pos.copy(),
                'error_2d': err_2d, 'error_3d': err_3d,
            })
            accum = sum(e['error_3d'] for e in self.placement_events)

            # Advance both indices when target is reached
            tar_z = self.right_plan[self.r_idx][2]
            if self._check_foot_reached_target(right_foot_pos, right_target):
                # Advance right (which just became stance)
                self.r_idx = min(self.r_idx + 1, len(self.right_plan) - 1)
                # Maintain swing = stance + 1
                self.l_idx = min(self.r_idx, len(self.left_plan) - 1)
            # else: keep indices and try again next cycle

            # right foot is stance
            self.r_off_phase_cmd = np.zeros(3)
            self.r_orn_phase_cmd = np.array([1,0,0,0])

            # computing offsets for the left foot
            self.l_off_phase_cmd, self.l_orn_phase_cmd = self.get_observation_cmd(data)[:2]

            # clip values
            self.l_off_phase_cmd[0] = np.clip(self.l_off_phase_cmd[0], -self.max_vert, self.max_vert) 
            self.l_off_phase_cmd[1] = np.clip(self.l_off_phase_cmd[1], EPS, self.feet_dist)
            self.l_off_phase_cmd[2] = self.left_plan[self.l_idx][2] - tar_z
        
        # switching phase from left swing foot to right
        if (self.prev_phase <= 0.5 and self.gait_phase > 0.5):
            # Left foot just finished swinging and becomes stance
            left_foot_pos = data.site_xpos[self.left_foot_id]
            left_target = self.left_plan[self.l_idx]

            # Record foot placement error
            err_3d = float(np.linalg.norm(left_foot_pos - left_target))
            err_2d = float(np.linalg.norm(left_foot_pos[:2] - left_target[:2]))
            self.placement_events.append({
                'side': 'left', 'step_idx': self.l_idx,
                'target': left_target.copy(), 'actual': left_foot_pos.copy(),
                'error_2d': err_2d, 'error_3d': err_3d,
            })
            accum = sum(e['error_3d'] for e in self.placement_events)

            # Advance both indices when target is reached
            tar_z = self.left_plan[self.l_idx][2]
            if self._check_foot_reached_target(left_foot_pos, left_target):
                # Advance left (which just became stance)
                self.l_idx = min(self.l_idx + 1, len(self.left_plan) - 1)
                # Maintain swing = stance + 1
                self.r_idx = min(self.l_idx, len(self.right_plan) - 1)
            # else: keep indices and try again next cycle

            # left foot is stance
            self.l_off_phase_cmd = np.zeros(3)
            self.l_orn_phase_cmd = np.array([1,0,0,0])

            # computing offsets for the right foot
            self.r_off_phase_cmd, self.r_orn_phase_cmd = self.get_observation_cmd(data)[2:4]

            # clip values
            self.r_off_phase_cmd[0] = np.clip(self.r_off_phase_cmd[0], -self.max_vert, self.max_vert) 
            self.r_off_phase_cmd[1] = np.clip(self.r_off_phase_cmd[1], -self.feet_dist, -EPS)
            # self.r_off_phase_cmd[2] = self.right_plan[self.r_idx][2] - self.left_plan[self.l_idx][2]
            self.r_off_phase_cmd[2] = self.right_plan[self.r_idx][2] - tar_z
        
        # print(f"{self.l_idx}, {self.r_idx}")

    def placement_summary(self):
        if not self.placement_events:
            return "No foot placements recorded."
        errs_3d = [e['error_3d'] for e in self.placement_events]
        errs_2d = [e['error_2d'] for e in self.placement_events]
        lines = [
            f"Foot Placement Summary ({len(self.placement_events)} placements):",
            f"  Mean 3D error : {np.mean(errs_3d):.3f} m",
            f"  Mean 2D error : {np.mean(errs_2d):.3f} m",
            f"  Max  3D error : {np.max(errs_3d):.3f} m",
            f"  Accumulated 3D: {sum(errs_3d):.3f} m",
        ]
        for e in self.placement_events:
            lines.append(
                f"    {e['side']:5s} step {e['step_idx']:2d} | "
                f"2D={e['error_2d']:.3f}m  3D={e['error_3d']:.3f}m | "
                f"tgt={np.round(e['target'], 3)}  act={np.round(e['actual'], 3)}"
            )
        return "\n".join(lines)
    
    def get_observation_cmd(self, data):
        is_left_swing = (self.gait_phase < 0.5)
        
        if is_left_swing:
            stance_id = self.right_foot_id
            swing_target_pos = self.left_plan[self.l_idx]
        else:
            stance_id = self.left_foot_id
            swing_target_pos = self.right_plan[self.r_idx]

        stance_pos = data.site_xpos[stance_id]
        stance_mat = data.site_xmat[stance_id].reshape(3, 3)
        stance_quat = np_R.from_matrix(stance_mat).as_quat(scalar_first=True) # wxyz
        
        # Target for swing foot (relative to current stance foot)
        target_quat = self.target_quat # np.array([1, 0, 0, 0]) # wxyz
        swing_rel_pos, swing_rel_quat = self._to_stance_frame(
            swing_target_pos, target_quat, stance_pos, stance_mat, stance_quat
        )
        
        # Stance foot always has zero offset relative to itself
        stance_rel_pos = np.zeros(3)
        stance_rel_quat = np.array([1, 0, 0, 0])
        
        # Assign to left/right based on which is swinging
        if is_left_swing:
            l_rel_pos, l_rel_quat = swing_rel_pos, swing_rel_quat
            r_rel_pos, r_rel_quat = stance_rel_pos, stance_rel_quat
        else:
            r_rel_pos, r_rel_quat = swing_rel_pos, swing_rel_quat
            l_rel_pos, l_rel_quat = stance_rel_pos, stance_rel_quat
        
        gait_info = np.array([np.cos(2 * np.pi * self.gait_phase), np.sin(2 * np.pi * self.gait_phase)])

        # print(f"[DEBUG] get_obs_cmd - L_rel_pos: {l_rel_pos}, R_rel_pos: {r_rel_pos}, phase: {self.gait_phase:.3f}, swing={'LEFT' if is_left_swing else 'RIGHT'}")
        
        return l_rel_pos, l_rel_quat, r_rel_pos, r_rel_quat, gait_info

    def get_cmd(self):
        gait_info = np.array([np.cos(2 * np.pi * self.gait_phase), np.sin(2 * np.pi * self.gait_phase)])
        return self.l_off_phase_cmd, self.l_orn_phase_cmd, self.r_off_phase_cmd, self.r_orn_phase_cmd, gait_info

    def _to_stance_frame(self, t_pos, t_quat, s_pos, s_mat, s_quat_wxyz):
        rel_pos = s_mat.T @ (t_pos - s_pos)
        
        def wxyz_to_xyzw(q): return np.array([q[1], q[2], q[3], q[0]])
        def xyzw_to_wxyz(q): return np.array([q[3], q[0], q[1], q[2]])

        r_s = np_R.from_quat(wxyz_to_xyzw(s_quat_wxyz))
        r_t = np_R.from_quat(wxyz_to_xyzw(t_quat))
        
        rel_rot = r_s.inv() * r_t
        rel_quat_xyzw = rel_rot.as_quat()
        
        return rel_pos, xyzw_to_wxyz(rel_quat_xyzw)

# =====================================================================================================================
class LMJPolicy:
    def __init__(self, policy_path: str) -> None:
        agent_conf, agent_state = PPOJax.load_agent(policy_path)

        train_state = agent_state.train_state
        train_state.params["log_std"] = np.ones_like(train_state.params["log_std"]) * -np.inf

        key = jax.random.key(0)
        key, _rng = jax.random.split(key)

        self.agent_conf = agent_conf
        self.train_state = train_state
        self._rng = _rng

        self.network_apply = agent_conf.network.apply
        self._jit_sample_action = jax.jit(self._sample_action, static_argnames=["network_apply"])
        print("Policy loaded and JIT function compiled.")

    @staticmethod
    def _sample_action(network_apply, params, run_stats, rng, obs):
        y, updates = network_apply({"params": params, "run_stats": run_stats}, obs, mutable=["run_stats"])
        pi, _ = y
        a = pi.mode()
        a = jnp.atleast_2d(a)
        return a

    def predict_action(self, obs):
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
    return (target_q - q) * kp + (target_dq - dq) * kd


def detect_failure_mode(model, data, stair_prefix="stair_1", fall_height=0.3,
                        robot_geoms=FOOT_AND_CALF_GEOMS, current_world_height=0.0):
    """Classify the current failure mode."""
    if (data.qpos[2] - current_world_height) >= fall_height:
        return None

    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""

        # Identify which geom is the stair and which is the robot
        g1_is_stair = stair_prefix in g1
        g2_is_stair = stair_prefix in g2
        if not (g1_is_stair ^ g2_is_stair):  # both stair or neither stair then skip
            continue

        robot_geom = g2 if g1_is_stair else g1
        if robot_geom not in robot_geoms:
            continue  # only care about foot/calf contacts

        # MuJoCo stores the contact frame row-major; the normal is the first *column*
        # geometrically, so nz = frame[6].  A riser has a horizontal normal (nz ≈ 0).
        if abs(c.frame[6]) < 0.3:
            return 'step_riser_contact'

    return 'balance'


# =====================================================================================================================
@hydra.main(config_name="fp_config.yaml")
def main(config: DictConfig):
    # laoding stuff
    STEP_LEN = config["command"]["step_spacing"]  # Maximum foot placement target distance
    N_STEPS = 5
    sign = config["command"]["direction"]
    STAIR_START_X = sign * 6 * STEP_LEN  # Must be multiple of STEP_LEN for proper alignment (2.1m)
    STEP_HEIGHT = config["command"]["step_height"]
    STEP_WIDTH = 2.0
    PLATFORM_LENGTH = STEP_LEN * 4
    FEET_DIST = float(config["command"]["feet_distance"])
    MAX_VERT = 0.4
    obs_dim = 16 
    success = False
    fallen = False
    riser_contact_times = []   # sim times (s) when a riser contact was detected
    mode = config["command"]["mode"]
    tolerance = config["command"].get("tolerance", 0.1)
    debug = config["command"].get("debug", False)

    xml_path = config["xml_path"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    agent_path = config["agent_path"]

    kps = np.array(config["lmj_kps"], dtype=np.float32)
    kds = np.array(config["lmj_kds"], dtype=np.float32)

    default_angles = np.array(config["default_angles"], dtype=np.float32)
    min_angles = np.array(config["min_angles"], dtype=np.float32)
    max_angles = np.array(config["max_angles"], dtype=np.float32)

    num_qj = len(default_angles)
    num_actions = config["num_actions"]
    cmd_params = config["command"]

    # --- Load Policy ---
    policy = LMJPolicy(policy_path=agent_path)

    num_obs = 3 + num_qj + 3 + num_qj + num_actions + 6
    print(f"Policy expects an observation size of: {num_obs}")
    print("Warming up the policy network for JIT compilation...")
    total_obs = max(policy.agent_conf.network.actor_obs_ind.max(), policy.agent_conf.network.critic_obs_ind.max()) + 1
    for _ in range(500):
        dummy_obs = jnp.zeros((1, total_obs), dtype=np.float32)
        _ = policy.predict_action(dummy_obs)
    print("Warmup complete.")

    # --- Setup MuJoCo ---
    spec = mujoco.MjSpec.from_file(xml_path)
    wb = spec.worldbody
    
    # Determine orientation based on obstacle position
    is_backward = STAIR_START_X < 0
    orientation_yaw = 180.0 if is_backward else 0.0
    yaw_off = cmd_params["yaw_off"]
    orientation_yaw += yaw_off
    
    first_step_center = [STAIR_START_X, 0.0, STEP_HEIGHT/2]

    wb = add_stairs_platform_stairs(
        world_body=wb,
        name="stair_1",
        first_step_coordinates=first_step_center,
        num_steps=N_STEPS,
        step_height=STEP_HEIGHT,
        step_length=STEP_LEN,
        step_width=STEP_WIDTH,
        orientation_yaw_deg=orientation_yaw,
        platform_length=PLATFORM_LENGTH,
        backend=np
    )
    
    # Always generate plan with positive coordinates, then flip if needed
    planner = StairPlanGenerator(
        np.abs(STAIR_START_X), N_STEPS, STEP_LEN, STEP_HEIGHT, STEP_WIDTH, FEET_DIST, platform_length=PLATFORM_LENGTH
    )
    l_plan, r_plan = planner.generate_plan()
    
    # If obstacle is backward, negate x-coordinates (keep same sequence)
    if is_backward:
        l_plan[:, 0] = -l_plan[:, 0]
        r_plan[:, 0] = -r_plan[:, 0]
        # l_plan = np.flip(l_plan, axis=0).copy() 
        # r_plan = np.flip(r_plan, axis=0).copy()
    
    # Rotate the foothold plans to match obstacle orientation (for yaw_off)
    if yaw_off != 0.0:
        # Convert yaw offset to radians (not total orientation_yaw, just the offset)
        yaw_rad = np.deg2rad(yaw_off)
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)
        
        # Rotate around the stair start position
        center = np.array([STAIR_START_X, 0.0, 0.0])
        
        for i in range(len(l_plan)):
            # Translate to origin
            pos = l_plan[i] - center
            # Rotate
            x_new = pos[0] * cos_yaw - pos[1] * sin_yaw
            y_new = pos[0] * sin_yaw + pos[1] * cos_yaw
            # Translate back
            l_plan[i, 0] = x_new + center[0]
            l_plan[i, 1] = y_new + center[1]
            # z stays the same
        
        for i in range(len(r_plan)):
            pos = r_plan[i] - center
            x_new = pos[0] * cos_yaw - pos[1] * sin_yaw
            y_new = pos[0] * sin_yaw + pos[1] * cos_yaw
            r_plan[i, 0] = x_new + center[0]
            r_plan[i, 1] = y_new + center[1]
    
    for i, pos in enumerate(l_plan):
        wb.add_site(name=f"tgt_L_{i}", pos=pos, size=(0.02, 0.02, 0.02), rgba=(0, 0, 1, 0.5), group=2)
    for i, pos in enumerate(r_plan):
        wb.add_site(name=f"tgt_R_{i}", pos=pos, size=(0.02, 0.02, 0.02), rgba=(1, 0, 0, 0.5), group=2)

    for geom in spec.geoms:
        if geom.name.endswith("_col"):
            geom.delete()

    m = spec.compile()
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # init state from policy config
    initial_qpos = np.array(config["init_state_params"]["qpos_init"], dtype=np.float32)
    initial_qvel = np.array(config["init_state_params"]["qvel_init"], dtype=np.float32)
    random_yaw_deg = np.random.uniform(-30, 30)
    random_yaw_rad = np.deg2rad(random_yaw_deg)
    random_quat = np_R.from_euler('z', random_yaw_rad).as_quat(scalar_first=True)
    initial_qpos[3:7] = random_quat
    print(f"  Randomized initial yaw: {random_yaw_deg:.2f}°")
    if len(initial_qpos) == m.nq and len(initial_qvel) == m.nv:
        d.qpos[:] = initial_qpos
        d.qvel[:] = initial_qvel
        print("Initial state loaded from policy config.")
    else:
        print("Warning: init state length mismatch. Using default init.")

    mujoco.mj_forward(m, d)
    initial_qpos = d.qpos.copy()
    initial_qvel = d.qvel.copy()

    # --- Controller ---
    gait_freq = float(config["command"]["gait_frequency"])
    planner_ctrl = PlanFollowingGaitGenerator(
        m, d, l_plan, r_plan, gait_freq, simulation_dt * control_decimation, max_vert=MAX_VERT, 
        feet_dist=FEET_DIST, target_yaw=np.deg2rad(orientation_yaw), tolerance=tolerance, debug=debug,
    )

    # PD Gains
    kps = np.array(config["lmj_kps"], dtype=np.float32)
    kds = np.array(config["lmj_kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    target_dof_pos = default_angles.copy()

    # controller state
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()

    cmd = np.zeros(obs_dim, dtype=np.float32)
    
    keyboard_state['paused'] = True  # Start paused
    keyboard_state['reset_requested'] = False
    
    print("\n=== Simulation Ready ===")
    print("Click on the viewer window, then:")
    print("Press 'P' to start/pause")
    print("Press '\\' (backslash) to reset\n")

    # --- Simulation Loop ---
    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            if not success and np.linalg.norm(d.qpos[:2] - planner_ctrl.left_plan[-1][:2]) < 0.5:
                success = True
                print("\n=== SUCCESS ===")
                print(planner_ctrl.placement_summary())

            # --- Failure detection ---
            if not fallen and not success:
                # Track riser contacts.
                # Also print ALL robot–stair contacts so we can verify geom names
                # and determine the correct contact-frame convention empirically.
                for ci in range(d.ncon):
                    c = d.contact[ci]
                    g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
                    g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
                    g1_stair = "stair_1" in g1
                    g2_stair = "stair_1" in g2
                    if not (g1_stair ^ g2_stair):
                        continue
                    robot_geom = g2 if g1_stair else g1
                    in_foot_set = robot_geom in FOOT_AND_CALF_GEOMS
                    if in_foot_set and (abs(c.frame[2]) < 0.3 or abs(c.frame[6]) < 0.3):
                        riser_contact_times.append(d.time)
                        # print(f"  --> RISER CONTACT at t={d.time:.2f}s")
                        break

                # Detect fall
                current_world_height = max(
                    planner_ctrl.left_plan[planner_ctrl.l_idx][2],
                    planner_ctrl.right_plan[planner_ctrl.r_idx][2]
                )
                failure_mode = detect_failure_mode(model=m, data=d, current_world_height=current_world_height)
                if failure_mode is not None:
                    # here we add some delay tolerance
                    RISER_LOOKBACK = 1.5  
                    if failure_mode == 'balance' and riser_contact_times:
                        last_riser_t = riser_contact_times[-1]
                        if (d.time - last_riser_t) <= RISER_LOOKBACK:
                            failure_mode = 'step_riser_contact'

                    fallen = True
                    print(f"\n=== FAILURE DETECTED at t={d.time:.2f}s ===")
                    print(f"  Failure mode : {failure_mode}")
                    if riser_contact_times:
                        recent = riser_contact_times[-5:]
                        print(f"  Riser contact events (last 5): {[f'{t:.2f}s' for t in recent]}")
                        print(f"  Total riser contact events   : {len(riser_contact_times)}")
                    else:
                        print("  No riser contacts recorded.")
                    print(planner_ctrl.placement_summary())
                    # input()

            # Handle reset request
            if keyboard_state['reset_requested']:
                print("Resetting controller and robot state")
                # Reset physics state
                d.qpos[:] = initial_qpos
                d.qvel[:] = initial_qvel
                mujoco.mj_forward(m, d)

                # Reset controller (now with updated foot positions)
                planner_ctrl.reset(d)
                action[:] = 0.0
                target_dof_pos[:] = default_angles
                keyboard_state['paused'] = True
                keyboard_state['reset_requested'] = False
                print("Reset complete - press 'P' to start")
                success = False
                fallen = False
                riser_contact_times.clear()
            
            if keyboard_state['paused']:
                viewer.sync()
                time.sleep(0.01)
                continue
            
            for _ in range(control_decimation):
                # PD Control
                tau = (target_dof_pos - d.qpos[7:]) * kps + (0.0 - d.qvel[6:]) * kds
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)
            
            planner_ctrl.update(d)
            
            l_off, l_orn, r_off, r_orn, gait_info = planner_ctrl.get_cmd()
            # l_off, l_orn, r_off, r_orn, gait_info = planner_ctrl.get_observation_cmd()
            # print(f"[DEBUG] LEFT: {l_off} - {l_orn}\t RIGHT: {r_off} - {r_orn}")
            
            # Basic state
            qj = d.qpos[7:]
            dqj = d.qvel[6:]
            quat = d.qpos[3:7]            
            base_ang_vel = d.qvel[3:6]
            projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
            
            # Command Vector
            cmd = np.concatenate([l_off, l_orn, r_off, r_orn, gait_info])
            
            obs_list = []
            obs_list += projected_gravity.flatten().tolist()
            obs_list += qj.flatten().tolist()
            obs_list += (base_ang_vel * 1.0).flatten().tolist()
            obs_list += (dqj * 0.1).flatten().tolist()
            obs_list += action.flatten().tolist()
            obs_list += cmd.flatten().tolist()

            # obs = [0.0] * 78 + obs_list
            critic_n_obs = 78 
            obs = [0.0] * critic_n_obs + obs_list
            obs = np.array(obs, dtype=np.float32).reshape(1, -1)

            # --- Policy Inference ---
            emitted_action = np.asarray(policy.predict_action(obs)).flatten()
            # emitted_action = np.clip(emitted_action, -1.0, 1.0)
            clipped_action = np.clip(emitted_action, -1.0, 1.0)
            action = emitted_action

            target_dof_pos = clipped_action[:num_qj] + default_angles[:num_qj]
            target_dof_pos = np.clip(target_dof_pos, min_angles, max_angles)
            
            viewer.sync()
            
            time_until_next = m.opt.timestep * control_decimation - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)

if __name__ == "__main__":
    main()