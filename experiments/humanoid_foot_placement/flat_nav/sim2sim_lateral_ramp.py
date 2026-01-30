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

from loco_mujoco.environments.utils import add_ramp_platform_ramp
from loco_mujoco.algorithms import PPOJax

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

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
class LateralRampPlanGenerator:
    """
    Generates foothold plans for lateral movement over a ramp structure.
    The robot moves sideways (left or right) crossing: ramp up -> platform -> ramp down.
    """
    def __init__(self, start_y, run, rise, platform_length, width, feet_spacing, 
                 step_size=0.25, direction='left', min_feet_distance=0.2):
        """
        Args:
            start_y: Y coordinate where the ramp structure starts
            run: Horizontal length of each ramp section
            rise: Height change of each ramp
            platform_length: Length of the flat platform
            width: Width of the ramp structure (along X axis)
            feet_spacing: Distance between left and right foot in X direction
            step_size: Lateral step size for foothold placement
            direction: 'left' or 'right' - direction of lateral movement
            min_feet_distance: Minimum distance to maintain between feet (default 0.2m)
        """
        self.start_y = start_y
        self.run = run
        self.rise = rise
        self.platform_length = platform_length
        self.width = width
        self.feet_spacing = feet_spacing
        self.step_size = step_size
        self.direction = direction
        self.min_feet_distance = min_feet_distance
        
        # Calculate key positions
        self.ramp_up_end_y = start_y + run
        self.platform_end_y = self.ramp_up_end_y + platform_length
        self.ramp_down_end_y = self.platform_end_y + run
        
    def generate_plan(self):
        """
        Generate foothold plans for left and right feet.
        Only the lead foot (left for left movement, right for right movement) 
        follows the actual ramp plan. The other foot maintains lateral spacing.
        """
        # Direction multiplier: +1 for left (positive Y), -1 for right (negative Y)
        dir_mult = 1 if self.direction == 'left' else -1
        is_left_lead = (self.direction == 'left')
        
        # Determine X positions for left and right feet
        # Left foot at +feet_spacing/2, right foot at -feet_spacing/2
        left_x = self.feet_spacing / 2.0
        right_x = -self.feet_spacing / 2.0
        
        # Generate the Y-coordinate schedule (same for both feet)
        y_schedule = []
        
        # Initial stance
        y_schedule.append(0.0)
        
        # Approach steps - move laterally towards the ramp
        num_approach = int(abs(self.start_y) / self.step_size)
        for i in range(1, num_approach + 1):
            y = dir_mult * i * self.step_size
            y_schedule.append(y)
        
        # RAMP UP - ascending while moving laterally
        num_ramp_steps = max(2, int(self.run / self.step_size))
        
        for i in range(1, num_ramp_steps + 1):
            progress = i / num_ramp_steps
            y = dir_mult * (abs(self.start_y) + progress * self.run)
            y_schedule.append(y)
        
        # PLATFORM - flat section at the top
        num_platform = max(2, int(self.platform_length / self.step_size))
        
        for i in range(1, num_platform + 1):
            progress = i / num_platform
            y = dir_mult * (abs(self.start_y) + self.run + progress * self.platform_length)
            y_schedule.append(y)
        
        # RAMP DOWN - descending while moving laterally
        for i in range(1, num_ramp_steps + 1):
            progress = i / num_ramp_steps
            y = dir_mult * (abs(self.start_y) + self.run + self.platform_length + progress * self.run)
            y_schedule.append(y)
        
        # Exit steps - continue moving laterally on flat ground
        last_y = y_schedule[-1]
        for i in range(1, num_approach + 1):
            y = last_y + dir_mult * i * self.step_size
            y_schedule.append(y)
        
        # Now create plans for both feet based on their X positions
        left_plan = []
        right_plan = []
        
        for y in y_schedule:
            # Calculate height for each foot based on its actual position
            # The feet are offset in X, so they're at slightly different Y positions on the rotated ramp
            # But since the ramp is oriented along Y axis, X offset doesn't affect the height calculation
            z_left = self._calculate_height_at_y(abs(y))
            z_right = self._calculate_height_at_y(abs(y))
            
            left_plan.append([left_x, y, z_left])
            right_plan.append([right_x, y, z_right])
        
        return np.array(left_plan, dtype=np.float32), np.array(right_plan, dtype=np.float32)
    
    def _calculate_height_at_y(self, y_abs):
        """
        Calculate the height (Z) at a given absolute Y position along the ramp structure.
        
        Args:
            y_abs: Absolute Y position (always positive, already adjusted for direction)
        
        Returns:
            Height (Z) at that position
        """
        # Before ramp: z = 0
        if y_abs < abs(self.start_y):
            return 0.0
        
        # On ramp up
        ramp_up_end = abs(self.start_y) + self.run
        if y_abs < ramp_up_end:
            progress = (y_abs - abs(self.start_y)) / self.run
            return progress * self.rise
        
        # On platform
        platform_end = ramp_up_end + self.platform_length
        if y_abs < platform_end:
            return self.rise
        
        # On ramp down
        ramp_down_end = platform_end + self.run
        if y_abs < ramp_down_end:
            progress = (y_abs - platform_end) / self.run
            return self.rise * (1.0 - progress)
        
        # After ramp: z = 0
        return 0.0

# =====================================================================================================================
class PlanFollowingGaitGenerator:
    def __init__(self, model, data, left_plan, right_plan, gait_freq, policy_dt, max_vert: float = 0.4, feet_dist: float = 0.2):
        self.model = model
        self.data = data
        
        self.left_plan = left_plan
        self.right_plan = right_plan
        
        self.gait_freq = gait_freq
        self.policy_dt = policy_dt
        self.gait_phase = 0.0
        self.max_vert = max_vert
        self.feet_dist = feet_dist

        self.l_idx = 0 # idx left plan
        self.r_idx = 0 # idx right plan
        
        self.prev_phase = 0.0 # to detect switches
        
        self.left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        self.right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

        # storing variables for giving the commands during the phase
        self.l_off_phase_cmd = np.zeros(3)
        self.r_off_phase_cmd = np.zeros(3)
        self.l_orn_phase_cmd = np.array([1,0,0,0])
        self.r_orn_phase_cmd = np.array([1,0,0,0])

        self.gp_off = policy_dt * gait_freq

    def reset(self):
        """Reset the plan indices and gait phase."""
        self.l_idx = 0
        self.r_idx = 0
        self.gait_phase = 0.0
        self.prev_phase = 0.0

        self.l_off_phase_cmd = np.zeros(3)
        self.r_off_phase_cmd = np.zeros(3)
        self.l_orn_phase_cmd = np.array([1,0,0,0])
        self.r_orn_phase_cmd = np.array([1,0,0,0])
        print(f"Controller reset: l_idx={self.l_idx}, r_idx={self.r_idx}, phase={self.gait_phase}")

        self.gp_off = self.policy_dt * self.gait_freq

    def set_gp_offset(self, gp_off: float, min_off: float = 0.01, max_off: float = 0.04):
        gp_off = np.clip(gp_off, -1, 1)
        self.gp_off = gp_off * (max_off - min_off) / 2.0 + (max_off + min_off)/ 2.0
        print(f"[DEBUG] Gait Phase Offset set to: {self.gp_off:.4f}")

    def update(self):
        self.prev_phase = self.gait_phase
        self.gait_phase = (self.gait_phase + self.gp_off) % 1.0
        
        # switching phase from right swing foot to left
        if self.prev_phase > 0.5 and self.gait_phase < 0.5:
            self.l_idx = min(self.r_idx + 1, len(self.left_plan) - 1)

            # right foot is stance
            self.r_off_phase_cmd = np.zeros(3)
            self.r_orn_phase_cmd = np.array([1,0,0,0])

            # computing offsets for the left foot
            self.l_off_phase_cmd, self.l_orn_phase_cmd = self.get_observation_cmd()[:2]

            if (self.l_idx == len(self.left_plan) -1 and self.r_idx == len(self.right_plan) -1) or \
                (self.l_idx == 0 and self.r_idx == 0):
                self.l_off_phase_cmd[2] = 0
            self.l_off_phase_cmd[0] = np.clip(self.l_off_phase_cmd[0], -self.feet_dist, self.feet_dist) 
            self.l_off_phase_cmd[1] = np.clip(self.l_off_phase_cmd[1], -self.max_vert, self.max_vert)
        
        # switching phase from left swing foot to right
        if self.prev_phase <= 0.5 and self.gait_phase > 0.5:
            self.r_idx = min(self.l_idx + 1, len(self.right_plan) - 1)

            # left foot is stance
            self.l_off_phase_cmd = np.zeros(3)
            self.l_orn_phase_cmd = np.array([1,0,0,0])

            # computing the offsets for the right foot
            self.r_off_phase_cmd, self.r_orn_phase_cmd = self.get_observation_cmd()[2:4]

            if (self.l_idx == len(self.left_plan) -1 and self.r_idx == len(self.right_plan) -1) or \
                (self.l_idx == 0 and self.r_idx == 0):
                self.r_off_phase_cmd[2] = 0
            self.r_off_phase_cmd[0] = np.clip(self.r_off_phase_cmd[0], -self.feet_dist, self.feet_dist) 
            self.r_off_phase_cmd[1] = np.clip(self.r_off_phase_cmd[1], -self.max_vert, self.max_vert)

        print(f"[DEBUG] Gait Phase: {self.gait_phase:.3f}, l_idx: {self.l_idx}, r_idx: {self.r_idx}")

    def get_observation_cmd(self):
        """
        Returns:
            l_offset (3), l_quat (4), r_offset (3), r_quat (4)
            All expressed in the STANCE FOOT frame.
        """
        is_left_swing = (self.gait_phase < 0.5)
        
        if is_left_swing:
            stance_id = self.right_foot_id
        else:
            stance_id = self.left_foot_id

        stance_pos = self.data.site_xpos[stance_id]
        stance_mat = self.data.site_xmat[stance_id].reshape(3, 3)
        stance_quat = np_R.from_matrix(stance_mat).as_quat(scalar_first=True) # wxyz
        
        target_pos_L = self.left_plan[self.l_idx]
        target_pos_R = self.right_plan[self.r_idx]
        
        target_quat_L = np.array([1, 0, 0, 0]) # wxyz
        target_quat_R = np.array([1, 0, 0, 0])
        
        l_rel_pos, l_rel_quat = self._to_stance_frame(target_pos_L, target_quat_L, stance_pos, stance_mat, stance_quat)
        r_rel_pos, r_rel_quat = self._to_stance_frame(target_pos_R, target_quat_R, stance_pos, stance_mat, stance_quat)

        if is_left_swing:
            r_rel_pos = np.zeros(3)
            r_rel_quat = np.array([1, 0, 0, 0])
        else:
            l_rel_pos = np.zeros(3)
            l_rel_quat = np.array([1, 0, 0, 0])
        
        gait_info = np.array([np.cos(2 * np.pi * self.gait_phase), np.sin(2 * np.pi * self.gait_phase)])
        
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

# =====================================================================================================================
@hydra.main(config_name="fp_config.yaml")
def main(config: DictConfig):
    # Loading parameters
    RUN = 1.5  # Horizontal length of ramp
    RISE = 0.4  # Height of ramp
    PLATFORM_LENGTH = 1.5  # Length of flat platform
    RAMP_WIDTH = 2.0  # Width of the ramp structure
    STEP_SIZE = 0.25  # Lateral step size
    
    # Direction: 'left' for positive Y, 'right' for negative Y
    sign = config["command"].get("direction", 1)
    direction = "left" if sign == 1 else "right"
    RAMP_START_Y = sign * 1.5  # Starting Y position of the ramp
    
    FEET_DIST = float(config["command"].get("feet_distance", 0.2))
    MIN_FEET_DISTANCE = 0.2  # Minimum distance between feet
    MAX_VERT = 0.4

    xml_path = config["xml_path"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    agent_path = config["agent_path"]

    kps = np.array(config["lmj_kps"], dtype=np.float32)
    kds = np.array(config["lmj_kds"], dtype=np.float32)

    default_angles = np.array(config["default_angles"], dtype=np.float32)
    min_angles = np.array(config["min_angles"], dtype=np.float32)
    max_angles = np.array(config["max_angles"], dtype=np.float32)
    asymmetric = config["scale_action_to_jnt_limits"]

    num_qj = len(default_angles)
    num_actions = config["num_actions"]
    is_gp_adaptive = config["command"].get("is_gp_adaptive", False)
    num_actions += 1 if is_gp_adaptive else 0
    cmd_params = config["command"]

    # --- Load Policy ---
    lmj_hydra_config = hydra.compose(config_name="conf_t1")
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
    
    # Determine orientation for lateral movement (90 or -90 degrees from forward)
    # Left movement: rotate structure 90° (facing left)
    # Right movement: rotate structure -90° (facing right)
    orientation_yaw = 90.0 if direction == "left" else -90.0
    
    # Starting position of the ramp structure (bottom of first ramp)
    # For lateral movement, the ramp extends along Y axis when rotated
    ramp_start_pos = [0.0, RAMP_START_Y, 0.0]

    # Add the ramp-platform-ramp structure
    wb = add_ramp_platform_ramp(
        world_body=wb,
        name="lateral_ramp",
        coordinates=ramp_start_pos,
        run=RUN,
        rise=RISE,
        platform_length=PLATFORM_LENGTH,
        platform_width=RAMP_WIDTH,
        width=RAMP_WIDTH,
        thickness=0.1,
        orientation_yaw_deg=orientation_yaw,
        color=[0.3, 0.3, 0.3, 1.0],
        friction=[1.0, 0.005, 0.0001],
        backend=np
    )
    
    # Generate foothold plan
    planner = LateralRampPlanGenerator(
        start_y=RAMP_START_Y,
        run=RUN,
        rise=RISE,
        platform_length=PLATFORM_LENGTH,
        width=RAMP_WIDTH,
        feet_spacing=FEET_DIST,
        step_size=STEP_SIZE,
        direction=direction,
        min_feet_distance=MIN_FEET_DISTANCE
    )
    l_plan, r_plan = planner.generate_plan()
    
    # Add visual markers only for the lead foot (the one actively following the ramp)
    if direction == "left":
        # Left foot is lead - show only left markers
        for i, pos in enumerate(l_plan):
            wb.add_site(name=f"tgt_L_{i}", pos=pos, size=(0.02, 0.02, 0.02), rgba=(0, 0, 1, 0.8), group=2)
    else:
        # Right foot is lead - show only right markers
        for i, pos in enumerate(r_plan):
            wb.add_site(name=f"tgt_R_{i}", pos=pos, size=(0.02, 0.02, 0.02), rgba=(1, 0, 0, 0.8), group=2)

    # Remove collision geoms if they exist
    for geom in spec.geoms:
        if geom.name.endswith("_col"):
            geom.delete()

    m = spec.compile()
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Initialize state from policy config
    try:
        initial_qpos = np.array(config["init_state_params"]["qpos_init"], dtype=np.float32)
        initial_qvel = np.array(config["init_state_params"]["qvel_init"], dtype=np.float32)
        if len(initial_qpos) == m.nq and len(initial_qvel) == m.nv:
            d.qpos[:] = initial_qpos
            d.qvel[:] = initial_qvel
            print("Initial state loaded from policy config.")
        else:
            print("Warning: init state length mismatch. Using default init.")
    except Exception as e:
        print(f"Warning: Could not load initial state ({e}). Using default init.")

    mujoco.mj_forward(m, d)

    # --- Controller ---
    gait_freq = float(config["command"]["gait_frequency"])
    planner_ctrl = PlanFollowingGaitGenerator(
        m, d, l_plan, r_plan, gait_freq, 
        simulation_dt * control_decimation, 
        max_vert=MAX_VERT,
        feet_dist=MIN_FEET_DISTANCE
    )

    # Controller state
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()

    keyboard_state['paused'] = True  # Start paused
    keyboard_state['reset_requested'] = False
    
    print("\n=== Lateral Ramp Navigation Ready ===")
    print(f"Direction: {direction.upper()}")
    print("Click on the viewer window, then:")
    print("Press 'P' to start/pause")
    print("Press '\\' (backslash) to reset\n")

    # --- Simulation Loop ---
    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        start_time = time.time()
        while viewer.is_running():
            step_start = time.time()
            
            # Handle reset request
            if keyboard_state['reset_requested']:
                print("Resetting controller")
                planner_ctrl.reset()
                action[:] = 0.0
                target_dof_pos[:] = default_angles
                keyboard_state['paused'] = True
                keyboard_state['reset_requested'] = False
            
            if keyboard_state['paused']:
                viewer.sync()
                time.sleep(0.01)
                continue
            
            for _ in range(control_decimation):
                # PD Control
                tau = (target_dof_pos - d.qpos[7:]) * kps + (0.0 - d.qvel[6:]) * kds
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)
            
            planner_ctrl.update()
            
            # Get commands
            l_off, l_orn, r_off, r_orn, gait_info = planner_ctrl.get_cmd()
            print(f"[DEBUG] LEFT: {l_off} - {l_orn}\t RIGHT: {r_off} - {r_orn}")
            
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

            critic_n_obs = 78 if not is_gp_adaptive else 79
            obs = [0.0] * critic_n_obs + obs_list
            obs = np.array(obs, dtype=np.float32).reshape(1, -1)

            # Override Head Pitch Angle in Observation
            obs[0, critic_n_obs + 3] = 0.0  # Head Yaw Angle
            obs[0, critic_n_obs + 4] = 0.0  # Head Pitch joint position

            # --- Policy Inference ---
            emitted_action = np.asarray(policy.predict_action(obs)).flatten()
            if is_gp_adaptive:
                planner_ctrl.set_gp_offset(emitted_action[-1], min_off=0.01, max_off=0.01)
            
            clipped_action = np.clip(emitted_action, -1.0, 1.0)
            if asymmetric:
                neg = - min_angles + default_angles
                pos = max_angles - default_angles
                clipped_action = np.clip(clipped_action, None, 0.0) * neg + np.clip(clipped_action, 0.0, None) * pos
            action = emitted_action

            # Target dof pos
            target_dof_pos = clipped_action[:num_qj] + default_angles[:num_qj]

            # Head override
            target_dof_pos[0] = 0.0
            target_dof_pos[1] = 0.0
            # Force the arms
            target_dof_pos[3] = -1.2
            target_dof_pos[7] = 1.2
            target_dof_pos = np.clip(target_dof_pos, min_angles, max_angles)
            
            viewer.sync()
            
            time_until_next = m.opt.timestep * control_decimation - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)

if __name__ == "__main__":
    main()
