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

EPS = 0.15

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
class RampPlanGenerator:
    """
    Generates a foot placement plan for a ramp-platform-ramp obstacle.

    The obstacle geometry (matching add_ramp_platform_ramp) is:
      - Ramp up  : from ramp_start_x, horizontal run=RUN, vertical rise=RISE
      - Platform : length PLATFORM_LENGTH at height RISE
      - Ramp down: symmetric descent back to z=0

    Footholds on the ramps are placed at every `step_len` along x, with z
    interpolated linearly along the slope surface. The plan structure mirrors
    StairPlanGenerator from sim2sim_stairs.py:
      1. Initial stance at origin
      2. Flat approach to the ramp
      3. Ramp up footholds
      4. Platform footholds
      5. Ramp down footholds
      6. Flat exit
    """

    def __init__(self, ramp_start_x, run, rise, platform_length, step_len, step_width, feet_spacing, thickness=0.05, n_flat_steps=3):
        self.ramp_start_x     = ramp_start_x       # x where the ramp surface begins
        self.run              = run                 # horizontal length of each ramp
        self.rise             = rise                # total height gained on the ramp
        self.platform_length  = platform_length
        self.step_len         = step_len            # stride length for foothold spacing
        self.step_width       = step_width
        self.feet_spacing     = feet_spacing
        self.thickness        = thickness           # must match add_ramp_platform_ramp
        self.n_flat_steps     = n_flat_steps

    def generate_plan(self):
        left_plan  = []
        right_plan = []

        # Same correction used in add_ramp_platform_ramp / add_slope:
        # the walking surface of the tilted slab sits above the nominal z by
        # thickness * run / (2 * hypotenuse) at both ends.
        hyp = np.sqrt(self.run**2 + self.rise**2)
        surface_z_correction = self.thickness * self.run / (2.0 * hyp)

        # ---- Initial stance ----
        left_plan.append ([0.0,  self.feet_spacing / 2, 0.0])
        right_plan.append([0.0, -self.feet_spacing / 2, 0.0])

        # ---- Flat approach ----
        # Edge-based grid: x = k * step_len for k = 1, 2, ...
        # int() of ramp_start_x/step_len gives the correct count directly
        # (float64 gives 5.999... for 6*step_len, so int()=5, which is right).
        # No -1 needed — same formula as StairPlanGenerator.
        n_approach = int(self.ramp_start_x / self.step_len)

        for i in range(n_approach):
            x = (i + 1) * self.step_len
            left_plan.append ([x,  self.feet_spacing / 2, 0.0])
            right_plan.append([x, -self.feet_spacing / 2, 0.0])

        # ---- Ramp up ----
        # Centred intervals: first foothold at ramp_start_x + 0.5*step_len,
        # which is half a step into the slope — clearly on the surface.
        # ramp_start_x is already shifted back by step_len/2 vs. RAMP_START_X,
        # so last approach at n_approach*step_len and first ramp at
        # ramp_start_x + 0.5*step_len = RAMP_START_X gives gap = step_len ✓
        n_ramp = max(1, int(self.run / self.step_len))
        ramp_up_end_x = self.ramp_start_x + self.run

        for i in range(n_ramp):
            x = self.ramp_start_x + (i + 0.5) * self.step_len
            z = self.rise * (x - self.ramp_start_x) / self.run + surface_z_correction
            left_plan.append ([x,  self.feet_spacing / 2, z])
            right_plan.append([x, -self.feet_spacing / 2, z])

        # Platform z = full rise + correction (flush with ramp top surface)
        last_z = self.rise + surface_z_correction

        # ---- Platform ----
        platform_start_x = ramp_up_end_x
        platform_end_x   = platform_start_x + self.platform_length
        n_platform       = max(1, int(self.platform_length / self.step_len))

        for i in range(n_platform):
            x = platform_start_x + (i + 0.5) * self.step_len
            left_plan.append ([x,  self.feet_spacing / 2, last_z])
            right_plan.append([x, -self.feet_spacing / 2, last_z])

        # ---- Ramp down ----
        # Same centred convention.
        for i in range(n_ramp):
            x = platform_end_x + (i + 0.5) * self.step_len
            t = (x - platform_end_x) / self.run
            z = self.rise * (1.0 - t) + surface_z_correction
            z = max(z, 0.0)
            left_plan.append ([x,  self.feet_spacing / 2, z])
            right_plan.append([x, -self.feet_spacing / 2, z])

        last_x = platform_end_x + self.run

        # ---- Flat exit ----
        # Centred intervals: first exit at last_x + 0.5*step_len, giving
        # exactly step_len gap from the last ramp foothold at last_x - 0.5*step_len.
        for i in range(n_approach):
            x = last_x + (i + 0.5) * self.step_len
            left_plan.append ([x,  self.feet_spacing / 2, 0.0])
            right_plan.append([x, -self.feet_spacing / 2, 0.0])

        return np.array(left_plan, dtype=np.float32), np.array(right_plan, dtype=np.float32)


# =====================================================================================================================
class PlanFollowingGaitGenerator:
    def __init__(
            self, model, data, left_plan, right_plan, gait_freq, policy_dt, max_vert: float = 0.4,
            feet_dist: float = 0.2, target_yaw: float = 0.0, mode: str = "fast", tolerance: float = 0.15,):
        self.model      = model
        self.data       = data
        self.left_plan  = left_plan
        self.right_plan = right_plan
        self.gait_freq  = gait_freq
        self.policy_dt  = policy_dt
        self.gait_phase = 0.0
        self.max_vert   = max_vert
        self.feet_dist  = feet_dist
        self.target_yaw = target_yaw
        self.l_idx      = 0
        self.r_idx      = 0
        half_yaw        = self.target_yaw / 2.0
        self.target_quat = np.array([np.cos(half_yaw), 0, 0, np.sin(half_yaw)])
        assert mode in ["slow", "fast"]
        self.mode       = mode
        self.prev_phase = 0.0
        self.tolerance    = tolerance

        self.left_foot_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        self.right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

        self.reset(data)

    def reset(self, data):
        self.l_idx = 0
        self.r_idx = 0
        self.placement_events = []  # list of dicts: {side, step_idx, target, actual, error_2d, error_3d}

        robot_quat = data.qpos[3:7]
        robot_yaw  = np_R.from_quat([robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]]).as_euler('xyz')[2]
        yaw_diff   = np.arctan2(np.sin(robot_yaw - self.target_yaw), np.cos(robot_yaw - self.target_yaw))

        if yaw_diff < 0:
            self.gait_phase      = 0.0
            self.r_off_phase_cmd = np.zeros(3)
            self.r_orn_phase_cmd = np.array([1, 0, 0, 0])
            self.l_off_phase_cmd, self.l_orn_phase_cmd, _, _, _ = self.get_observation_cmd(data)
        else:
            self.gait_phase      = 0.5
            self.l_off_phase_cmd = np.zeros(3)
            self.l_orn_phase_cmd = np.array([1, 0, 0, 0])
            _, _, self.r_off_phase_cmd, self.r_orn_phase_cmd, _ = self.get_observation_cmd(data)

        self.prev_phase = self.gait_phase

        self.l_off_phase_cmd[0] = np.clip(self.l_off_phase_cmd[0], -self.max_vert, self.max_vert)
        self.l_off_phase_cmd[1] = np.clip(self.l_off_phase_cmd[1],  0,             self.feet_dist)
        self.r_off_phase_cmd[0] = np.clip(self.r_off_phase_cmd[0], -self.max_vert, self.max_vert)
        self.r_off_phase_cmd[1] = np.clip(self.r_off_phase_cmd[1], -self.feet_dist, 0)


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
            self.r_off_phase_cmd[2] = self.right_plan[self.r_idx][2] - tar_z
        

    def get_observation_cmd(self, data):
        is_left_swing = (self.gait_phase < 0.5)

        if is_left_swing:
            stance_id       = self.right_foot_id
            swing_target    = self.left_plan[self.l_idx]
        else:
            stance_id       = self.left_foot_id
            swing_target    = self.right_plan[self.r_idx]

        stance_pos  = data.site_xpos[stance_id]
        stance_mat  = data.site_xmat[stance_id].reshape(3, 3)
        stance_quat = np_R.from_matrix(stance_mat).as_quat(scalar_first=True)

        swing_rel_pos, swing_rel_quat = self._to_stance_frame(
            swing_target, self.target_quat, stance_pos, stance_mat, stance_quat
        )
        stance_rel_pos  = np.zeros(3)
        stance_rel_quat = np.array([1, 0, 0, 0])

        if is_left_swing:
            l_rel_pos, l_rel_quat = swing_rel_pos,  swing_rel_quat
            r_rel_pos, r_rel_quat = stance_rel_pos, stance_rel_quat
        else:
            r_rel_pos, r_rel_quat = swing_rel_pos,  swing_rel_quat
            l_rel_pos, l_rel_quat = stance_rel_pos, stance_rel_quat

        gait_info = np.array([np.cos(2 * np.pi * self.gait_phase), np.sin(2 * np.pi * self.gait_phase)])
        return l_rel_pos, l_rel_quat, r_rel_pos, r_rel_quat, gait_info

    def get_cmd(self):
        gait_info = np.array([np.cos(2 * np.pi * self.gait_phase), np.sin(2 * np.pi * self.gait_phase)])
        return self.l_off_phase_cmd, self.l_orn_phase_cmd, self.r_off_phase_cmd, self.r_orn_phase_cmd, gait_info

    def _to_stance_frame(self, t_pos, t_quat, s_pos, s_mat, s_quat_wxyz):
        rel_pos = s_mat.T @ (t_pos - s_pos)

        def wxyz_to_xyzw(q): return np.array([q[1], q[2], q[3], q[0]])
        def xyzw_to_wxyz(q): return np.array([q[3], q[0], q[1], q[2]])

        r_s      = np_R.from_quat(wxyz_to_xyzw(s_quat_wxyz))
        r_t      = np_R.from_quat(wxyz_to_xyzw(t_quat))
        rel_quat = (r_s.inv() * r_t).as_quat()
        return rel_pos, xyzw_to_wxyz(rel_quat)


# =====================================================================================================================
class LMJPolicy:
    def __init__(self, policy_path: str) -> None:
        agent_conf, agent_state = PPOJax.load_agent(policy_path)

        train_state = agent_state.train_state
        train_state.params["log_std"] = np.ones_like(train_state.params["log_std"]) * -np.inf

        key = jax.random.key(0)
        key, _rng = jax.random.split(key)

        self.agent_conf  = agent_conf
        self.train_state = train_state
        self._rng        = _rng

        self.network_apply      = agent_conf.network.apply
        self._jit_sample_action = jax.jit(self._sample_action, static_argnames=["network_apply"])
        print("Policy loaded and JIT function compiled.")

    @staticmethod
    def _sample_action(network_apply, params, run_stats, rng, obs):
        y, updates = network_apply({"params": params, "run_stats": run_stats}, obs, mutable=["run_stats"])
        pi, _ = y
        a = pi.mode()
        return jnp.atleast_2d(a)

    def predict_action(self, obs):
        return self._jit_sample_action(
            self.network_apply, self.train_state.params, self.train_state.run_stats, self._rng, obs
        )


# =====================================================================================================================
def quat_rotate_inverse(q, v):
    """Rotates a vector by the inverse of a quaternion. MuJoCo quaternions are [w, x, y, z]."""
    q_w   = q[0]
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
    STEP_LEN         = config["command"]["step_spacing"]
    sign             = config["command"]["direction"]
    RAMP_START_X     = sign * 6 * STEP_LEN      
    RISE             = float(config["command"]["rise"])
    RUN              = STEP_LEN * 5             
    RAMP_WIDTH       = 2.0
    RAMP_THICKNESS   = 0.05        # must match add_ramp_platform_ramp default
    PLATFORM_LENGTH  = STEP_LEN * 4
    FEET_DIST        = float(config["command"]["feet_distance"])
    MAX_VERT         = 0.4
    obs_dim          = 16
    success          = False
    mode             = config["command"]["mode"]
    tolerance        = config["command"]["tolerance"]

    xml_path           = config["xml_path"]
    simulation_dt      = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    agent_path         = config["agent_path"]

    kps            = np.array(config["lmj_kps"],        dtype=np.float32)
    kds            = np.array(config["lmj_kds"],        dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    min_angles     = np.array(config["min_angles"],     dtype=np.float32)
    max_angles     = np.array(config["max_angles"],     dtype=np.float32)

    num_qj      = len(default_angles)
    num_actions = config["num_actions"]
    cmd_params  = config["command"]

    # --- Load Policy ---
    policy = LMJPolicy(policy_path=agent_path)

    print("Warming up the policy network for JIT compilation...")
    total_obs = max(policy.agent_conf.network.actor_obs_ind.max(), policy.agent_conf.network.critic_obs_ind.max()) + 1
    for _ in range(500):
        _ = policy.predict_action(jnp.zeros((1, total_obs), dtype=np.float32))
    print("Warmup complete.")

    # --- Setup MuJoCo ---
    spec = mujoco.MjSpec.from_file(xml_path)
    wb   = spec.worldbody

    is_backward     = RAMP_START_X < 0
    orientation_yaw = 180.0 if is_backward else 0.0
    yaw_off         = cmd_params["yaw_off"]
    orientation_yaw += yaw_off

    # add_ramp_platform_ramp takes the bottom-start corner of the up-slope as `coordinates`
    wb = add_ramp_platform_ramp(
        world_body=wb,
        name="ramp_1",
        coordinates=[RAMP_START_X - np.sign(RAMP_START_X) * STEP_LEN / 2, 0.0, 0.0],
        run=RUN,
        rise=RISE,
        platform_length=PLATFORM_LENGTH,
        platform_width=RAMP_WIDTH,
        width=RAMP_WIDTH,
        thickness=RAMP_THICKNESS,
        orientation_yaw_deg=orientation_yaw,
        backend=np,
    )

    # --- Foot plan ---
    planner = RampPlanGenerator(
        ramp_start_x=np.abs(RAMP_START_X) - STEP_LEN / 2,
        run=RUN,
        rise=RISE,
        platform_length=PLATFORM_LENGTH,
        step_len=STEP_LEN,
        step_width=RAMP_WIDTH,
        feet_spacing=FEET_DIST,
        thickness=RAMP_THICKNESS,
    )
    l_plan, r_plan = planner.generate_plan()

    # Mirror x if backward
    if is_backward:
        l_plan[:, 0] = -l_plan[:, 0]
        r_plan[:, 0] = -r_plan[:, 0]

    # Rotate for yaw offset
    if yaw_off != 0.0:
        yaw_rad = np.deg2rad(yaw_off)
        cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)
        center = np.array([RAMP_START_X, 0.0, 0.0])

        for plan in [l_plan, r_plan]:
            for i in range(len(plan)):
                pos   = plan[i] - center
                x_new = pos[0] * cos_yaw - pos[1] * sin_yaw
                y_new = pos[0] * sin_yaw + pos[1] * cos_yaw
                plan[i, 0] = x_new + center[0]
                plan[i, 1] = y_new + center[1]

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

    # --- Initial state ---
    initial_qpos = np.array(config["init_state_params"]["qpos_init"], dtype=np.float32)
    initial_qvel = np.array(config["init_state_params"]["qvel_init"], dtype=np.float32)
    random_yaw_deg = np.random.uniform(-30, 30)
    random_quat    = np_R.from_euler('z', np.deg2rad(random_yaw_deg)).as_quat(scalar_first=True)
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
    gait_freq    = float(config["command"]["gait_frequency"])
    planner_ctrl = PlanFollowingGaitGenerator(
        m, d, l_plan, r_plan, gait_freq,
        simulation_dt * control_decimation,
        max_vert=MAX_VERT,
        feet_dist=FEET_DIST,
        target_yaw=np.deg2rad(orientation_yaw),
        mode=mode,
    )

    action         = np.zeros(num_actions,  dtype=np.float32)
    target_dof_pos = default_angles.copy()
    cmd            = np.zeros(obs_dim,       dtype=np.float32)

    keyboard_state['paused']          = True
    keyboard_state['reset_requested'] = False

    print("\n=== Simulation Ready ===")
    print("Click on the viewer window, then:")
    print("Press 'P' to start/pause")
    print("Press '\\' (backslash) to reset\n")

    # --- Simulation Loop ---
    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            if np.linalg.norm(d.qpos[:2] - planner_ctrl.left_plan[-1][:2]) < 0.5:
                if not success:
                    print("SUCCESS")
                success = True

            if keyboard_state['reset_requested']:
                print("Resetting controller and robot state")
                d.qpos[:] = initial_qpos
                d.qvel[:] = initial_qvel
                mujoco.mj_forward(m, d)
                planner_ctrl.reset(d)
                action[:]         = 0.0
                target_dof_pos[:] = default_angles
                keyboard_state['paused']          = True
                keyboard_state['reset_requested'] = False
                success = False
                print("Reset complete - press 'P' to start")

            if keyboard_state['paused']:
                viewer.sync()
                time.sleep(0.01)
                continue

            for _ in range(control_decimation):
                tau = (target_dof_pos - d.qpos[7:]) * kps + (0.0 - d.qvel[6:]) * kds
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)

            planner_ctrl.update(d)
            l_off, l_orn, r_off, r_orn, gait_info = planner_ctrl.get_cmd()
            print(f"[DEBUG] LEFT: {l_off} - {l_orn}\t RIGHT: {r_off} - {r_orn}")

            qj                = d.qpos[7:]
            dqj               = d.qvel[6:]
            quat              = d.qpos[3:7]
            base_ang_vel      = d.qvel[3:6]
            projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))

            cmd = np.concatenate([l_off, l_orn, r_off, r_orn, gait_info])

            obs_list  = []
            obs_list += projected_gravity.flatten().tolist()
            obs_list += qj.flatten().tolist()
            obs_list += (base_ang_vel * 1.0).flatten().tolist()
            obs_list += (dqj * 0.1).flatten().tolist()
            obs_list += action.flatten().tolist()
            obs_list += cmd.flatten().tolist()

            critic_n_obs = 78
            obs = np.array([0.0] * critic_n_obs + obs_list, dtype=np.float32).reshape(1, -1)

            emitted_action = np.asarray(policy.predict_action(obs)).flatten()
            clipped_action = np.clip(emitted_action, -1.0, 1.0)
            action         = emitted_action

            target_dof_pos = clipped_action[:num_qj] + default_angles[:num_qj]
            target_dof_pos = np.clip(target_dof_pos, min_angles, max_angles)

            viewer.sync()

            time_until_next = m.opt.timestep * control_decimation - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)


if __name__ == "__main__":
    main()