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
    def __init__(self, start_x, num_steps, step_len, step_height, step_width, feet_spacing, n_flat_steps=3):
        self.start_x = start_x
        self.num_steps = num_steps
        self.step_len = step_len
        self.step_height = step_height
        self.step_width = step_width
        self.feet_spacing = feet_spacing
        self.n_flat_steps = n_flat_steps 

    def generate_plan(self):
        # generate a list of footholds for the left and right foot (shape is (N,3) for both)
        left_plan = []
        right_plan = []

        # Adjust stair start to be half step before to avoid edge placement
        adjusted_start = self.start_x - self.step_len / 2
        approach_dist = adjusted_start
        
        # Initial stance
        left_plan.append([0.0, self.feet_spacing/2, 0.0])
        right_plan.append([0.0, -self.feet_spacing/2, 0.0])

        # Generate walking steps up to the stairs
        n_approach = int(approach_dist / self.step_len)
        for i in range(1, n_approach + 1):
            x = i * self.step_len
            left_plan.append([x, self.feet_spacing/2, 0.0])
            right_plan.append([x, -self.feet_spacing/2, 0.0])

        last_x = n_approach * self.step_len
        last_z = 0.0

        # stairs part of the plan
        for i in range(1, self.num_steps + 1):
            z = i * self.step_height
            target_x = self.start_x + (i-1)*self.step_len  # center of the step
            
            left_plan.append([target_x, self.feet_spacing/2, z])
            right_plan.append([target_x, -self.feet_spacing/2, z])
            
            last_x = target_x
            last_z = z

        # platform
        for i in range(1, self.n_flat_steps + 1):
            x = last_x + i * self.step_len
            left_plan.append([x, self.feet_spacing/2, last_z])
            right_plan.append([x, -self.feet_spacing/2, last_z])

        return np.array(left_plan, dtype=np.float32), np.array(right_plan, dtype=np.float32)

# =====================================================================================================================
class PlanFollowingGaitGenerator:
    def __init__(self, model, data, left_plan, right_plan, gait_freq, policy_dt):
        self.model = model
        self.data = data
        
        self.left_plan = left_plan
        self.right_plan = right_plan
        
        self.gait_freq = gait_freq
        self.policy_dt = policy_dt
        self.gait_phase = 0.0
        
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

    def update(self):
        self.prev_phase = self.gait_phase
        self.gait_phase = (self.gait_phase + self.policy_dt * self.gait_freq) % 1.0
        
        # switching phase from right swing foot to left
        if self.prev_phase > 0.5 and self.gait_phase < 0.5:
            # self.l_idx = min(self.l_idx + 1, len(self.left_plan) - 1)
            self.l_idx = min(self.r_idx + 1, len(self.left_plan) - 1)

            # right foot is stance
            self.r_off_phase_cmd = np.zeros(3)
            self.r_orn_phase_cmd = np.array([1,0,0,0])

            # computing offsets for the left foot
            self.l_off_phase_cmd, self.l_orn_phase_cmd = self.get_observation_cmd()[:2]

            if (self.l_idx == len(self.left_plan) -1 and self.r_idx == len(self.right_plan) -1) or \
                (self.l_idx == 0 and self.r_idx == 0):
                self.l_off_phase_cmd, self.l_orn_phase_cmd = self.left_plan[-1] - self.right_plan[-1], np.array([1,0,0,0])
        
        # switching phase from right swing foot to left
        if self.prev_phase <= 0.5 and self.gait_phase > 0.5:
            # self.r_idx = min(self.r_idx + 1, len(self.right_plan) - 1)
            self.r_idx = min(self.l_idx + 1, len(self.right_plan) - 1)

            # left foot is stance
            self.l_off_phase_cmd = np.zeros(3)
            self.l_orn_phase_cmd = np.array([1,0,0,0])

            # computing the offsets for the right foot
            self.r_off_phase_cmd, self.r_orn_phase_cmd = self.get_observation_cmd()[2:4]

            if (self.l_idx == len(self.left_plan) -1 and self.r_idx == len(self.right_plan) -1) or \
                (self.l_idx == 0 and self.r_idx == 0):
                self.r_off_phase_cmd, self.r_orn_phase_cmd = self.right_plan[-1] - self.left_plan[-1], np.array([1,0,0,0])

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
    # laoding stuff
    STAIR_START_X = 2.0
    N_STEPS = 5
    STEP_LEN = 0.4
    STEP_HEIGHT = config["command"]["step_height"]
    STEP_WIDTH = 2.0
    FEET_DIST = float(config["command"]["feet_distance"])

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
    
    first_step_center = [STAIR_START_X, 0.0, STEP_HEIGHT/2]
    # wb = add_stair(
    #     world_body=wb,
    #     name="stair_1",
    #     first_step_coordinates=first_step_center,
    #     num_steps=N_STEPS,
    #     step_height=STEP_HEIGHT,
    #     step_length=STEP_LEN,
    #     step_width=STEP_WIDTH,
    #     down=False,
    #     orientation_yaw_deg=0.0,
    #     backend=np
    # )
    # wb = add_stair_and_flat(
    #     world_body=wb,
    #     name="stair_1",
    #     first_step_coordinates=first_step_center,
    #     num_steps=N_STEPS,
    #     step_height=STEP_HEIGHT,
    #     step_length=STEP_LEN,
    #     step_width=STEP_WIDTH,
    #     down=False,
    #     orientation_yaw_deg=0.0,
    #     platform_length=STEP_LEN * 3,
    #     backend=np
    # )
    wb = add_stairs_platform_stairs(
        world_body=wb,
        name="stair_1",
        first_step_coordinates=first_step_center,
        num_steps=N_STEPS,
        step_height=STEP_HEIGHT,
        step_length=STEP_LEN,
        step_width=STEP_WIDTH,
        orientation_yaw_deg=0.0,
        platform_length=STEP_LEN * 3,
        backend=np
    )
    
    planner = StairPlanGenerator(STAIR_START_X, N_STEPS, STEP_LEN, STEP_HEIGHT, STEP_WIDTH, FEET_DIST)
    l_plan, r_plan = planner.generate_plan()
    
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
    planner_ctrl = PlanFollowingGaitGenerator(m, d, l_plan, r_plan, gait_freq, simulation_dt * control_decimation)

    # PD Gains
    kps = np.array(config["lmj_kps"], dtype=np.float32)
    kds = np.array(config["lmj_kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    target_dof_pos = default_angles.copy()

    # controller state
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    target_dof_kps = kps.copy()
    target_dof_kds = kds.copy()

    cmd = np.zeros(16, dtype=np.float32)
    counter = 1
    gait_frequency = float(cmd_params["gait_frequency"])
    
    keyboard_state['paused'] = True  # Start paused
    keyboard_state['reset_requested'] = False
    
    print("\n=== Simulation Ready ===")
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
            
            # l_off, l_orn, r_off, r_orn, gait_info = planner_ctrl.get_observation_cmd()
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

            obs = [0.0] * 78 + obs_list
            obs = np.array(obs, dtype=np.float32).reshape(1, -1)

            # Override Head Pitch Angle in Observation
            obs[0, 81] = 0.0  # Head Yaw Angle
            obs[0, 82] = 0.0  # Head Pitch joint position

            # --- Policy Inference ---
            emitted_action = np.asarray(policy.predict_action(obs)).flatten()
            emitted_action = np.clip(emitted_action, -1.0, 1.0)
            action = emitted_action

            # target dof pos
            target_dof_pos = action[:num_qj] + default_angles[:num_qj]

            # head override
            target_dof_pos[0] = 0.0
            target_dof_pos[1] = 0.0 # 1.0
            # force the arms
            target_dof_pos[3] = -1.2
            target_dof_pos[7] = 1.2
            target_dof_pos = np.clip(target_dof_pos, min_angles, max_angles)
            
            viewer.sync()
            
            time_until_next = m.opt.timestep * control_decimation - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)

if __name__ == "__main__":
    main()