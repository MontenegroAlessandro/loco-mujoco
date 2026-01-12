# LIBs and SETUP
# libraries
import time, os, sys, mujoco, mujoco.viewer, numpy as np, yaml, hydra, jax, jax.numpy as jnp, argparse
from scipy.spatial.transform import Rotation as np_R
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.environments.utils import add_box, add_stair, add_slope, add_ramp_platform_ramp
from track_utils import GaitGenerator, Checkpoint, pd_control, quat_rotate_inverse, CheckpointController
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast

# Add parent directory to import path to find lmj and other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['JAX_PLATFORMS'] = "cpu"
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'

# ==================================================== LMJ Policy ====================================================
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
        self._jit_sample_action = jax.jit(
            self._sample_action, static_argnames=["network_apply"]
        )
        print("Policy loaded and JIT function compiled.")

    @staticmethod
    def _sample_action(network_apply, params, run_stats, rng, obs):
        y, updates = network_apply(
            {'params': params, 'run_stats': run_stats},
            obs,
            mutable=["run_stats"]
        )
        pi, _ = y
        a = pi.mode()
        a = jnp.atleast_2d(a)
        return a

    def predict_action(self, obs):
        a = self._jit_sample_action(
            self.network_apply,
            self.train_state.params,
            self.train_state.run_stats,
            self._rng,
            obs
        )
        return a

# ======================================================== Main ====================================================== 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="fp_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Config loading
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
    
    base_num_actions = config["num_actions"] 
    experiment = config["experiment"]
    gait_frequency = experiment["gait_frequency"]
    feet_dist = experiment["feet_distance"]
    max_step_len = experiment["max_step_len"]

    # Checkpoints
    init_x = 2.0
    delta_x = 2.5
    delta_z = 0.3
    z_off = min(0.08, delta_z/delta_x)
    checkpoints = [
        # start (j) ramp
        Checkpoint(chk_pos=[init_x, 0.0, 0.0], next_pos=None, mov_mode="FWD", xy_max_offset=0.2, z_offset=0.0),
        # start plane
        Checkpoint(chk_pos=[init_x + delta_x, 0.0, delta_z], next_pos=None, mov_mode="FWD", xy_max_offset=0.2, z_offset=z_off),
        # end plane
        Checkpoint(chk_pos=[init_x + 2 * delta_x, 0.0, delta_z], next_pos=None, mov_mode="FWD", xy_max_offset=0.1, z_offset=0.0),
        # end ramp
        Checkpoint(chk_pos=[init_x + 3 * delta_x, 0.0, 0.0], next_pos=None, mov_mode="FWD", xy_max_offset=0.1, z_offset=0.0),
        # end end
        Checkpoint(chk_pos=[init_x + 4 * delta_x, 0.0, 0.0], next_pos=None, mov_mode="FWD", xy_max_offset=0.2, z_offset=0.0),
    ]

    initial_goal = checkpoints[0].chk_pos

    # Policy
    policy = LMJPolicy(policy_path=agent_path)
    total_obs = 169 
    for _ in range(10): 
        dummy_obs = np.zeros((1, total_obs), dtype=np.float32)
        _ = policy.predict_action(dummy_obs)

    # MuJoCo Setup
    spec = mujoco.MjSpec.from_file(xml_path)
    wb = spec.worldbody
    
    # Visualization Sites
    wb.add_site(name="goal", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=(0.2, 0.65, 0.0), 
                pos=(initial_goal[0], initial_goal[1], 0.65), rgba=(0.0, 1.0, 0.0, 0.5))
    wb.add_site(name="foot_0", type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.1, 0.04, 0.01),       
                pos=(0.1, 0.0, 0.0), rgba=(0.0, 1.0, 1.0, 0.9), quat=(0.0, 0.0, 0.0, 1.0))
    wb.add_site(name="foot_1", type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.1, 0.04, 0.01),      
                pos=(0.1, 0.0, 0.0), rgba=(1.0, 0.55, 0.0, 0.9), quat=(0.0, 0.0, 0.0, 1.0))

    # Add obstacles
    wb = add_ramp_platform_ramp(
        world_body = wb, 
        name = "Ramp-Platform-Ramp",
        coordinates = checkpoints[0].chk_pos, 
        run = delta_x,           
        rise = delta_z,          
        platform_length = delta_x, 
        platform_width = 1.0, 
        width = 1.0,
        thickness = 0.01,
        orientation_yaw_deg = 0.0
    )

    for geom in spec.geoms:
        if geom.name.endswith("_col"): geom.delete()

    m = spec.compile()
    d = mujoco.MjData(m)
    
    # IDs
    goal_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "goal")
    foot_0_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "foot_0")
    foot_1_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "foot_1")
    left_foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    right_foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    # Init State
    m.opt.timestep = simulation_dt
    try:
        d.qpos[:] = np.array(config['experiment']['init_state_params']['qpos_init'])
        d.qvel[:] = np.array(config['experiment']['init_state_params']['qvel_init'])
    except: pass
    
    mujoco.mj_forward(m, d)

    # Controller Init
    action = np.zeros(base_num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    counter = 1
    
    GG = GaitGenerator(feet_distance=feet_dist, stop_steps=2)
    ctrl_dt = simulation_dt * control_decimation
    ChkCtrl = CheckpointController(GG, checkpoints, dt=ctrl_dt)

    print(f" EXPERIMENT STARTED: {len(checkpoints)} Checkpoints queued.")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start_time = time.time()
        while viewer.is_running() and (time.time() - start_time) < simulation_duration:
            step_start = time.time()

            # Low Level Control
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            if counter % control_decimation == 0:
                # Observations
                qj, dqj = d.qpos[7:], d.qvel[6:]
                quat, base_ang_vel = d.qpos[3:7], d.qvel[3:6]
                robot_pos = d.qpos[:3]
                proj_grav = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
                gait_process = (counter * simulation_dt * gait_frequency) % 1.0
                
                # --- High Level Control ---
                # Checkpoint controller updates GG internally and returns command
                foot_offset, gait_info = ChkCtrl.get_command(robot_pos, quat, gait_process)
                l_offset, l_orn_offset, r_offset, r_orn_offset = foot_offset

                # --- Visuals ---
                # Update Goal visualization
                curr_goal = ChkCtrl.checkpoints[ChkCtrl.current_chk_idx].chk_pos
                m.site_pos[goal_site_id] = np.array(curr_goal) + np.array([0, 0, 0.65])

                # Update Footholds visualization (only if sampling new goal)
                if GG.sample_goal:
                    if GG.swing_foot_idx == 0: # Left Cmd, Stance Right
                        stance_pos = d.site_xpos[right_foot_id]
                        rot_stance = np_R.from_matrix(d.site_xmat[right_foot_id].reshape(3, 3))
                        stance_yaw_rot = np_R.from_euler("z", rot_stance.as_euler("xyz")[2])
                        
                        target_pos = stance_pos + stance_yaw_rot.apply(l_offset)
                        cmd_rot = stance_yaw_rot * np_R.from_quat(quat_scalarfirst2scalarlast(l_orn_offset))
                        
                        m.site_pos[foot_0_site_id] = [target_pos[0], target_pos[1], 0.01]
                        m.site_quat[foot_0_site_id] = cmd_rot.as_quat(scalar_first=True)
                    else: # Right Cmd, Stance Left
                        stance_pos = d.site_xpos[left_foot_id]
                        rot_stance = np_R.from_matrix(d.site_xmat[left_foot_id].reshape(3, 3))
                        stance_yaw_rot = np_R.from_euler("z", rot_stance.as_euler("xyz")[2])
                        
                        target_pos = stance_pos + stance_yaw_rot.apply(r_offset)
                        cmd_rot = stance_yaw_rot * np_R.from_quat(quat_scalarfirst2scalarlast(r_orn_offset))
                        
                        m.site_pos[foot_1_site_id] = [target_pos[0], target_pos[1], 0.01]
                        m.site_quat[foot_1_site_id] = cmd_rot.as_quat(scalar_first=True)

                # --- Inference ---
                cmd = np.concatenate([l_offset, l_orn_offset, r_offset, r_orn_offset, gait_info], dtype=np.float32)
                
                obs_data = []
                obs_data += proj_grav.flatten().tolist()
                obs_data += qj.flatten().tolist()
                obs_data += (base_ang_vel * 1.0).flatten().tolist()
                obs_data += (dqj * 0.1).flatten().tolist()
                obs_data += action.flatten().tolist()
                obs_data += cmd.flatten().tolist()
                
                # Padding
                obs = np.array([0.] * (total_obs - len(obs_data)) + obs_data, dtype=np.float32).reshape(1, -1)
                
                emitted_action = np.clip(policy.predict_action(obs).flatten(), -1.0, 1.0)
                action = emitted_action 
                target_dof_pos = np.clip(action * 1.0 + default_angles, min_angles, max_angles)

            counter += 1
            viewer.sync()
            
            t_rem = m.opt.timestep - (time.time() - step_start)
            if t_rem > 0: time.sleep(t_rem)