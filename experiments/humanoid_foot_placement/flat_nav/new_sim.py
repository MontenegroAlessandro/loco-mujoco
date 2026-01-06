# LIBs and SETUP
# libraries
import time, os, sys, mujoco, mujoco.viewer, numpy as np, yaml, hydra, jax, jax.numpy as jnp, argparse
from scipy.spatial.transform import Rotation as np_R
from loco_mujoco.algorithms import PPOJax

# Add parent directory to import path to find lmj and other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['JAX_PLATFORMS'] = "cpu"
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'

# =========================== NAVIGATION CONTROLLER ===========================
class NavigationController:
    def __init__(self, goal_pos, dt, kp=0.1, kd=0.1, stop_threshold=0.3, max_step_len=0.3, max_step_diff=0.1, align_threshold=0.5):
        self.goal_pos = np.array(goal_pos[:2]) # XY only
        self.dt = dt
        self.kp = kp
        self.kd = kd

        self.stop_threshold = stop_threshold
        self.align_threshold = align_threshold
        self.max_step_len = max_step_len
        self.max_step_diff = max_step_diff

        self.last_vert_dist = 0.0
        self.prev_yaw_err = 0.0
        self.finished = False

    def get_command(self, robot_pos, robot_quat):
        """
        Calculate steering angle and step length based on goal direction.
        """
        # Check if goal is reached
        curr_xy = robot_pos[:2]
        dist_to_goal = np.linalg.norm(self.goal_pos - curr_xy)

        if dist_to_goal < self.stop_threshold:
            self.finished = True
            return 0.0, 0.0, "STILL"

        # Calculate desired heading (Yaw)
        # Vector from robot to goal
        error_vec = self.goal_pos - curr_xy
        desired_yaw = np.arctan2(error_vec[1], error_vec[0])
        
        # Get current robot yaw
        curr_rot = np_R.from_quat([robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]]) # scalar last
        curr_yaw = curr_rot.as_euler('xyz')[2]

        # Calculate Yaw Error (wrapped to [-pi, pi])
        yaw_err = desired_yaw - curr_yaw
        yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi

        # derivative control part
        d_err = (yaw_err - self.prev_yaw_err) / self.dt
        self.prev_yaw_err = yaw_err

        # Control Logic
        # Steering: Turn towards the goal
        # We clip it to avoid crazy spins, but the GaitGenerator handles clipping too.
        steering_cmd = np.clip((self.kp * yaw_err) + (self.kd * d_err), -np.pi/4, np.pi/4) 

        # Velocity: Slow down if we need to turn sharply
        if abs(yaw_err) < self.align_threshold:
            # We are pointing roughly at the goal -> Full speed
            vert_dist_cmd = self.max_step_len
        else:
            # We are not aligned -> Turn in place (or very slow steps)
            vert_dist_cmd = 0.0

        if dist_to_goal < (self.max_step_len * 2):
            vert_dist_cmd = min(vert_dist_cmd, dist_to_goal)

        return vert_dist_cmd, steering_cmd, "FWD"

# =========================== LMJ Policy ===========================
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

def quat_rotate_inverse(q, v):
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

class GaitGenerator:
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
        # We only need vertical (FWD) and STILL for this experiment
        if mov_dir == "STILL":
            cmd = self._gen_still_cmd(reset=reset, gp=gp)
        elif mov_dir in ["FWD", "BWD"]:
            cmd = self._gen_vertical_cmd(gp=gp)
        else:
            # Fallback
            cmd = self._gen_still_cmd(reset=reset, gp=gp)
        
        return cmd
    
    def _gen_still_cmd(self, reset: bool = False, gp: float = 0.0):
        swing_foot_idx = 0 if (gp < 0.5) else 1
        zero_orn_offset = np.array([1, 0, 0, 0], dtype=np.float32)
        zero_pos_offset = np.zeros(3, dtype=np.float32)
        
        if self.gaits_to_still > 0:
            gait_info = np.array([np.cos(2 * np.pi * gp), np.sin(2 * np.pi * gp)])
            l_pos_offset = np.array([0, self.feet_distance, 0.0]) if swing_foot_idx == 0 else zero_pos_offset
            r_pos_offset = np.array([0, -self.feet_distance, 0.0]) if swing_foot_idx == 1 else zero_pos_offset
            l_orn_offset = zero_orn_offset
            r_orn_offset = zero_orn_offset
        else:
            gait_info = np.zeros(2, dtype=np.float32)
            l_pos_offset = np.array([0, self.feet_distance, 0.0])
            r_pos_offset = np.array([0, -self.feet_distance, 0.0])
            l_orn_offset = zero_orn_offset
            r_orn_offset = zero_orn_offset
        
        return l_pos_offset, l_orn_offset, r_pos_offset, r_orn_offset, gait_info
    
    def _gen_vertical_cmd(self, gp: float = 0.0, direction: int = 1):
        swing_foot_idx = 0 if (gp < 0.5) else 1
        zero_orn_offset = np.array([1, 0, 0, 0], dtype=np.float32)
        zero_pos_offset = np.zeros(3, dtype=np.float32)
        
        # adjust the steering angle
        steering_angle = np.clip(self.steering_angle, -np.pi, np.pi)
        steering_foot_idx = 0 if (steering_angle >= 0 and steering_angle <= np.pi) else 1
        steering_orn_offset = np_R.from_euler("z", steering_angle).as_quat(scalar_first=True)
        
        gait_info = np.array([np.cos(2 * np.pi * gp), np.sin(2 * np.pi * gp)])
        # pos gen uses self.vertical_dist which is updated by the NavController
        l_pos_offset = np.array([self.vertical_dist, self.feet_distance, 0]) if swing_foot_idx == 0 else zero_pos_offset
        r_pos_offset = np.array([self.vertical_dist, -self.feet_distance, 0]) if swing_foot_idx == 1 else zero_pos_offset
        # orn gen
        l_orn_offset = steering_orn_offset if steering_foot_idx == 0 else zero_orn_offset
        r_orn_offset = steering_orn_offset if steering_foot_idx == 1 else zero_orn_offset

        return l_pos_offset, l_orn_offset, r_pos_offset, r_orn_offset, gait_info

# ======================================================= MAIN =======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        default="fp_config.yaml", 
        help="Path to the deployment configuration file."
    )
    args = parser.parse_args()

    # Load config 
    print(f"Loading configuration from {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

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

    num_qj = len(default_angles) 
    base_num_actions = config["num_actions"] 

    experiment = config["experiment"]
    goal_coordinates = experiment["goal_coordinates"]
    max_step_len = experiment["max_step_len"]

    # Load Policy 
    policy = LMJPolicy(policy_path=agent_path)
    num_actions = base_num_actions
    
    num_obs = 3 + num_qj + 3 + num_qj + num_actions + 6
    
    print(f"Policy expects an observation size of: {num_obs}")
    print("Warming up the policy network for JIT compilation...")
    total_obs = 169
    for _ in range(500): 
        dummy_obs = jnp.zeros((1, total_obs), dtype=np.float32)
        _ = policy.predict_action(dummy_obs)
    print("Warmup complete.")

    # Load robot model
    spec = mujoco.MjSpec.from_file(xml_path)
    wb = spec.worldbody
        
    # Visualization sites
    wb.add_site(
        name=f"goal",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(0.2, 0.65, 0.0), 
        pos=(goal_coordinates[0], goal_coordinates[1], 0.65), 
        quat=(0.0, 0.0, 0.0, 1.0),
        group=0,
        rgba=(0.0, 1.0, 0.0, 0.5) # Green for GOAL
    )

    wb.add_site(
        name=f"foot_0",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.1, 0.04, 0.01),       
        pos=(0.1, 0.0, 0.0),             
        quat=(0, 0, 0, 1),     
        group=0,
        rgba=(0.0, 1.0, 1.0, 0.9),
    )
    
    wb.add_site(
        name=f"foot_1",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.1, 0.04, 0.01),      
        pos=(0.1, 0.0, 0.0),            
        quat=(0, 0, 0, 1),     
        group=0,
        rgba=(1.0, 0.55, 0.0, 0.9),
    )

    # get model spec
    for geom in spec.geoms:
        if geom.name.endswith("_col"):
            geom.delete()

    m = spec.compile()
    d = mujoco.MjData(m)
    left_foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    right_foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    # Initialize Simulation
    m.opt.timestep = simulation_dt

    try:
        initial_qpos = np.array(config['experiment']['init_state_params']['qpos_init'], dtype=np.float32)
        initial_qvel = np.array(config['experiment']['init_state_params']['qvel_init'], dtype=np.float32)
        if len(initial_qpos) == m.nq and len(initial_qvel) == m.nv:
            d.qpos[:] = initial_qpos
            d.qvel[:] = initial_qvel
        else:
            print("Warning: Initial state mismatch, using default.")
    except (AttributeError, KeyError) as e:
        print(f"Warning: Could not load initial state ({e}).")
    
    mujoco.mj_forward(m, d)

    # Initialize context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    target_dof_kps = kps.copy()
    target_dof_kds = kds.copy()
    
    cmd = np.zeros(16, dtype=np.float32) 
    counter = 1
    gait_frequency = experiment["gait_frequency"]
    feet_dist = experiment["feet_distance"]
    swing_foot_idx = 0 if ((counter * simulation_dt * gait_frequency) % 1.0 < 0.5) else 1
    sample_goal = True
    last_cmd = ""

    ctrl_dt = simulation_dt * control_decimation
    
    # Init Gait Generator and Navigation Controller
    GG = GaitGenerator(feet_distance=feet_dist, vertical_dist=0.0, lateral_dist=0.0, steering_angle=0.0)
    NavCtrl = NavigationController(
        goal_pos=goal_coordinates, 
        stop_threshold=0.2, 
        max_step_len=max_step_len,
        dt=ctrl_dt,
        kp=1.0, # 1.5,
        kd=0.0, # 0.1
    )

    print("\n" + "="*60)
    print(f" EXPERIMENT STARTED: Navigating to {goal_coordinates}")
    print("="*60 + "\n")

    # --- Start Simulation and Viewer ---
    # NOTE: No key_callback needed anymore
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start_time = time.time()
        while viewer.is_running() and (time.time() - start_time) < simulation_duration:
            step_start = time.time()

            tau = pd_control(target_dof_pos, d.qpos[7:], target_dof_kps, np.zeros_like(kds), d.qvel[6:], target_dof_kds)
            d.ctrl[:] = tau
            
            mujoco.mj_step(m, d)

            if counter % control_decimation == 0:
                # --- Prepare Observations ---
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7] 
                base_ang_vel = d.qvel[3:6]
                projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))

                # --- NAVIGATION UPDATE ---
                # Update the GaitGenerator based on current robot state and goal
                robot_pos = d.qpos[:3]
                vert_dist, steer_angle, mov_dir = NavCtrl.get_command(robot_pos, quat)
                
                # Apply commands to GG
                GG.vertical_dist = float(vert_dist)
                GG.steering_angle = float(steer_angle)

                # --- Create Command Vector `cmd` ---
                gait_process = (counter * simulation_dt * gait_frequency) % 1.0
                
                # SET GOALS (Swing foot logic)
                if swing_foot_idx == 0 and (gait_process >= 0.5 and gait_process < 1):
                    swing_foot_idx = 1
                    sample_goal = True
                    GG.gaits_to_still = np.maximum(GG.gaits_to_still - 1, 0)
                elif swing_foot_idx == 1 and (gait_process < 0.5 and gait_process >= 0):
                    swing_foot_idx = 0
                    sample_goal = True
                    GG.gaits_to_still = np.maximum(GG.gaits_to_still - 1, 0)
                
                if mov_dir == "STILL" and last_cmd != mov_dir:
                    GG.gaits_to_still = 2 # Start stop sequence if reached goal
                    last_cmd = mov_dir
                    
                l_offset, l_orn_offset, r_offset, r_orn_offset, gait_info = GG.query_cmd(
                    mov_dir=mov_dir, reset=False, gp=gait_process
                ) 
                
                if sample_goal:
                    sample_goal = False
                    if swing_foot_idx == 0:
                        rot_stance = np_R.from_matrix(d.site("right_foot").xmat.reshape(3, 3))
                        stance_yaw = rot_stance.as_euler("xyz")[2]
                        rot_stance_flat = np_R.from_euler("z", stance_yaw)
                        cmd_quat = np.array([l_orn_offset[1], l_orn_offset[2], l_orn_offset[3], l_orn_offset[0]])
                        rot_cmd = np_R.from_quat(cmd_quat)
                        target_rot = rot_stance_flat * rot_cmd
                        m.site("foot_1").pos = d.site_xpos[right_foot_id] + rot_stance_flat.apply(l_offset)
                        m.site("foot_1").pos[2] = 0
                        m.site("foot_1").quat = target_rot.as_quat(scalar_first=True)
                    else:
                        rot_stance = np_R.from_matrix(d.site("left_foot").xmat.reshape(3, 3))
                        stance_yaw = rot_stance.as_euler("xyz")[2]
                        rot_stance_flat = np_R.from_euler("z", stance_yaw)
                        cmd_quat = np.array([r_orn_offset[1], r_orn_offset[2], r_orn_offset[3], r_orn_offset[0]])
                        rot_cmd = np_R.from_quat(cmd_quat)
                        target_rot = rot_stance_flat * rot_cmd
                        m.site("foot_0").pos = d.site_xpos[left_foot_id] + rot_stance_flat.apply(r_offset) 
                        m.site("foot_0").pos[2] = 0                 
                        m.site("foot_0").quat = target_rot.as_quat(scalar_first=True)
                
                cmd = np.concatenate(
                    [l_offset, l_orn_offset, r_offset, r_orn_offset, gait_info], dtype=np.float32
                )

                # --- Construct Observation Vector ---
                obs_list = []
                obs_list += projected_gravity.flatten().tolist()
                obs_list += qj.flatten().tolist()
                obs_list += (base_ang_vel * 1.0).flatten().tolist()
                obs_list += (dqj * 0.1).flatten().tolist()
                obs_list += action.flatten().tolist()
                obs_list += cmd.flatten().tolist()

                obs = [0.] * (78) + obs_list
                obs = np.array(obs, dtype=np.float32).reshape(1, -1)

                # --- Policy Inference ---
                emitted_action = np.asarray(policy.predict_action(obs)).flatten()
                emitted_action = np.clip(emitted_action, -1.0, 1.0)
                action = action * 0.0 + emitted_action * 1.0

                target_dof_pos = action[:num_qj] + default_angles[:num_qj] 
                target_dof_pos = np.clip(target_dof_pos, min_angles, max_angles)
                target_dof_kps = kps.copy()
                target_dof_kds = kds.copy()

            counter += 1
            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)