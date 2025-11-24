import time
import os
import sys

# Add parent directory to import path to find lmj and other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ['JAX_PLATFORMS'] = "cpu"
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'

import mujoco.viewer
import mujoco
import numpy as np
import yaml
import hydra
from omegaconf import DictConfig # Using omegaconf.DictConfig for hydra config type
import pickle # Still needed for PPOJax.load_agent
import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation as np_R
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast

from loco_mujoco.algorithms import PPOJax

class LMJPolicy:
    def __init__(self, policy_path: str) -> None: # Removed control_func_path
        # Load agent configuration and state from the policy checkpoint
        agent_conf, agent_state = PPOJax.load_agent(policy_path)

        # Removed: self.control_func = pickle.load(fileobj) as it's no longer used

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
        self._jit_sample_action = jax.jit(
            self._sample_action, static_argnames=["network_apply"]
        )
        print("Policy loaded and JIT function compiled.")

    @staticmethod
    def _sample_action(network_apply, params, run_stats, rng, obs): # MODIFIED: Removed batch_stats
        """This function is JIT-compiled for speed."""
        # MODIFIED: Removed batch_stats from the dictionary and mutable list
        y, updates = network_apply(
            {'params': params, 'run_stats': run_stats},
            obs,
            mutable=["run_stats"]
        )
        pi, _ = y
        a = pi.mode()  # Get the deterministic action
        a = jnp.atleast_2d(a)
        return a

    def predict_action(self, obs):
        """Uses the precompiled JIT function to get the action."""
        # MODIFIED: Removed self.train_state.batch_stats from the call
        a = self._jit_sample_action(
            self.network_apply,
            self.train_state.params,
            self.train_state.run_stats,
            self._rng,
            obs
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
    """Calculates PD control torques."""
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("path", type=str, help="Path to the policy checkpoint folder.")
    parser.add_argument("--config", type=str, default="deploy_mujoco_config_h1_dfki.yaml",
                        help="Path to the deployment configuration file.")
    args = parser.parse_args()

    # --- Load Configuration ---
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

    num_qj = len(default_angles) # Number of actuated joints (23)
    base_num_actions = config["num_actions"] # This is also 23

    cmd_params = config["command"]

    # --- Load Policy ---
    # Initialize Hydra to access environment config used during training
    # The config_path should point to the directory containing your hydra config files
    # hydra.initialize(config_path="./") # Adjust path if your hydra config is elsewhere
    hydra.initialize(config_path="../train/")
    # lmj_hydra_config = hydra.compose(config_name="conf_t1") # Use the appropriate config name
    lmj_hydra_config = hydra.compose(config_name="conf")

    # policy = LMJPolicy(policy_path=args.path) # Removed control_func_path
    policy = LMJPolicy(policy_path=agent_path)

    # Determine actual num_actions and observation size based on policy's environment config
    num_actions = base_num_actions
    # if policy.agent_conf.config.experiment.env_params.control_params.varstiff:
    #     num_actions *= 2
    #     print(f"Variable stiffness detected. Action space size extended to {num_actions}.")
    
    # Calculate the total observation size for policy warmup and runtime
    # obs = [projected_gravity (3), qj (num_qj), base_ang_vel (3), dqj (num_qj), action (num_actions), cmd (6)]
    num_obs = 3 + num_qj + 3 + num_qj + num_actions + 6
    
    print(f"Policy expects an observation size of: {num_obs}")
    print("Warming up the policy network for JIT compilation...")
    total_obs = 169
    for _ in range(500): # Reduced warmup steps from 1000 to 500
        dummy_obs = jnp.zeros((1, total_obs), dtype=np.float32)
        _ = policy.predict_action(dummy_obs)
    print("Warmup complete.")

    # Load robot model
    spec = mujoco.MjSpec.from_file(xml_path)

    # m = mujoco.MjModel.from_xml_path(xml_path)

    # get model spec
    # delete all geoms whose names end in "_col" from spec
    for geom in spec.geoms:
        if geom.name.endswith("_col"):
            geom.delete()

    m = spec.compile()
    d = mujoco.MjData(m)
    left_foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    right_foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    # # --- Initialize Simulation ---
    # m = mujoco.MjModel.from_xml_path(xml_path)
    # d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Set initial robot state from policy's environment config
    try:
        initial_qpos = np.array(lmj_hydra_config.experiment.env_params.init_state_params.qpos_init, dtype=np.float32)
        initial_qvel = np.array(lmj_hydra_config.experiment.env_params.init_state_params.qvel_init, dtype=np.float32)
        if len(initial_qpos) == m.nq and len(initial_qvel) == m.nv:
            d.qpos[:] = initial_qpos
            d.qvel[:] = initial_qvel
            print(initial_qpos)
            print("Initial state (qpos, qvel) loaded from policy's environment config.")
        else:
            print(f"Warning: Initial qpos/qvel length mismatch with model. "
                  f"Config qpos: {len(initial_qpos)} (expected {m.nq}), "
                  f"qvel: {len(initial_qvel)} (expected {m.nv}). Using default MuJoCo init.")
    except (AttributeError, KeyError) as e:
        print(f"Warning: Could not load initial state from policy config ({e}). Using default MuJoCo initialization.")
    
    # Update physics state after setting qpos/qvel (important for correct sensor readings etc.)
    mujoco.mj_forward(m, d)

    # Initialize context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    target_dof_kps = kps.copy()
    target_dof_kds = kds.copy()
    
    # command parameters
    cmd = np.zeros(16, dtype=np.float32) 
    counter = 0
    gait_frequency = cmd_params["gait_frequency"]
    des_dist = cmd_params["distance"]
    # directions
    fwd_pos = np.array([[des_dist, des_dist, 0.0], [des_dist, -des_dist, 0.0]]) # left, right
    bwd_pos = np.array([[-des_dist, des_dist, 0.0], [-des_dist, -des_dist, 0.0]]) # left, right
    hold_pos = np.array([[0.0, des_dist, 0.0], [0.0, -des_dist, 0.0]]) # left, right
    pos_array = [hold_pos, fwd_pos, hold_pos, bwd_pos]
    # goals
    swing_foot_idx = 0
    # gait_swithces
    num_gaits = 0
    idx = 0
    max_gaits = 20

    # --- Start Simulation and Viewer ---
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start_time = time.time()
        while viewer.is_running() and (time.time() - start_time) < simulation_duration:
            step_start = time.time()

            # Step the simulation forward. The PD controller runs at the physics rate.
            tau = pd_control(target_dof_pos, d.qpos[7:], target_dof_kps, np.zeros_like(kds), d.qvel[6:], target_dof_kds)
            d.ctrl[:] = tau
            
            mujoco.mj_step(m, d)
            # counter += 1

            # Run the policy at the defined control frequency
            if counter % control_decimation == 0:
                # --- Prepare Observations ---
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7] # Pelvis orientation [w, x, y, z] from free joint
                # base_ang_vel = d.sensor("angular-velocity").data.astype(np.float32) # Assuming sensor name "angular-velocity"
                base_ang_vel = d.qvel[3:6]
                projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))

                # --- Create Command Vector `cmd` ---
                gait_process = (counter * simulation_dt * gait_frequency) % 1.0
                
                swing_pos_offset = pos_array[idx][swing_foot_idx]
                stance_pos_offset = np.zeros(3, np.float32)
                swing_orn_offset = np_R.from_euler('z', np.deg2rad(0)).as_quat(scalar_first=True)
                stance_orn_offset = np_R.from_euler('z', 0).as_quat(scalar_first=True)
                
                # SET GOALS
                if swing_foot_idx == 0 and (gait_process >= 0.5 and gait_process < 1):
                    swing_foot_idx = 1
                    num_gaits += 1
                elif swing_foot_idx == 1 and (gait_process < 0.5 and gait_process >= 0):
                    swing_foot_idx = 0
                    num_gaits += 1
                # switch walking scheme when needed
                if num_gaits % max_gaits == 0:
                    idx = (idx + 1) % len(pos_array)

                # MANAGE OBS
                if swing_foot_idx == 0:
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

                # --- Construct Observation Vector ---
                obs_list = []
                obs_list += projected_gravity.flatten().tolist()
                obs_list += qj.flatten().tolist()
                obs_list += (base_ang_vel * 1.0).flatten().tolist()
                obs_list += (dqj * 0.1).flatten().tolist()
                obs_list += action.flatten().tolist() # 'action' here is the previous action
                obs_list += cmd.flatten().tolist()

                obs = [0.] * (78) + obs_list

                obs = np.array(obs, dtype=np.float32).reshape(1, -1)

                # --- Policy Inference ---
                emitted_action = np.asarray(policy.predict_action(obs)).flatten()

                emitted_action = np.clip(emitted_action, -1.0, 1.0)

                # Apply smoothing/filtering to the action
                action = action * 0.0 + emitted_action * 1.0

                # Deconstruct action vector into control commands
                target_dof_pos = action[:num_qj] + default_angles[:num_qj] # Use num_qj here as it's the base action for positions

                # Clip target_dof_pos to joint limits
                target_dof_pos = np.clip(target_dof_pos, min_angles, max_angles)
                # FIXME: FIX HEAD AND SHOULDERS
                target_dof_pos[0] = 0
                target_dof_pos[1] = 0
                target_dof_pos[3] = -1.2
                target_dof_pos[7] = 1.2

                target_dof_kps = kps.copy()
                target_dof_kds = kds.copy()

            # counter update
            counter += 1

            # Sync viewer and maintain real-time simulation speed
            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)