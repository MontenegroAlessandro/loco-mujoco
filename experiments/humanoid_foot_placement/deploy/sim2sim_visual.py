from shutil import move
import time
import os
import sys

# Add parent directory to import path to find lmj and other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

import mujoco.viewer
import mujoco
import numpy as np
import yaml
import hydra
from omegaconf import DictConfig  # Using omegaconf.DictConfig for hydra config type
import pickle  # Still needed for PPOJax.load_agent
import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation as np_R
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast
import copy
import cv2

from loco_mujoco.algorithms import PPOJax

from gait_generators import GaitGenerator, VisualGaitGenerator


class LMJPolicy:
    def __init__(self, policy_path: str) -> None:  # Removed control_func_path
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
        self._jit_sample_action = jax.jit(self._sample_action, static_argnames=["network_apply"])
        print("Policy loaded and JIT function compiled.")

    @staticmethod
    def _sample_action(network_apply, params, run_stats, rng, obs):  # MODIFIED: Removed batch_stats
        """This function is JIT-compiled for speed."""
        # MODIFIED: Removed batch_stats from the dictionary and mutable list
        y, updates = network_apply({"params": params, "run_stats": run_stats}, obs, mutable=["run_stats"])
        pi, _ = y
        a = pi.mode()  # Get the deterministic action
        a = jnp.atleast_2d(a)
        return a

    def predict_action(self, obs):
        """Uses the precompiled JIT function to get the action."""
        # MODIFIED: Removed self.train_state.batch_stats from the call
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
    """Calculates PD control torques."""
    return (target_q - q) * kp + (target_dq - dq) * kd


@hydra.main(config_name="config_sim2sim_visual.yaml")
def main(config: DictConfig):

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

    num_qj = len(default_angles)  # Number of actuated joints (23)
    base_num_actions = config["num_actions"]  # This is also 23

    cmd_params = config["command"]

    # --- Load Policy ---
    # Initialize Hydra to access environment config used during training
    # The config_path should point to the directory containing your hydra config files
    # hydra.initialize(config_path="./") # Adjust path if your hydra config is elsewhere
    lmj_hydra_config = hydra.compose(config_name="conf_t1")
    policy = LMJPolicy(policy_path=agent_path)

    # Determine actual num_actions and observation size based on policy's environment config
    num_actions = base_num_actions

    # Calculate the total observation size for policy warmup and runtime
    # obs = [projected_gravity (3), qj (num_qj), base_ang_vel (3), dqj (num_qj), action (num_actions), cmd (6)]
    num_obs = 3 + num_qj + 3 + num_qj + num_actions + 6

    print(f"Policy expects an observation size of: {num_obs}")
    print("Warming up the policy network for JIT compilation...")
    total_obs = max(policy.agent_conf.network.actor_obs_ind.max(), policy.agent_conf.network.critic_obs_ind.max()) + 1
    for _ in range(500):  # Reduced warmup steps from 1000 to 500
        dummy_obs = jnp.zeros((1, total_obs), dtype=np.float32)
        _ = policy.predict_action(dummy_obs)
    print("Warmup complete.")

    # Load robot model
    spec = mujoco.MjSpec.from_file(xml_path)
    wb = spec.worldbody

    # ==================================================OBSTACLEs==================================================
    # this part is only needed for initialization, then the boxes will be added when doing the reset
    wb.add_site(
        name=f"foot_0",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.1, 0.04, 0.001),  # *
        pos=(0.1, 0.0, 0.0),  # **
        quat=(0, 0, 0, 1),
        group=1,
        rgba=(1.0, 0.5, 0.0, 0.5),
    )

    wb.add_site(
        name=f"foot_1",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.1, 0.04, 0.001),  # *
        pos=(0.1, 0.0, 0.0),  # **
        quat=(0, 0, 0, 1),
        group=1,
        rgba=(1.0, 1.0, 0.0, 0.5),
    )

    # Add visual sites as foot targets
    target_dist = 0.25
    target_angle_range = 45  # degrees
    target_site_pos = np.zeros(3)
    angle = 0
    site_idx = 0
    for i in range(10):
        if i > 5:
            angle += np.random.uniform(-target_angle_range, target_angle_range)
        target_site_pos += target_dist * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0.0])
        feet_pos = target_site_pos + cmd_params["feet_distance"] / 2 * np.array(
            [np.cos(np.deg2rad(angle + (-1) ** i * 90)), np.sin(np.deg2rad(angle + (-1) ** i * 90)), 0.0]
        )
        wb.add_site(
            name=f"target_{i}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=(0.08, 0.005, 0.0),  # *
            pos=feet_pos,
            quat=(0, 0, 0, 1),
            group=0,
            rgba=(0.0, 0.0, 1.0, 1.0),
        )

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
    m.opt.timestep = simulation_dt

    cam_width = 320
    cam_height = 200
    rgb_renderer = mujoco.Renderer(m, width=cam_width, height=cam_height)
    depth_renderer = mujoco.Renderer(m, width=cam_width, height=cam_height)
    depth_renderer.enable_depth_rendering()
    rgb_renderer._scene_option.sitegroup[1] = 0
    depth_renderer._scene_option.sitegroup[1] = 0

    rgb_viewport = mujoco.MjrRect(920, 0, cam_width, cam_height)
    depth_viewport = mujoco.MjrRect(1240, 0, cam_width, cam_height)
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
            print(
                f"Warning: Initial qpos/qvel length mismatch with model. "
                f"Config qpos: {len(initial_qpos)} (expected {m.nq}), "
                f"qvel: {len(initial_qvel)} (expected {m.nv}). Using default MuJoCo init."
            )
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
    counter = 1
    gait_frequency = cmd_params["gait_frequency"]

    # init the gait generator
    # GG = GaitGenerator(feet_distance=feet_dist, vertical_dist=0.0, lateral_dist=0.0, steering_angle=0.0)
    GG = VisualGaitGenerator(
        robot_model=m, robot_data=d, cam_width=cam_width, cam_height=cam_height,
        feet_distance=cmd_params["feet_distance"], stop_steps=cmd_params["stop_steps"])
    GG.print_instruction()

    # ===========================================TELEOPERATION via KEYBOARD===========================================
    # --- Start Simulation and Viewer ---
    with mujoco.viewer.launch_passive(m, d, key_callback=GG.key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            for i in range(control_decimation):
                counter += 1
                # Step the simulation forward. The PD controller runs at the physics rate.
                tau = pd_control(
                    target_dof_pos, d.qpos[7:], target_dof_kps, np.zeros_like(kds), d.qvel[6:], target_dof_kds
                )
                d.ctrl[:] = tau

                mujoco.mj_step(m, d)

            # get rgb and depth images from head camera
            rgb_renderer.update_scene(d, camera="head_camera")
            depth_renderer.update_scene(d, camera="head_camera")
            rgb_array = rgb_renderer.render()
            depth_array = depth_renderer.render()
            depth_rgb = np.clip((depth_array - 0.2) / 1.8 * 255.0, 0, 255)
            depth_rgb = depth_rgb[:, :, None].repeat(3, axis=2).astype(np.uint8)

            # --- Prepare Observations ---
            qj = d.qpos[7:]
            dqj = d.qvel[6:]
            quat = d.qpos[3:7]  # Pelvis orientation [w, x, y, z] from free joint
            # base_ang_vel = d.sensor("angular-velocity").data.astype(np.float32) # Assuming sensor name "angular-velocity"
            base_ang_vel = d.qvel[3:6]
            projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))

            # --- Create Command Vector `cmd` ---
            gait_process = (counter * simulation_dt * gait_frequency) % 1.0

            # foot_offset, gait_info = GG.query_cmd(gp=gait_process)
            foot_offset, gait_info, rgb_array = GG.query_cmd(
                rgb_image=rgb_array, depth_image=depth_array, joint_pos=qj, gp=gait_process)
            l_offset, l_orn_offset, r_offset, r_orn_offset = foot_offset

            if GG.sample_goal:
                if GG.swing_foot_idx == 0:
                    # left foot swing
                    # get stance foot rotation
                    rot_stance = np_R.from_matrix(d.site("right_foot").xmat.reshape(3, 3))
                    stance_yaw = rot_stance.as_euler("xyz")[2]
                    rot_stance_flat = np_R.from_euler("z", stance_yaw)
                    # compute the quat
                    cmd_quat = np.array([l_orn_offset[1], l_orn_offset[2], l_orn_offset[3], l_orn_offset[0]])
                    rot_cmd = np_R.from_quat(cmd_quat)
                    # compute target rot
                    target_rot = rot_stance_flat * rot_cmd
                    # update site
                    m.site("foot_1").pos = d.site_xpos[right_foot_id] + rot_stance_flat.apply(l_offset)
                    m.site("foot_1").pos[2] = 0
                    m.site("foot_1").quat = target_rot.as_quat(scalar_first=True)
                else:
                    # right foot swing
                    # get stance foot rotation
                    rot_stance = np_R.from_matrix(d.site("left_foot").xmat.reshape(3, 3))
                    stance_yaw = rot_stance.as_euler("xyz")[2]
                    rot_stance_flat = np_R.from_euler("z", stance_yaw)
                    # compute the quat
                    cmd_quat = np.array([r_orn_offset[1], r_orn_offset[2], r_orn_offset[3], r_orn_offset[0]])
                    rot_cmd = np_R.from_quat(cmd_quat)
                    # compute the quat
                    target_rot = rot_stance_flat * rot_cmd
                    # update site
                    m.site("foot_0").pos = d.site_xpos[left_foot_id] + rot_stance_flat.apply(r_offset)
                    m.site("foot_0").pos[2] = 0
                    m.site("foot_0").quat = target_rot.as_quat(scalar_first=True)

                # move the visual targets
                target_has_passed = False
                target_pos = m.site("target_{}".format(site_idx)).pos
                robot_pos = d.qpos[:3]
                robot_orn = np_R.from_quat(d.qpos[3:7], scalar_first=True).as_matrix()
                target_pos_in_robot = robot_orn.T @ (target_pos - robot_pos)
                if target_pos_in_robot[0] < -0.05:
                    target_has_passed = True

                if target_has_passed:
                    angle += np.random.uniform(-target_angle_range, target_angle_range)
                    target_site_pos += target_dist * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0.0])
                    feet_pos = target_site_pos + cmd_params["feet_distance"] / 2 * np.array(
                        [
                            np.cos(np.deg2rad(angle + (-1) ** site_idx * 90)),
                            np.sin(np.deg2rad(angle + (-1) ** site_idx * 90)),
                            0.0,
                        ]
                    )
                    m.site("target_{}".format(site_idx)).pos = feet_pos
                    site_idx = (site_idx + 1) % 10

                mujoco.mj_fwdPosition(m, d)

            cmd = np.concatenate([l_offset, l_orn_offset, r_offset, r_orn_offset, gait_info], dtype=np.float32)

            # --- Construct Observation Vector ---
            obs_list = []
            obs_list += projected_gravity.flatten().tolist()
            obs_list += qj.flatten().tolist()
            obs_list += (base_ang_vel * 1.0).flatten().tolist()
            obs_list += (dqj * 0.1).flatten().tolist()
            obs_list += action.flatten().tolist()  # 'action' here is the previous action
            obs_list += cmd.flatten().tolist()

            obs = [0.0] * (78) + obs_list

            obs = np.array(obs, dtype=np.float32).reshape(1, -1)

            # Override Head Pitch Angle in Observation
            obs[0, 81] = 0.0  # Head Yaw Angle
            obs[0, 82] = 0.0  # Head Pitch joint position

            # --- Policy Inference ---
            emitted_action = np.asarray(policy.predict_action(obs)).flatten()

            emitted_action = np.clip(emitted_action, -1.0, 1.0)

            # Apply smoothing/filtering to the action
            action = action * 0.0 + emitted_action * 1.0

            # Deconstruct action vector into control commands
            target_dof_pos = (
                action[:num_qj] + default_angles[:num_qj]
            )  # Use num_qj here as it's the base action for positions

            # Override head joint for control
            target_dof_pos[0] = 0.0
            target_dof_pos[1] = 1.0

            # Clip target_dof_pos to joint limits
            target_dof_pos = np.clip(target_dof_pos, min_angles, max_angles)

            target_dof_kps = kps.copy()
            target_dof_kds = kds.copy()

            # Sync viewer and maintain real-time simulation speed
            viewer.sync()
            viewer.set_images([(rgb_viewport, rgb_array), (depth_viewport, depth_rgb)])

            time_until_next_step = m.opt.timestep * control_decimation - (time.time() - step_start)
            # time.sleep(0.1)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
