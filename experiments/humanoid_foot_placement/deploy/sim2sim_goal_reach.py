from shutil import move
import time
import os
import sys

# from experiments.humanoid_foot_placement.deploy.gait_generators import GoalReachingGaitGenerator, GaitGenerator
from gait_generators import GoalReachingGaitGenerator, GaitGenerator

# Add parent directory to import path to find lmj and other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

import mujoco.viewer
import mujoco
import numpy as np
import hydra
from omegaconf import DictConfig
import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation as np_R

from loco_mujoco.algorithms import PPOJax

# =========================
# GOAL MODE
# =========================
MODE_STOP = "stop"
MODE_RANDOM_NEXT = "random_next"

GOAL_MODE = MODE_RANDOM_NEXT


# =========================
# GOAL
# =========================
INIT_GOAL_POS = np.array([-1.5, -2.5, 0.0], dtype=np.float32)
REACH_THRESH = 0.25
REACH_HOLD_STEPS = 15

# ---- random goal sampling box (edit to your scene) ----
GOAL_X_RANGE = (-3.0, 3.0)
GOAL_Y_RANGE = (-4.0, -0.5)
GOAL_Z = 0.0

MIN_GOAL_DIST = 0.8



def sample_next_goal(cur_pos_xy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample a new goal in world frame."""
    for _ in range(100):  # avoid infinite loop
        gx = rng.uniform(GOAL_X_RANGE[0], GOAL_X_RANGE[1])
        gy = rng.uniform(GOAL_Y_RANGE[0], GOAL_Y_RANGE[1])
        g = np.array([gx, gy, GOAL_Z], dtype=np.float32)
        if np.linalg.norm((g - np.array([cur_pos_xy[0], cur_pos_xy[1], 0.0], dtype=np.float32))[:2]) >= MIN_GOAL_DIST:
            return g
    # fallback if too strict
    return np.array([gx, gy, GOAL_Z], dtype=np.float32)


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


@hydra.main(config_name="config_sim2sim_goal_reach.yaml")
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

    # Load robot model
    spec = mujoco.MjSpec.from_file(xml_path)
    wb = spec.worldbody

    wb.add_site(
        name="foot_0",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.1, 0.04, 0.001),
        pos=(0.1, 0.0, 0.0),
        quat=(0, 0, 0, 1),
        group=1,
        rgba=(1.0, 0.5, 0.0, 0.5),
    )

    wb.add_site(
        name="foot_1",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.1, 0.04, 0.001),
        pos=(0.1, 0.0, 0.0),
        quat=(0, 0, 0, 1),
        group=1,
        rgba=(1.0, 1.0, 0.0, 0.5),
    )

    wb.add_site(
        name="goal_site",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(0.035, 0.035, 0.035),
        pos=(float(INIT_GOAL_POS[0]), float(INIT_GOAL_POS[1]), 0.035),
        quat=(1, 0, 0, 0),
        group=2,
        rgba=(0.1, 1.0, 0.1, 0.9),
    )

    # delete all *_col
    for geom in spec.geoms:
        if geom.name.endswith("_col"):
            geom.delete()

    m = spec.compile()
    d = mujoco.MjData(m)

    left_foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    right_foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    m.opt.timestep = simulation_dt

    # init state from policy config
    try:
        initial_qpos = np.array(lmj_hydra_config.experiment.env_params.init_state_params.qpos_init, dtype=np.float32)
        initial_qvel = np.array(lmj_hydra_config.experiment.env_params.init_state_params.qvel_init, dtype=np.float32)
        if len(initial_qpos) == m.nq and len(initial_qvel) == m.nv:
            d.qpos[:] = initial_qpos
            d.qvel[:] = initial_qvel
            print("Initial state loaded from policy config.")
        else:
            print("Warning: init state length mismatch. Using default init.")
    except Exception as e:
        print(f"Warning: Could not load initial state ({e}). Using default init.")

    mujoco.mj_forward(m, d)

    # runtime goal + RNG
    rng = np.random.default_rng(1)  # change seed
    goal_pos = INIT_GOAL_POS.copy()

    # sync goal site position
    m.site("goal_site").pos = np.array([float(goal_pos[0]), float(goal_pos[1]), 0.035], dtype=np.float32)
    mujoco.mj_fwdPosition(m, d)

    # controller state
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    target_dof_kps = kps.copy()
    target_dof_kds = kds.copy()

    cmd = np.zeros(16, dtype=np.float32)
    gait_frequency = float(cmd_params["gait_frequency"])
    policy_dt = simulation_dt * control_decimation

    # init goal-reaching gait generator
    GG = GoalReachingGaitGenerator(
        model=m,
        data=d,
        max_angle=np.deg2rad(30.0),
        gait_frequency=gait_frequency,
        policy_dt=policy_dt,
        feet_distance=float(cmd_params["feet_distance"]),
        stop_steps=int(cmd_params["stop_steps"]),
    )
    GG.print_instruction()

    reached_hold = 0
    sim_start_time = time.time()

    with mujoco.viewer.launch_passive(m, d, key_callback=GG.key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()


            if (time.time() - sim_start_time) > float(simulation_duration):
                print("[done] timeout reached, exiting.")
                break

            for i in range(control_decimation):
                tau = pd_control(
                    target_dof_pos, d.qpos[7:], target_dof_kps, np.zeros_like(kds), d.qvel[6:], target_dof_kds
                )
                d.ctrl[:] = tau

                mujoco.mj_step(m, d)

            # --- Prepare Observations ---
            qj = d.qpos[7:]
            dqj = d.qvel[6:]
            quat = d.qpos[3:7]            # base quat (wxyz)
            base_ang_vel = d.qvel[3:6]
            projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))

            # --- gait phase ---

            foot_offset, gait_info = GG.query_cmd(
                goal_pos=goal_pos,
                q_pos=d.qpos[:].copy(),
            )
            l_offset, l_orn_offset, r_offset, r_orn_offset = foot_offset

            if GG.sample_goal:
                if GG.swing_foot_idx == 0:
                    rot_stance = np_R.from_matrix(d.site("right_foot").xmat.reshape(3, 3))
                    stance_yaw = rot_stance.as_euler("xyz")[2]
                    rot_stance_flat = np_R.from_euler("z", stance_yaw)

                    cmd_quat = np.array([l_orn_offset[1], l_orn_offset[2], l_orn_offset[3], l_orn_offset[0]])
                    rot_cmd = np_R.from_quat(cmd_quat)
                    target_rot = rot_stance_flat * rot_cmd

                    m.site("foot_1").pos = d.site_xpos[right_foot_id] + rot_stance_flat.apply(l_offset)
                    m.site("foot_1").pos[2] = 0.0
                    m.site("foot_1").quat = target_rot.as_quat(scalar_first=True)

                else:
                    rot_stance = np_R.from_matrix(d.site("left_foot").xmat.reshape(3, 3))
                    stance_yaw = rot_stance.as_euler("xyz")[2]
                    rot_stance_flat = np_R.from_euler("z", stance_yaw)

                    cmd_quat = np.array([r_orn_offset[1], r_orn_offset[2], r_orn_offset[3], r_orn_offset[0]])
                    rot_cmd = np_R.from_quat(cmd_quat)
                    target_rot = rot_stance_flat * rot_cmd

                    m.site("foot_0").pos = d.site_xpos[left_foot_id] + rot_stance_flat.apply(r_offset)
                    m.site("foot_0").pos[2] = 0.0
                    m.site("foot_0").quat = target_rot.as_quat(scalar_first=True)

                mujoco.mj_fwdPosition(m, d)

            # --- cmd vector ---
            cmd = np.concatenate([l_offset, l_orn_offset, r_offset, r_orn_offset, gait_info], dtype=np.float32)

            # --- Construct Observation Vector ---
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
            target_dof_pos[1] = 1.0

            target_dof_pos = np.clip(target_dof_pos, min_angles, max_angles)
            target_dof_kps = kps.copy()
            target_dof_kds = kds.copy()


            dist_to_goal = float(np.linalg.norm((goal_pos - d.qpos[:3])[:2]))
            if dist_to_goal < REACH_THRESH:
                reached_hold += 1
            else:
                reached_hold = 0

            if reached_hold == 1:
                print(f"[goal] close enough: dist={dist_to_goal:.3f} < {REACH_THRESH}, holding...")

            if reached_hold >= REACH_HOLD_STEPS:
                if GOAL_MODE == MODE_STOP:
                    print(f"[done] reached goal. dist={dist_to_goal:.3f}.")

                elif GOAL_MODE == MODE_RANDOM_NEXT:
                    print(f"[goal] reached. dist={dist_to_goal:.3f}. sampling next goal...")


                    if GG.gaits_to_still == 0:
                        goal_pos = sample_next_goal(cur_pos_xy=d.qpos[:2], rng=rng)


                    m.site("goal_site").pos = np.array([float(goal_pos[0]), float(goal_pos[1]), 0.035], dtype=np.float32)

                    reached_hold = 0

                    mujoco.mj_fwdPosition(m, d)

                    print(f"[goal] new goal = ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")

            viewer.sync()

            time_until_next_step = m.opt.timestep * control_decimation - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()