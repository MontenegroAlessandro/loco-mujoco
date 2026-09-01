import time
import os
import sys
import math
import csv
from dataclasses import dataclass, asdict
from typing import List

# (optional) set env before jax import if you care
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

from gait_generators import PlannedFootstepGaitGenerator
from footstep_planner import *  # TrafficCone, LatticeConfig, action_set, CollisionChecker, FootRect, StepConstraints, RStarParams, State, rstar_plan
from loco_mujoco.algorithms import PPOJax

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def enable_auto_remove(GG, mov_dir):
    GG.teleop["move_enabled"] = True
    GG.teleop["mov_dir"] = mov_dir


# -------------------------
# Utils
# -------------------------
def quat_rotate_inverse(q, v):
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def yaw_to_quat_wxyz(yaw: float):
    return math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)


def sample_nonoverlap_xy(rng, placed_xy, x_rng, y_rng, cone_base, clearance, max_tries=1000):
    cone_radius = np.sqrt(2) * cone_base / 2
    min_dist = 2 * cone_radius + clearance
    for _ in range(max_tries):
        x = rng.uniform(*x_rng)
        y = rng.uniform(*y_rng)
        ok = True
        for px, py in placed_xy:
            if (x - px) ** 2 + (y - py) ** 2 < min_dist**2:
                ok = False
                break
        if ok:
            return float(x), float(y)
    return float(rng.uniform(*x_rng)), float(rng.uniform(*y_rng))


# -------------------------
# Policy
# -------------------------
class LMJPolicy:
    def __init__(self, policy_path: str) -> None:
        agent_conf, agent_state = PPOJax.load_agent(policy_path)

        train_state = agent_state.train_state
        train_state.params["log_std"] = np.ones_like(train_state.params["log_std"]) * -np.inf  # deterministic

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
        return self._jit_sample_action(
            self.network_apply, self.train_state.params, self.train_state.run_stats, self._rng, obs
        )


# -------------------------
# Trial result
# -------------------------
@dataclass
class TrialResult:
    trial: int
    seed: int
    success: bool
    reason: str       # success / plan_failed / timeout
    time_s: float
    final_dist: float


# -------------------------
# One trial
# -------------------------
def run_trial(config: DictConfig, policy: LMJPolicy, seed: int, trial_idx: int, render: bool) -> TrialResult:
    rng = np.random.default_rng(int(seed))

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
    asymmetric = config.get("scale_action_to_jnt_limits", False)
    scale_neg = - (min_angles - default_angles)
    scale_pos = max_angles - default_angles

    num_qj = len(default_angles)  # Number of actuated joints (23)
    base_num_actions = config["num_actions"]  # This is also 23

    cmd_params = config["command"]
    is_gp_adaptive = cmd_params["is_gp_adaptive"]
    base_num_actions += 1 if is_gp_adaptive else 0

    action_delay_steps = 5

    # --- Load robot spec ---
    spec = mujoco.MjSpec.from_file(xml_path)
    wb = spec.worldbody

    # --- Build arena obstacles (cones) ---
    arena = config["arena"]
    cones: list[TrafficCone] = []
    for i in range(int(arena.n_cones)):
        placed_xy = [(c.cx, c.cy) for c in cones]
        x, y = sample_nonoverlap_xy(
            rng, placed_xy,
            arena.field.x_range, arena.field.y_range,
            cone_base=0.30, clearance=0.30
        )
        cone_yaw = rng.uniform(-np.pi, np.pi)
        cones.append(TrafficCone(cx=x, cy=y, yaw=cone_yaw, half=0.30))

        wb.add_geom(
            name=f"cone_{i:03d}",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="traffic_cone",
            pos=(float(x), float(y), 0.0),
            quat=yaw_to_quat_wxyz(cone_yaw),
            rgba=(1.0, 0.4, 0.0, 1.0),
            contype=1,
            conaffinity=1,
        )

    # goal marker
    goal_disk_half_h = 0.001
    wb.add_site(
        name="goal_site",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(0.20, goal_disk_half_h, 0.0),
        pos=(0.0, 0.0, goal_disk_half_h),
        quat=(1, 0, 0, 0),
        group=2,
        rgba=(1.0, 0.2, 0.2, 0.35),
    )

    # delete all *_col
    for geom in spec.geoms:
        if geom.name.endswith("_col"):
            geom.delete()

    m = spec.compile()
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # init from config (optional)
    try:
        initial_qpos = np.array(config.init_state_params.qpos_init, dtype=np.float32)
        initial_qvel = np.array(config.init_state_params.qvel_init, dtype=np.float32)
        if len(initial_qpos) == m.nq and len(initial_qvel) == m.nv:
            d.qpos[:] = initial_qpos
            d.qvel[:] = initial_qvel
    except Exception:
        pass

    mujoco.mj_forward(m, d)

    # fixed goal
    goal_pos = np.array([arena["goal"]["pos"][0], arena["goal"]["pos"][1], arena["goal"]["pos"][2]], dtype=np.float32)
    m.site("goal_site").pos = np.array([float(goal_pos[0]), float(goal_pos[1]), 0.0], dtype=np.float32)
    mujoco.mj_fwdPosition(m, d)

    # controller state
    action = np.zeros(base_num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    target_dof_kps = kps.copy()
    target_dof_kds = kds.copy()

    # Initialize context variables
    last_action = np.zeros(base_num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    delayed_target_dof_pos = target_dof_pos.copy()
    target_dof_kps = kps.copy()
    target_dof_kds = kds.copy()

    cmd = np.zeros(16, dtype=np.float32)
    counter = 1
    gait_frequency = float(cmd_params["gait_frequency"])

    # planner
    lattice_config = LatticeConfig()
    cons = StepConstraints()
    params = RStarParams()
    foot_actions = action_set(lattice_config)

    cc = CollisionChecker(cones=cones, foot=FootRect())

    # start state from left foot
    left_foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    right_foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    left_foot_pos = d.site_xpos[left_foot_id]
    left_foot_xmat = d.site_xmat[left_foot_id].reshape(3, 3)
    left_foot_yaw = np_R.from_matrix(left_foot_xmat).as_euler("xyz")[2]
    start = State(float(left_foot_pos[0]), float(left_foot_pos[1]), float(left_foot_yaw), 0)

    ok, path, stats = rstar_plan(
        start,
        (float(goal_pos[0]), float(goal_pos[1])),
        foot_actions,
        lattice_config,
        cc,
        cons,
        params,
        rng_seed=int(seed),
    )
    if (not ok) or (path is None) or (len(path) == 0):
        return TrialResult(trial=trial_idx, seed=seed, success=False, reason="plan_failed",
                           time_s=0.0, final_dist=float("inf"))

    GG = PlannedFootstepGaitGenerator(
        model=m, data=d,
        feet_distance=float(cmd_params["feet_distance"]),
        stop_steps=int(cmd_params["stop_steps"]),
    )
    GG.set_plan([(s.x, s.y, s.yaw, s.foot) for s in path])
    enable_auto_remove(GG, "FWD")

    reached_hold = 0
    sim_t = 0.0
    control_dt = simulation_dt * control_decimation

    def _check_fall(m, d) -> bool:
        # 1) base height
        base_z = float(d.qpos[2])
        if base_z < float(config.get("fail", {}).get("min_base_z", 0.35)):
            return True

        # 2) roll/pitch (optional, 很实用)
        quat_wxyz = d.qpos[3:7].copy()
        # scipy expects (x,y,z,w)
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
        rpy = np_R.from_quat(quat_xyzw).as_euler("xyz", degrees=False)
        roll, pitch = float(rpy[0]), float(rpy[1])
        rp_max = float(config.get("fail", {}).get("max_roll_pitch", np.deg2rad(60.0)))
        if abs(roll) > rp_max or abs(pitch) > rp_max:
            return True

        return False

    def _check_collision_with_cones(m, d) -> bool:
        # 扫所有 contact，看有没有 cone geom
        for ci in range(d.ncon):
            c = d.contact[ci]
            g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            if (g1 and "cone_" in g1) or (g2 and "cone_" in g2):
                return True
        return False

    def step_once():
        nonlocal counter, last_action, target_dof_pos, target_dof_kps, target_dof_kds, reached_hold, sim_t, cmd

        # physics PD
        for _ in range(control_decimation):
            counter += 1
            tau = pd_control(
                target_dof_pos, d.qpos[7:], target_dof_kps,
                np.zeros_like(kds), d.qvel[6:], target_dof_kds
            )
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

        sim_t += control_dt

        # obs
        qj = d.qpos[7:]
        dqj = d.qvel[6:]
        quat = d.qpos[3:7]
        base_ang_vel = d.qvel[3:6]
        projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))

        foot_offset, gait_info = GG.query_cmd(q_pos=d.qpos[:].copy(), goal_pos=goal_pos)
        l_offset, l_orn_offset, r_offset, r_orn_offset = foot_offset

        # debug foot sites
        if GG.sample_goal:
            if GG.swing_foot_idx == 0:
                rot_stance = np_R.from_matrix(d.site("right_foot").xmat.reshape(3, 3))
                stance_yaw = rot_stance.as_euler("xyz")[2]
                rot_stance_flat = np_R.from_euler("z", stance_yaw)

                cmd_quat = np.array([l_orn_offset[1], l_orn_offset[2], l_orn_offset[3], l_orn_offset[0]])
                rot_cmd = np_R.from_quat(cmd_quat)
                target_rot = rot_stance_flat * rot_cmd

            else:
                rot_stance = np_R.from_matrix(d.site("left_foot").xmat.reshape(3, 3))
                stance_yaw = rot_stance.as_euler("xyz")[2]
                rot_stance_flat = np_R.from_euler("z", stance_yaw)

                cmd_quat = np.array([r_orn_offset[1], r_orn_offset[2], r_orn_offset[3], r_orn_offset[0]])
                rot_cmd = np_R.from_quat(cmd_quat)
                target_rot = rot_stance_flat * rot_cmd


            mujoco.mj_fwdPosition(m, d)

        cmd = np.concatenate([l_offset, l_orn_offset, r_offset, r_orn_offset, gait_info], dtype=np.float32)

        # --- Construct Observation Vector ---
        obs_list = []
        obs_list += projected_gravity.flatten().tolist()
        obs_list += qj.flatten().tolist()
        obs_list += (base_ang_vel * 1.0).flatten().tolist()
        obs_list += (dqj * 0.1).flatten().tolist()
        obs_list += last_action.flatten().tolist()  # 'action' here is the previous action
        obs_list += cmd.flatten().tolist()

        critic_n_obs = 78 if not is_gp_adaptive else 79
        # obs = [0.0] * (78) + obs_list
        obs = [0.0] * critic_n_obs + obs_list
        obs = np.array(obs, dtype=np.float32).reshape(1, -1)

        # Override Head Pitch Angle in Observation
        obs[0, critic_n_obs + 3] = 0.0  # Head Yaw Angle
        obs[0, critic_n_obs + 4] = 0.0  # Head Pitch joint position

        # --- Policy Inference ---
        emitted_action = np.asarray(policy.predict_action(obs)).flatten()
        if is_gp_adaptive:
            GG.set_gp_offset(emitted_action[-1])
            # print(f"Mapped GP Offset: {GG.gp_off:.4f}\nUnmapped GP Offset: {gp_off:.4f}")

        # Apply smoothing/filtering to the action
        last_action = last_action * 0.0 + emitted_action * 1.0
        clipped_action = np.clip(emitted_action, -1.0, 1.0)
        if asymmetric and is_gp_adaptive:
            clipped_action = np.clip(clipped_action[:-1], None, 0.0) * scale_neg + np.clip(clipped_action[:-1], 0.0,
                                                                                           None) * scale_pos

        # Deconstruct action vector into control commands
        target_dof_pos = (
                clipped_action[:num_qj] + default_angles[:num_qj]
        )  # Use num_qj here as it's the base action for positions

        # Override head joint for control
        target_dof_pos[0] = 0.0
        target_dof_pos[1] = 0.0  # 1.0

        # Clip target_dof_pos to joint limits
        target_dof_pos = np.clip(target_dof_pos, min_angles, max_angles)

        target_dof_kps = kps.copy()
        target_dof_kds = kds.copy()


        dist_to_goal = float(np.linalg.norm((goal_pos - d.qpos[:3])[:2]))

        if _check_fall(m, d):
            return True, dist_to_goal, "fall"

        if _check_collision_with_cones(m, d):
            return True, dist_to_goal, "collision"

        if dist_to_goal < float(config["reach"]["threshold"]):
            reached_hold += 1
        else:
            reached_hold = 0

        if reached_hold >= int(config["reach"]["hold_steps"]):
            return True, dist_to_goal, "success"
        return False, dist_to_goal, "running"

    # loop
    if render:
        sim_start_wall = time.time()
        with mujoco.viewer.launch_passive(m, d, key_callback=GG.key_callback) as viewer:
            while viewer.is_running():
                step_wall = time.time()

                if sim_t >= simulation_duration:
                    dist = float(np.linalg.norm((goal_pos - d.qpos[:3])[:2]))
                    return TrialResult(trial_idx, seed, False, "timeout", sim_t, dist)

                done, dist, reason = step_once()
                if done:
                    if reason == "success":
                        return TrialResult(trial_idx, seed, True, "success", sim_t, dist)
                    else:
                        return TrialResult(trial_idx, seed, False, reason, sim_t, dist)

                viewer.sync()

                # realtime pacing
                time_until_next = control_dt - (time.time() - step_wall)
                if time_until_next > 0:
                    time.sleep(time_until_next)

        # viewer closed by user
        dist = float(np.linalg.norm((goal_pos - d.qpos[:3])[:2]))
        return TrialResult(trial_idx, seed, False, "timeout", sim_t, dist)

    else:
        while sim_t < simulation_duration:
            done, dist, reason = step_once()
            if done:
                if reason == "success":
                    return TrialResult(trial_idx, seed, True, "success", sim_t, dist)
                else:
                    return TrialResult(trial_idx, seed, False, reason, sim_t, dist)

        dist = float(np.linalg.norm((goal_pos - d.qpos[:3])[:2]))
        return TrialResult(trial_idx, seed, False, "timeout", sim_t, dist)



@hydra.main(config_name="config_sim2sim_obstacle_avoidance")
def main(config: DictConfig):
    eval_config = config.get("eval", {})
    n_trials = int(eval_config.get("n_trials", 10))
    render = bool(eval_config.get("render", False))

    base_seed = int(config["seed"])

    # load policy once
    policy = LMJPolicy(policy_path=config["agent_path"])

    # warmup once
    total_obs = max(policy.agent_conf.network.actor_obs_ind.max(), policy.agent_conf.network.critic_obs_ind.max()) + 1
    for _ in range(int(config["policy_warmup_steps"])):
        dummy_obs = jnp.zeros((1, total_obs), dtype=np.float32)
        _ = policy.predict_action(dummy_obs)

    results: List[TrialResult] = []
    for i in range(n_trials):
        seed = base_seed + i
        do_render = render

        r = run_trial(config, policy, seed=seed, trial_idx=i, render=do_render)
        results.append(r)
        print(f"[trial {i:03d}] seed={seed} success={r.success} reason={r.reason} "
              f"time={r.time_s:.2f}s final_dist={r.final_dist:.3f}")

    # summary + save csv
    succ = sum(int(r.success) for r in results)
    print("\n========== SUMMARY ==========")
    print(f"trials={len(results)}  success={succ}  success_rate={succ/len(results)*100:.1f}%")

    out_csv = "eval_results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    print(f"[saved] {out_csv}")


if __name__ == "__main__":
    main()
