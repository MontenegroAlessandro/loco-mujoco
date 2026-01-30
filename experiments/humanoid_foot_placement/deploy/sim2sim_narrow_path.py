import os
import sys
import time
import numpy as np

# Add parent directory to import path to find lmj and other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

import mujoco
import mujoco.viewer
import hydra
from omegaconf import DictConfig
import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation as np_R

from loco_mujoco.algorithms import PPOJax
from experiments.humanoid_foot_placement.deploy.gait_generators import NarrowPathGaitGenerator


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



def quat_rotate_inverse(q, v):
    """Rotate vector by inverse quaternion. MuJoCo quat is [w,x,y,z]."""
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def _yaw_to_quat_wxyz(yaw: float):
    """MuJoCo quat (w,x,y,z) for pure yaw."""
    r = np_R.from_euler("z", float(yaw))
    q_xyzw = r.as_quat()  # (x,y,z,w)
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def _quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)


def _quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)



# feet-on-platform
def spawn_feet_on_platform_upright(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    start_xy: np.ndarray,
    platform_top_z: float,
    default_angles: np.ndarray,
    kps: np.ndarray,
    kds: np.ndarray,
    simulation_dt: float,
    control_decimation: int,
    foot_sites=("left_foot", "right_foot"),
    margin: float = 0.02,
    settle_seconds: float = 1.0,
    settle_kp_scale: float = 1.8,
    settle_kd_scale: float = 1.5,
    upright_after_settle: bool = True,
):

    start_xy = np.asarray(start_xy, dtype=np.float64)
    d.qvel[:] = 0.0

    # --- 1) joints to default pose ---
    d.qpos[7 : 7 + len(default_angles)] = np.asarray(default_angles, dtype=np.float64)

    # --- 2) base XY ---
    d.qpos[0] = float(start_xy[0])
    d.qpos[1] = float(start_xy[1])

    # --- 3) upright base quat (keep yaw from current quat) ---
    mujoco.mj_forward(m, d)
    q_wxyz = np.asarray(d.qpos[3:7], dtype=np.float64)
    r0 = np_R.from_quat(_quat_wxyz_to_xyzw(q_wxyz))
    yaw = float(r0.as_euler("xyz")[2])
    d.qpos[3:7] = _yaw_to_quat_wxyz(yaw)

    mujoco.mj_forward(m, d)

    # --- 4) base Z from foot alignment ---
    zmin = 1e9
    for nm in foot_sites:
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, nm)
        zmin = min(zmin, float(d.site_xpos[sid][2]))

    dz = float(d.qpos[2]) - zmin  # base_z - foot_min_z in this pose
    d.qpos[2] = float(platform_top_z + dz + margin)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    # --- 5) PD hold settle (stronger waist/legs) ---
    kps_settle = np.asarray(kps, dtype=np.float64).copy()
    kds_settle = np.asarray(kds, dtype=np.float64).copy()

    # Your joint order: waist=10, legs 11..22
    idxs = [10] + list(range(11, 23))
    for i in idxs:
        if 0 <= i < len(kps_settle):
            kps_settle[i] *= float(settle_kp_scale)
            kds_settle[i] *= float(settle_kd_scale)


    steps = max(1, int(settle_seconds / (simulation_dt * control_decimation)))
    for _ in range(steps):
        for _ in range(control_decimation):
            qj = d.qpos[7 : 7 + len(default_angles)]
            dqj = d.qvel[6 : 6 + len(default_angles)]
            tau = (default_angles - qj) * kps_settle + (0.0 - dqj) * kds_settle
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            d.qpos[0] = start_xy[0]
            d.qpos[1] = start_xy[1]
            d.qvel[0] = 0.0
            d.qvel[1] = 0.0

    mujoco.mj_forward(m, d)

    # --- 6) optional re-upright (keep yaw) ---
    if upright_after_settle:
        q_wxyz = np.asarray(d.qpos[3:7], dtype=np.float64)
        r1 = np_R.from_quat(_quat_wxyz_to_xyzw(q_wxyz))
        yaw = float(r1.as_euler("xyz")[2])
        d.qpos[3:7] = _yaw_to_quat_wxyz(yaw)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

    # debug
    zmin2 = 1e9
    for nm in foot_sites:
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, nm)
        zmin2 = min(zmin2, float(d.site_xpos[sid][2]))
    print(
        f"[spawn] platform_top_z={platform_top_z:.3f}  foot_min_z={zmin2:.3f}  base_z={float(d.qpos[2]):.3f}  yaw={yaw:.3f}"
    )



def add_platform(wb, name: str, center_xy, half_xy, half_z, rgba):
    wb.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(float(half_xy[0]), float(half_xy[1]), float(half_z)),
        pos=(float(center_xy[0]), float(center_xy[1]), float(half_z)),
        rgba=(float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])),
        contype=0xFFFF,
        conaffinity=0xFFFF,
        friction=(1.2, 0.01, 0.0001),
    )


def add_segment_box(wb, name: str, p0_xy, p1_xy, top_z: float, width: float, thick: float, rgba):
    p0 = np.asarray(p0_xy, dtype=np.float32)
    p1 = np.asarray(p1_xy, dtype=np.float32)
    dxy = p1 - p0
    L = float(np.linalg.norm(dxy))
    if L < 1e-6:
        return

    yaw = float(np.arctan2(dxy[1], dxy[0]))
    mid = 0.5 * (p0 + p1)
    quat = _yaw_to_quat_wxyz(yaw)

    wb.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(L * 0.5, width * 0.5, thick * 0.5),
        pos=(float(mid[0]), float(mid[1]), float(top_z - thick * 0.5)),
        quat=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
        rgba=(float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])),
        contype=0xFFFF,
        conaffinity=0xFFFF,
        friction=(1.2, 0.01, 0.0001),
    )


def build_waypoints(task_cfg) -> list:
    mode = str(task_cfg["mode"]).lower()

    if mode == "bridge":
        return []

    raise ValueError(f"Unknown task.mode={mode}")

def sample_point_on_platform(center, length, rng, margin=0.4):
    cx, cy = center[0], center[1]
    l_x, l_y = length[0], length[1]

    x = rng.uniform(cx - (l_x - margin), cx + (l_x - margin))
    y = rng.uniform(cy - (l_y - margin), cy + (l_y - margin))

    return np.array([x, y], dtype=np.float32)

def build_task_geometry(spec: mujoco.MjSpec, task_cfg, rng):
    wb = spec.worldbody

    # debug sites
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

    start_xy = np.array(task_cfg["start_xy"], dtype=np.float32)
    goal_xy = np.array(task_cfg["goal_xy"], dtype=np.float32)

    p_cfg = task_cfg["platform"]
    half_xy = np.array(p_cfg["half_xy"], dtype=np.float32)
    half_z = float(p_cfg["half_z"])
    p_rgba = p_cfg["rgba"]

    add_platform(wb, "platform_left", start_xy, half_xy, half_z, p_rgba)
    add_platform(wb, "platform_right", goal_xy, half_xy, half_z, p_rgba)

    platform_top_z = 2.0 * half_z

    path_cfg = task_cfg["path"]
    path_w = float(path_cfg["width"])
    path_thick = float(path_cfg["thickness"])
    extra_h = float(path_cfg.get("extra_height", 0.0))
    path_top_z = platform_top_z + extra_h
    path_rgba = path_cfg["rgba"]

    anchors = task_cfg.get("anchors", {})
    eps = float(anchors.get("edge_epsilon", 0.01))

    half_x = float(half_xy[0])
    path_entry = start_xy + np.array([half_x + eps, 0.0], dtype=np.float32)
    path_exit = goal_xy - np.array([half_x + eps, 0.0], dtype=np.float32)

    wps = build_waypoints(task_cfg)
    path_pts = [path_entry] + wps + [path_exit]

    destination = sample_point_on_platform(
        goal_xy,
        half_xy,
        rng
    )
    start_approach = start_xy + np.array([half_x - 0.40, 0.0], dtype=np.float32)
    destination_approach = goal_xy - np.array([half_x - 0.40, 0.0], dtype=np.float32)

    nav_pts = [start_approach] + path_pts + [destination_approach] + [destination]

    for i in range(len(path_pts) - 1):
        add_segment_box(
            wb,
            name=f"path_seg_{i:02d}",
            p0_xy=path_pts[i],
            p1_xy=path_pts[i + 1],
            top_z=path_top_z,
            width=path_w,
            thick=path_thick,
            rgba=path_rgba,
        )

    goal_site_z = path_top_z + 0.12
    wb.add_site(
        name="goal_site",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(0.04, 0.04, 0.04),
        pos=(float(path_entry[0]), float(path_exit[1]), float(goal_site_z)),
        quat=(1, 0, 0, 0),
        group=2,
        rgba=(0.1, 1.0, 0.1, 0.9),
    )

    return platform_top_z, path_top_z, nav_pts



@hydra.main(config_name="config_sim2sim_narrow_path.yaml")
def main(config: DictConfig):
    rng = np.random.default_rng(config["seed"])

    xml_path = config["xml_path"]
    simulation_duration = float(config["simulation_duration"])
    simulation_dt = float(config["simulation_dt"])
    control_decimation = int(config["control_decimation"])
    agent_path = config["agent_path"]

    kps = np.array(config["lmj_kps"], dtype=np.float32)
    kds = np.array(config["lmj_kds"], dtype=np.float32)

    default_angles = np.array(config["default_angles"], dtype=np.float32)
    min_angles = np.array(config["min_angles"], dtype=np.float32)
    max_angles = np.array(config["max_angles"], dtype=np.float32)

    num_qj = len(default_angles)
    num_actions = int(config["num_actions"])
    cmd_params = config["command"]

    task_cfg = config["task"]
    assert str(task_cfg.get("name", "")) == "narrow_path"

    mode = str(task_cfg["mode"]).lower()
    start_xy = np.array(task_cfg["start_xy"], dtype=np.float32)
    half_xy = np.array(task_cfg["platform"]["half_xy"], dtype=np.float32)

    reach_thresh = float(task_cfg.get("reach_thresh", 0.25))
    reach_hold_steps = int(task_cfg.get("reach_hold_steps", 12))
    fall_margin = float(task_cfg.get("fall_margin", 0.35))

    # allow gap_x for bridge mode
    if mode == "bridge":
        bridge_cfg = task_cfg.get("bridge", {})
        if "gap_x" in bridge_cfg:
            gap_x = float(bridge_cfg["gap_x"])
            task_cfg["goal_xy"] = [float(start_xy[0] + gap_x), float(start_xy[1])]

    # policy
    policy = LMJPolicy(policy_path=agent_path)
    total_obs = max(policy.agent_conf.network.actor_obs_ind.max(), policy.agent_conf.network.critic_obs_ind.max()) + 1
    for _ in range(200):
        dummy_obs = jnp.zeros((1, total_obs), dtype=jnp.float32)
        _ = policy.predict_action(dummy_obs)

    # geometry
    spec = mujoco.MjSpec.from_file(xml_path)
    platform_top_z, path_top_z, path_pts = build_task_geometry(spec, task_cfg, rng)

    # get model spec
    # delete all geoms whose names end in "_col" from spec
    for geom in spec.geoms:
        if geom.name.endswith("_col"):
            geom.delete()

    m = spec.compile()
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    left_foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    right_foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    spawn_xy = sample_point_on_platform(start_xy, half_xy, rng)

    spawn_feet_on_platform_upright(
        m, d,
        start_xy=spawn_xy,
        platform_top_z=platform_top_z,
        default_angles=default_angles,
        kps=kps, kds=kds,
        simulation_dt=simulation_dt,
        control_decimation=control_decimation,
        foot_sites=("left_foot", "right_foot"),
        margin=0.02,
        settle_seconds=float(task_cfg.get("spawn_settle_seconds", 1.0)),
        settle_kp_scale=float(task_cfg.get("spawn_settle_kp_scale", 1.8)),
        settle_kd_scale=float(task_cfg.get("spawn_settle_kd_scale", 1.5)),
        upright_after_settle=True,
    )

    # goal site height
    goal_site_z = float(path_top_z + 0.12)

    goal_stage = 0
    reached_hold = 0

    def set_goal_stage(stage: int):
        nonlocal goal_stage
        goal_stage = int(np.clip(stage, 0, len(path_pts) - 1))
        gxy = path_pts[goal_stage]
        m.site("goal_site").pos = np.array([float(gxy[0]), float(gxy[1]), goal_site_z], dtype=np.float32)
        mujoco.mj_fwdPosition(m, d)
        print(f"[goal] stage={goal_stage}/{len(path_pts)-1} -> ({gxy[0]:.2f}, {gxy[1]:.2f})")

    set_goal_stage(0)

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    target_dof_kps = kps.copy()
    target_dof_kds = kds.copy()

    gait_frequency = float(cmd_params["gait_frequency"])
    policy_dt = simulation_dt * control_decimation

    GG = NarrowPathGaitGenerator(
        model=m,
        data=d,
        max_angle=np.deg2rad(30.0),
        gait_frequency=gait_frequency,
        policy_dt=policy_dt,
        feet_distance=float(cmd_params["feet_distance"]),
        stop_steps=int(cmd_params["stop_steps"]),
        path_pts=path_pts
    )

    GG.print_instruction()

    sim_start_time = time.time()
    FALL_Z_THRESH = float(platform_top_z - fall_margin)

    sim_t = 0.0  # simulation time accumulator [s]
    t_start_walk = None  # first time movement becomes enabled & non-STILL
    t_reach_final = None  # time when final waypoint is reached
    done_once = False  # avoid printing multiple times

    with mujoco.viewer.launch_passive(m, d, key_callback=GG.key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            if (time.time() - sim_start_time) > simulation_duration:
                print("[done] timeout reached, exiting.")
                break

            # physics steps
            for _ in range(control_decimation):
                tau = pd_control(
                    target_dof_pos, d.qpos[7:], target_dof_kps, np.zeros_like(kds), d.qvel[6:], target_dof_kds
                )
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)

            sim_t += control_decimation * simulation_dt

            # fell off -> respawn
            base_z = float(d.qpos[2])

            if base_z < FALL_Z_THRESH:
                print(f"[fail] fell off (base_z={base_z:.3f} < {FALL_Z_THRESH:.3f}). respawn.")
                reached_hold = 0
                set_goal_stage(0)
                spawn_feet_on_platform_upright(
                    m, d,
                    start_xy=start_xy,
                    platform_top_z=platform_top_z,
                    default_angles=default_angles,
                    kps=kps, kds=kds,
                    simulation_dt=simulation_dt,
                    control_decimation=control_decimation,
                    foot_sites=("left_foot", "right_foot"),
                    margin=0.02,
                    settle_seconds=float(task_cfg.get("spawn_settle_seconds", 1.0)),
                    settle_kp_scale=float(task_cfg.get("spawn_settle_kp_scale", 1.8)),
                    settle_kd_scale=float(task_cfg.get("spawn_settle_kd_scale", 1.5)),
                    upright_after_settle=True,
                )

            # obs parts
            qj = d.qpos[7:]
            dqj = d.qvel[6:]
            quat = d.qpos[3:7]
            base_ang_vel = d.qvel[3:6]
            projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))

            if t_start_walk is None:
                if GG.teleop.get("move_enabled", False) and GG.teleop.get("mov_dir", "STILL") != "STILL":
                    t_start_walk = sim_t
                    print(f"[time] walking started at sim_t={t_start_walk:.3f}s")

            gxy = path_pts[goal_stage]
            goal_pos = np.array([float(gxy[0]), float(gxy[1]), 0.0], dtype=np.float32)

            foot_offset, gait_info = GG.query_cmd(goal_pos=goal_pos, q_pos=d.qpos[:].copy(), goal_stage=goal_stage)
            l_offset, l_orn_offset, r_offset, r_orn_offset = foot_offset

            # visualize sampled foot target
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

            cmd = np.concatenate([l_offset, l_orn_offset, r_offset, r_orn_offset, gait_info], dtype=np.float32)

            obs_list = []
            obs_list += projected_gravity.flatten().tolist()
            obs_list += qj.flatten().tolist()
            obs_list += (base_ang_vel * 1.0).flatten().tolist()
            obs_list += (dqj * 0.1).flatten().tolist()
            obs_list += action.flatten().tolist()
            obs_list += cmd.flatten().tolist()

            obs = [0.0] * 78 + obs_list
            obs = np.array(obs, dtype=np.float32).reshape(1, -1)

            obs[0, 81] = 0.0
            obs[0, 82] = 0.0

            emitted_action = np.asarray(policy.predict_action(obs)).flatten()
            emitted_action = np.clip(emitted_action, -1.0, 1.0)
            action = emitted_action

            target_dof_pos = action[:num_qj] + default_angles[:num_qj]
            target_dof_pos[0] = 0.0
            target_dof_pos[1] = 1.0
            target_dof_pos = np.clip(target_dof_pos, min_angles, max_angles)

            target_dof_kps = kps.copy()
            target_dof_kds = kds.copy()

            dist_to_goal = float(np.linalg.norm((goal_pos[:2] - d.qpos[:2])[:2]))
            if dist_to_goal < reach_thresh:
                reached_hold += 1
            else:
                reached_hold = 0

            if reached_hold >= reach_hold_steps and not done_once:
                reached_hold = 0
                if goal_stage < len(path_pts) - 1:
                    set_goal_stage(goal_stage + 1)
                else:
                    print("[done] reached final waypoint.")
                    done_once = True
                    t_reach_final = sim_t
                    if t_start_walk is None:
                        # if user never enabled movement, fall back to measuring from sim start
                        t_start_walk = 0.0
                    travel_time = t_reach_final - t_start_walk
                    print(f"[time] reached final at sim_t={t_reach_final:.3f}s")
                    print(f"[time] travel_time = {travel_time:.3f} s (sim time)")


            viewer.sync()

            time_until_next_step = m.opt.timestep * control_decimation - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
