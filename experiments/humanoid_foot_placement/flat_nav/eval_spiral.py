import time
import os
import sys
import mujoco
import numpy as np
import hydra
from omegaconf import DictConfig
import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation as np_R
from dataclasses import dataclass
from typing import List
import json
from pathlib import Path
import imageio

from loco_mujoco.environments.utils import add_spiral_staircase
from loco_mujoco.algorithms import PPOJax

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

# Import classes from sim2sim_stairs_spiral
from sim2sim_stairs_spiral import (CurvedStairPlanGenerator, SpiralPlanFollowingGaitGenerator,
                                   LMJPolicy, quat_rotate_inverse)

# Collision geom names that can realistically contact a stair riser.
FOOT_AND_CALF_GEOMS = frozenset([
    "left_foot_1", "left_foot_2",
    "right_foot_1", "right_foot_2",
    "left_calf", "right_calf",
])

# Stair geom name prefix used by add_spiral_staircase
STAIR_PREFIX = "curved_stair_1"


@dataclass
class EvaluationResult:
    """Store results for a single trial"""
    step_height: float
    trial_num: int
    success: bool
    final_distance: float
    completion_percentage: float
    episode_length: float
    fell: bool
    timeout: bool
    max_distance_reached: float
    # Foot placement metrics
    n_placements: int = 0
    mean_placement_error_2d: float = 0.0
    std_placement_error_2d: float = 0.0
    mean_placement_error_3d: float = 0.0
    std_placement_error_3d: float = 0.0
    accumulated_placement_error: float = 0.0
    # Failure mode
    failure_mode: str = "none"       # "none" | "step_riser_contact" | "balance"
    n_riser_contacts: int = 0


def detect_riser_contact(model, data):
    """Check if any foot/calf geom is touching a stair riser this sub-step.
    
    Returns True on the first riser contact found.
    A riser contact has a contact normal whose z-component (frame[6]) is
    close to zero (the surface is near-vertical).
    """
    for ci in range(data.ncon):
        c = data.contact[ci]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        g1_stair = STAIR_PREFIX in g1
        g2_stair = STAIR_PREFIX in g2
        if not (g1_stair ^ g2_stair):
            continue
        robot_geom = g2 if g1_stair else g1
        # MuJoCo contact frame: columns = [normal, t1, t2] row-major
        # normal_z = frame[6].  Riser → |normal_z| ≈ 0; tread → |normal_z| ≈ 1
        normal_z = c.frame[6]
        if robot_geom in FOOT_AND_CALF_GEOMS and abs(normal_z) < 0.6:
            return True
    return False


def detect_failure_mode(model, data, current_world_height=0.0, fall_height=float('inf')):
    """Classify failure by scanning active contacts at fall time."""
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        g1_is_stair = STAIR_PREFIX in g1
        g2_is_stair = STAIR_PREFIX in g2
        if not (g1_is_stair ^ g2_is_stair):
            continue
        robot_geom = g2 if g1_is_stair else g1
        if robot_geom not in FOOT_AND_CALF_GEOMS:
            continue
        if abs(c.frame[6]) < 0.3:
            return 'step_riser_contact'
    return 'balance'


class SpiralStairEvaluator:
    def __init__(self, config: DictConfig):
        self.config = config
        self.results: List[EvaluationResult] = []

    def check_success(self, robot_pos, goal_pos, threshold=0.5):
        distance = np.linalg.norm(robot_pos[:2] - goal_pos[:2])
        return distance < threshold

    def check_fallen(self, d, current_height=0, min_rel_height=0.3):
        base_pos = d.qpos[:3].copy()
        return (base_pos[2] - current_height) < min_rel_height

    def calculate_completion(self, robot_pos, start_pos, goal_pos):
        total_distance   = np.linalg.norm(goal_pos[:2] - start_pos[:2])
        covered_distance = np.linalg.norm(robot_pos[:2] - start_pos[:2])
        return min(100.0, (covered_distance / total_distance) * 100.0)

    def run_trial(self, step_height: float, trial_num: int, max_time: float = 30.0,
                  headless: bool = True, randomize_orientation: bool = False,
                  max_yaw_deg: float = 10.0, record_video: bool = False) -> EvaluationResult:
        print(f"\n{'='*60}")
        print(f"Trial {trial_num + 1} - Step Height: {step_height}m")
        print(f"{'='*60}")

        # ---- Parameters (mirroring sim2sim_stairs_spiral main()) ----
        STEP_LEN          = self.config["command"]["step_spacing"]
        STEP_HEIGHT       = step_height
        STEP_WIDTH        = 1.0
        FEET_DIST         = float(self.config["command"]["feet_distance"])
        MAX_VERT          = 0.4
        N_STEPS           = 8
        ROTATION_PER_STEP = -10.0
        INITIAL_YAW_DEG   = 0.0
        FIRST_STEP_X      = 6 * STEP_LEN
        FIRST_STEP_Y      = 0.0
        PLATFORM_LENGTH   = STEP_LEN * 3
        N_PLATFORM_STEPS  = 3
        mode              = self.config["command"].get("mode", "fast")

        xml_path           = self.config["xml_path"]
        simulation_dt      = self.config["simulation_dt"]
        control_decimation = self.config["control_decimation"]
        agent_path         = self.config["agent_path"]

        kps            = np.array(self.config["lmj_kps"], dtype=np.float32)
        kds            = np.array(self.config["lmj_kds"], dtype=np.float32)
        default_angles = np.array(self.config["default_angles"], dtype=np.float32)
        min_angles     = np.array(self.config["min_angles"], dtype=np.float32)
        max_angles     = np.array(self.config["max_angles"], dtype=np.float32)

        num_qj     = len(default_angles)
        num_actions = self.config["num_actions"]

        approach_distance = np.sqrt(FIRST_STEP_X**2 + FIRST_STEP_Y**2)
        N_APPROACH_STEPS  = max(1, int((approach_distance - 1e-6) / STEP_LEN))

        # ---- Load Policy ----
        policy = LMJPolicy(policy_path=agent_path)

        # ---- Setup MuJoCo ----
        spec = mujoco.MjSpec.from_file(xml_path)
        wb   = spec.worldbody

        wb = add_spiral_staircase(
            world_body=wb,
            name=STAIR_PREFIX,
            first_step_coordinates=[FIRST_STEP_X, FIRST_STEP_Y, STEP_HEIGHT / 2],
            num_steps=N_STEPS,
            step_height=STEP_HEIGHT,
            step_length=STEP_LEN,
            step_width=STEP_WIDTH,
            rotation_per_step_deg=ROTATION_PER_STEP,
            initial_yaw_deg=INITIAL_YAW_DEG,
            platform_length=PLATFORM_LENGTH,
            platform_width=STEP_WIDTH,
            color=[0.25, 0.25, 0.25, 1.0],
            backend=np,
        )

        # ---- Foot plan ----
        planner = CurvedStairPlanGenerator(
            first_step_xy=[FIRST_STEP_X, FIRST_STEP_Y],
            num_steps=N_STEPS,
            step_height=STEP_HEIGHT,
            step_length=STEP_LEN,
            step_width=STEP_WIDTH,
            feet_spacing=FEET_DIST,
            rotation_per_step_deg=ROTATION_PER_STEP,
            initial_yaw_deg=INITIAL_YAW_DEG,
            n_approach_steps=N_APPROACH_STEPS,
            n_platform_steps=N_PLATFORM_STEPS,
            platform_length=PLATFORM_LENGTH,
        )
        l_plan, r_plan, target_yaws = planner.generate_plan()

        goal_pos = l_plan[-1]
        for i, pos in enumerate(l_plan):
            wb.add_site(name=f"tgt_L_{i}", pos=pos, size=(0.02, 0.02, 0.02), rgba=(0, 0, 1, 0.5), group=2)
        for i, pos in enumerate(r_plan):
            wb.add_site(name=f"tgt_R_{i}", pos=pos, size=(0.02, 0.02, 0.02), rgba=(1, 0, 0, 0.5), group=2)

        for geom in spec.geoms:
            if geom.name.endswith("_col"):
                geom.delete()

        # ---- Explicit contact pairs for foot/calf vs stair geoms ----
        stair_geom_names = [
            g.name for g in spec.geoms
            if g.name and STAIR_PREFIX in g.name
        ]
        for stair_name in stair_geom_names:
            for robot_name in FOOT_AND_CALF_GEOMS:
                pair = spec.add_pair()
                pair.geomname1 = robot_name
                pair.geomname2 = stair_name
        print(f"  Added {len(stair_geom_names) * len(FOOT_AND_CALF_GEOMS)} explicit"
              f" contact pairs ({len(stair_geom_names)} stair geoms × "
              f"{len(FOOT_AND_CALF_GEOMS)} robot geoms)")

        m = spec.compile()
        d = mujoco.MjData(m)
        m.opt.timestep = simulation_dt

        # ---- Video ----
        if record_video:
            m.vis.global_.offwidth  = 1280
            m.vis.global_.offheight = 720
        cam_z = 0.8

        video_writer = None
        if record_video:
            video_dir = Path("videos")
            video_dir.mkdir(exist_ok=True)
            video_path = video_dir / f"spiral_h_{step_height}m_trial_{trial_num}.mp4"
            video_writer = imageio.get_writer(video_path, fps=30)
            renderer = mujoco.Renderer(m, height=720, width=1280)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance  = 4.0
            cam.elevation = -20.0

        render_dt        = 1.0 / 30.0
        last_render_time = 0.0

        # ---- Initial state ----
        try:
            initial_qpos = np.array(self.config["init_state_params"]["qpos_init"], dtype=np.float32)
            initial_qvel = np.array(self.config["init_state_params"]["qvel_init"], dtype=np.float32)

            if randomize_orientation:
                random_yaw_deg = np.random.uniform(-max_yaw_deg, max_yaw_deg)
                random_quat    = np_R.from_euler('z', np.deg2rad(random_yaw_deg)).as_quat(scalar_first=True)
                initial_qpos[3:7] = random_quat
                print(f"  Randomized initial yaw: {random_yaw_deg:.2f}°")

            if len(initial_qpos) == m.nq and len(initial_qvel) == m.nv:
                d.qpos[:] = initial_qpos
                d.qvel[:] = initial_qvel
        except Exception as e:
            print(f"Warning: Could not load initial state ({e}). Using default init.")

        mujoco.mj_forward(m, d)
        start_pos = d.qpos[:3].copy()

        # ---- Controller ----
        gait_freq    = float(self.config["command"]["gait_frequency"])
        planner_ctrl = SpiralPlanFollowingGaitGenerator(
            m, d,
            l_plan, r_plan, target_yaws,
            gait_freq,
            simulation_dt * control_decimation,
            max_vert=MAX_VERT,
            feet_dist=FEET_DIST,
            mode=mode,
        )

        target_dof_pos = default_angles.copy()
        action         = np.zeros(num_actions, dtype=np.float32)

        # ---- Tracking variables ----
        start_time           = time.time()
        max_distance_reached = 0.0
        success       = False
        fell          = False
        timeout       = False
        failure_mode  = "none"
        riser_contact_times = []

        # ---- Simulation loop ----
        viewer = None
        if not headless:
            viewer = mujoco.viewer.launch_passive(m, d)

        try:
            while True:
                current_time = time.time() - start_time

                # Timeout
                if current_time > max_time:
                    timeout = True
                    print(f"  Timeout reached ({max_time}s)")
                    break

                # Fall detection
                current_world_height = max(
                    planner_ctrl.left_plan[planner_ctrl.l_idx, 2],
                    planner_ctrl.right_plan[planner_ctrl.r_idx, 2]
                )
                if self.check_fallen(d=d, current_height=current_world_height):
                    fell = True
                    failure_mode = detect_failure_mode(m, d, fall_height=float('inf'))
                    # Time lookback: if a riser contact happened recently, override
                    RISER_LOOKBACK = 1.5
                    if failure_mode == 'balance' and riser_contact_times:
                        if (d.time - riser_contact_times[-1]) <= RISER_LOOKBACK:
                            failure_mode = 'step_riser_contact'
                    print(f"  Robot fell at t={current_time:.2f}s  |  failure mode: {failure_mode}")
                    break

                # Success check
                robot_pos = d.qpos[:3]
                max_distance_reached = max(
                    max_distance_reached,
                    np.linalg.norm(robot_pos[:2] - start_pos[:2])
                )
                if self.check_success(robot_pos, goal_pos):
                    success = True
                    print(f"  SUCCESS! Goal reached at t={current_time:.2f}s")
                    break

                # ---- Physics sub-steps with riser contact detection ----
                for _ in range(control_decimation):
                    tau = (target_dof_pos - d.qpos[7:]) * kps + (0.0 - d.qvel[6:]) * kds
                    d.ctrl[:] = tau
                    mujoco.mj_step(m, d)

                    # Check riser contacts at every sub-step (transient contacts)
                    if not fell and not success:
                        if detect_riser_contact(m, d):
                            riser_contact_times.append(d.time)

                planner_ctrl.update(d)
                l_off, l_orn, r_off, r_orn, gait_info = planner_ctrl.get_cmd()

                # Build observation
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
                obs = [0.0] * critic_n_obs + obs_list
                obs = np.array(obs, dtype=np.float32).reshape(1, -1)
                obs[0, critic_n_obs + 3] = 0.0
                obs[0, critic_n_obs + 4] = 0.0

                # Policy inference
                emitted_action = np.asarray(policy.predict_action(obs)).flatten()
                clipped_action = np.clip(emitted_action, -1.0, 1.0)
                action         = emitted_action

                target_dof_pos = clipped_action[:num_qj] + default_angles[:num_qj]
                target_dof_pos = np.clip(target_dof_pos, min_angles, max_angles)

                if viewer is not None:
                    viewer.sync()

                if record_video and (d.time - last_render_time) >= render_dt:
                    cam.lookat    = d.qpos[0:3]
                    cam.lookat[2] = cam_z
                    renderer.update_scene(d, cam)
                    pixels = renderer.render()
                    video_writer.append_data(pixels)
                    last_render_time = d.time
        finally:
            if viewer is not None:
                viewer.close()
            if video_writer is not None:
                video_writer.close()
                print(f"Video saved at: {video_path}")

        # ---- Final metrics ----
        final_pos      = d.qpos[:3]
        final_distance = np.linalg.norm(final_pos[:2] - goal_pos[:2])
        completion     = self.calculate_completion(final_pos, start_pos, goal_pos)

        placement_events = planner_ctrl.placement_events
        n_placements     = len(placement_events)
        if n_placements > 0:
            errs_2d     = [e['error_2d'] for e in placement_events]
            errs_3d     = [e['error_3d'] for e in placement_events]
            mean_err_2d = float(np.mean(errs_2d))
            std_err_2d  = float(np.std(errs_2d))
            mean_err_3d = float(np.mean(errs_3d))
            std_err_3d  = float(np.std(errs_3d))
            accum_err   = float(np.sum(errs_3d))
        else:
            mean_err_2d = std_err_2d = mean_err_3d = std_err_3d = accum_err = 0.0

        result = EvaluationResult(
            step_height=step_height,
            trial_num=trial_num,
            success=success,
            final_distance=final_distance,
            completion_percentage=completion,
            episode_length=current_time,
            fell=fell,
            timeout=timeout,
            max_distance_reached=max_distance_reached,
            n_placements=n_placements,
            mean_placement_error_2d=mean_err_2d,
            std_placement_error_2d=std_err_2d,
            mean_placement_error_3d=mean_err_3d,
            std_placement_error_3d=std_err_3d,
            accumulated_placement_error=accum_err,
            failure_mode=failure_mode,
            n_riser_contacts=len(riser_contact_times),
        )

        print(f"\n  Results:")
        print(f"    Success: {success}")
        print(f"    Completion: {completion:.1f}%")
        print(f"    Final distance to goal: {final_distance:.3f}m")
        print(f"    Episode length: {current_time:.2f}s")
        print(f"    Fell: {fell}")
        if fell:
            print(f"    Failure mode: {failure_mode}")
        print(f"    Foot placements: {n_placements}  |  "
              f"2D: {mean_err_2d:.3f}±{std_err_2d:.3f}m  "
              f"3D: {mean_err_3d:.3f}±{std_err_3d:.3f}m  "
              f"accum={accum_err:.3f}m")
        print(f"    Riser contact events: {len(riser_contact_times)}")

        return result

    def run_evaluation(self, step_heights: List[float], num_trials: int = 10,
                       headless: bool = True, max_time: float = 30.0,
                       randomize_orientation: bool = False, max_yaw_deg: float = 10.0,
                       record_video: bool = False) -> List[EvaluationResult]:
        print(f"\n{'='*60}")
        print(f"Starting Spiral Stair Evaluation")
        print(f"Step heights: {step_heights}")
        print(f"Trials per height: {num_trials}")
        print(f"Max time per trial: {max_time}s")
        if randomize_orientation:
            print(f"Orientation randomization: ±{max_yaw_deg}°")
        print(f"{'='*60}\n")

        for height in step_heights:
            for trial in range(num_trials):
                result = self.run_trial(height, trial, max_time=max_time,
                                        headless=headless,
                                        randomize_orientation=randomize_orientation,
                                        max_yaw_deg=max_yaw_deg,
                                        record_video=record_video)
                self.results.append(result)

        self.print_summary()
        return self.results

    def print_summary(self):
        print(f"\n{'='*60}")
        print("SPIRAL STAIR EVALUATION SUMMARY")
        print(f"{'='*60}\n")

        heights = sorted(set(r.step_height for r in self.results))

        for height in heights:
            height_results = [r for r in self.results if r.step_height == height]
            n_trials     = len(height_results)
            n_success    = sum(1 for r in height_results if r.success)
            success_rate = (n_success / n_trials) * 100 if n_trials > 0 else 0

            avg_completion = np.mean([r.completion_percentage for r in height_results])
            avg_distance   = np.mean([r.final_distance for r in height_results])
            n_fell         = sum(1 for r in height_results if r.fell)
            n_timeout      = sum(1 for r in height_results if r.timeout)
            avg_time       = np.mean([r.episode_length for r in height_results])

            # Foot placement stats
            placed = [r for r in height_results if r.n_placements > 0]
            if placed:
                avg_mean_err_2d = float(np.mean([r.mean_placement_error_2d for r in placed]))
                std_mean_err_2d = float(np.std( [r.mean_placement_error_2d for r in placed]))
                avg_mean_err_3d = float(np.mean([r.mean_placement_error_3d for r in placed]))
                std_mean_err_3d = float(np.std( [r.mean_placement_error_3d for r in placed]))
                avg_accum_err   = float(np.mean([r.accumulated_placement_error for r in placed]))
                std_accum_err   = float(np.std( [r.accumulated_placement_error for r in placed]))
            else:
                avg_mean_err_2d = std_mean_err_2d = avg_mean_err_3d = std_mean_err_3d = float('nan')
                avg_accum_err = std_accum_err = float('nan')

            avg_riser = np.mean([r.n_riser_contacts for r in height_results])

            # Failure mode breakdown
            failed         = [r for r in height_results if r.fell]
            n_riser_fail   = sum(1 for r in failed if r.failure_mode == 'step_riser_contact')
            n_balance_fail = sum(1 for r in failed if r.failure_mode == 'balance')

            print(f"Step Height: {height}m")
            print(f"  Success Rate: {success_rate:.1f}% ({n_success}/{n_trials})")
            print(f"  Avg Completion: {avg_completion:.1f}%")
            print(f"  Avg Final Distance: {avg_distance:.3f}m")
            print(f"  Falls: {n_fell}  (riser contact: {n_riser_fail}, balance: {n_balance_fail})")
            print(f"  Timeouts: {n_timeout}")
            print(f"  Avg Episode Length: {avg_time:.2f}s")
            print(f"  Foot Placement Error  — "
                  f"2D: {avg_mean_err_2d:.3f}±{std_mean_err_2d:.3f}m  "
                  f"3D: {avg_mean_err_3d:.3f}±{std_mean_err_3d:.3f}m  "
                  f"accum: {avg_accum_err:.3f}±{std_accum_err:.3f}m")
            print(f"  Avg Riser Contact Events per trial: {avg_riser:.1f}")
            print()

    def save_results(self, output_path: str = "spiral_stair_evaluation_results.json"):
        results_dict = {
            "results": [
                {
                    "step_height": r.step_height,
                    "trial_num": r.trial_num,
                    "success": r.success,
                    "final_distance": r.final_distance,
                    "completion_percentage": r.completion_percentage,
                    "episode_length": r.episode_length,
                    "fell": r.fell,
                    "timeout": r.timeout,
                    "max_distance_reached": r.max_distance_reached,
                    "n_placements": r.n_placements,
                    "mean_placement_error_2d": r.mean_placement_error_2d,
                    "std_placement_error_2d": r.std_placement_error_2d,
                    "mean_placement_error_3d": r.mean_placement_error_3d,
                    "std_placement_error_3d": r.std_placement_error_3d,
                    "accumulated_placement_error": r.accumulated_placement_error,
                    "failure_mode": r.failure_mode,
                    "n_riser_contacts": r.n_riser_contacts,
                }
                for r in self.results
            ]
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nResults saved to: {output_file}")


@hydra.main(config_name="fp_config.yaml")
def main(config: DictConfig):
    STEP_HEIGHTS = [
        0.01,
        0.02,
        0.04,
        0.08,
        0.10,
        0.11,
        0.115,
        0.12,
        0.125,
        0.13,
    ]
    NUM_TRIALS  = 100
    MAX_TIME    = 30.0
    HEADLESS    = True
    OUTPUT_FILE = "fwd_spiral_stair_evaluation_results.json"
    RECORD_VIDEO = False

    RANDOMIZE_ORIENTATION = True
    MAX_YAW_DEG = 30.0

    evaluator = SpiralStairEvaluator(config)
    _ = evaluator.run_evaluation(
        step_heights=STEP_HEIGHTS,
        num_trials=NUM_TRIALS,
        headless=HEADLESS,
        max_time=MAX_TIME,
        randomize_orientation=RANDOMIZE_ORIENTATION,
        max_yaw_deg=MAX_YAW_DEG,
        record_video=RECORD_VIDEO,
    )

    evaluator.save_results(OUTPUT_FILE)
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()