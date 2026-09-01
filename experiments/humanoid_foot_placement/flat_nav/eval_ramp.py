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

from loco_mujoco.environments.utils import add_ramp_platform_ramp
from loco_mujoco.algorithms import PPOJax

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

# Import classes from sim2sim_ramp
from sim2sim_ramp import (RampPlanGenerator, PlanFollowingGaitGenerator, LMJPolicy,
                          quat_rotate_inverse, pd_control)


@dataclass
class EvaluationResult:
    """Store results for a single trial"""
    rise: float
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


class RampEvaluator:
    def __init__(self, config: DictConfig):
        self.config = config
        self.results: List[EvaluationResult] = []

    def check_success(self, robot_pos, goal_pos, threshold=0.5):
        """Check if robot reached the goal"""
        distance = np.linalg.norm(robot_pos[:2] - goal_pos[:2])
        return distance < threshold

    def check_fallen_old(self, d, current_height=0, min_rel_height=0.3):
        base_pos = d.qpos[:3].copy()
        return (base_pos[2] - current_height) < min_rel_height

    def check_fallen(self, d, current_height=0, min_rel_height=0.1, max_tilt_cos=0.5):
        base_pos = d.qpos[:3].copy()
        height_fallen = (base_pos[2] - current_height) < min_rel_height
        # Orientation check: gravity projected into body frame has z ≈ -1 when upright.
        # If z > -max_tilt_cos the robot is tilted more than arccos(max_tilt_cos) ≈ 60°.
        # quat = d.qpos[3:7]
        # proj_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        # orientation_fallen = proj_gravity[2] > -max_tilt_cos
        # return height_fallen or orientation_fallen
        return height_fallen

    def calculate_completion(self, robot_pos, start_pos, goal_pos):
        """Calculate what percentage of the path was completed"""
        total_distance = np.linalg.norm(goal_pos[:2] - start_pos[:2])
        covered_distance = np.linalg.norm(robot_pos[:2] - start_pos[:2])
        return min(100.0, (covered_distance / total_distance) * 100.0)

    def run_trial(self, rise: float, trial_num: int, max_time: float = 30.0,
                  headless: bool = True, randomize_orientation: bool = False,
                  max_yaw_deg: float = 10.0, record_video: bool = False) -> EvaluationResult:
        print(f"\n{'='*60}")
        print(f"Trial {trial_num + 1} - Rise: {rise}m")
        print(f"{'='*60}")

        # Setup parameters (mirroring sim2sim_ramp.py main())
        STEP_LEN        = self.config["command"]["step_spacing"]
        sign            = self.config["command"]["direction"]
        RAMP_START_X    = sign * 6 * STEP_LEN
        RISE            = rise
        RUN             = STEP_LEN * 5
        RAMP_WIDTH      = 2.0
        RAMP_THICKNESS  = 0.05
        PLATFORM_LENGTH = STEP_LEN * 4
        FEET_DIST       = float(self.config["command"]["feet_distance"])
        MAX_VERT        = 0.4
        tolerance = self.config["command"].get("tolerance", 0.15)

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
        cmd_params  = self.config["command"]

        # Load Policy
        policy = LMJPolicy(policy_path=agent_path)

        # Setup MuJoCo
        spec = mujoco.MjSpec.from_file(xml_path)
        wb   = spec.worldbody

        is_backward     = RAMP_START_X < 0
        orientation_yaw = 180.0 if is_backward else 0.0
        yaw_off         = cmd_params.get("yaw_off", 0.0)
        orientation_yaw += yaw_off

        # add_ramp_platform_ramp takes the bottom-start corner of the up-slope
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

        # Generate plan
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

        # Add visual markers
        goal_pos = l_plan[-1]
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

        # Video setup
        if record_video:
            m.vis.global_.offwidth  = 1280
            m.vis.global_.offheight = 720
        cam_z = 0.8

        video_writer = None
        if record_video:
            video_dir = Path("videos")
            video_dir.mkdir(exist_ok=True)
            video_path = video_dir / f"ramp_l_{STEP_LEN}m_rise_{rise}m_trial_{trial_num}.mp4"
            video_writer = imageio.get_writer(video_path, fps=30)
            renderer = mujoco.Renderer(m, height=720, width=1280)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance  = 4.0
            cam.elevation = -20.0

        render_dt        = 1.0 / 30.0
        last_render_time = 0.0

        # Initialize state
        try:
            initial_qpos = np.array(self.config["init_state_params"]["qpos_init"], dtype=np.float32)
            initial_qvel = np.array(self.config["init_state_params"]["qvel_init"], dtype=np.float32)

            if randomize_orientation:
                random_yaw_deg = np.random.uniform(-max_yaw_deg, max_yaw_deg)
                random_yaw_rad = np.deg2rad(random_yaw_deg)
                random_quat    = np_R.from_euler('z', random_yaw_rad).as_quat(scalar_first=True)
                initial_qpos[3:7] = random_quat
                print(f"  Randomized initial yaw: {random_yaw_deg:.2f}°")

            if len(initial_qpos) == m.nq and len(initial_qvel) == m.nv:
                d.qpos[:] = initial_qpos
                d.qvel[:] = initial_qvel
        except Exception as e:
            print(f"Warning: Could not load initial state ({e}). Using default init.")

        mujoco.mj_forward(m, d)
        start_pos = d.qpos[:3].copy()

        # Controller setup
        gait_freq    = float(self.config["command"]["gait_frequency"])
        planner_ctrl = PlanFollowingGaitGenerator(
            m, d, l_plan, r_plan, gait_freq,
            simulation_dt * control_decimation,
            max_vert=MAX_VERT,
            feet_dist=FEET_DIST,
            target_yaw=orientation_yaw, tolerance=tolerance,
        )

        target_dof_pos = default_angles.copy()
        action         = np.zeros(num_actions, dtype=np.float32)

        # Tracking variables
        start_time           = time.time()
        max_distance_reached = 0.0
        success  = False
        fell     = False
        timeout  = False

        # Simulation loop
        viewer = None
        if not headless:
            viewer = mujoco.viewer.launch_passive(m, d)

        try:
            while True:
                current_time = time.time() - start_time

                # Check timeout
                if current_time > max_time:
                    timeout = True
                    print(f"  Timeout reached ({max_time}s)")
                    break

                # Check if fallen — use the height of the current foothold targets
                if planner_ctrl.gait_phase >= 0.5: # left swings, right is stance
                    current_world_height = planner_ctrl.left_plan[planner_ctrl.l_idx, 2]
                else:
                    current_world_height = planner_ctrl.right_plan[planner_ctrl.r_idx, 2]
                if self.check_fallen(d=d, current_height=current_world_height):
                    fell = True
                    print(f"  Robot fell at t={current_time:.2f}s")
                    break

                # Check success
                robot_pos = d.qpos[:3]
                max_distance_reached = max(
                    max_distance_reached,
                    np.linalg.norm(robot_pos[:2] - start_pos[:2])
                )

                if self.check_success(robot_pos, goal_pos):
                    success = True
                    print(f"  SUCCESS! Goal reached at t={current_time:.2f}s")
                    break

                # Control loop
                for _ in range(control_decimation):
                    tau = (target_dof_pos - d.qpos[7:]) * kps + (0.0 - d.qvel[6:]) * kds
                    d.ctrl[:] = tau
                    mujoco.mj_step(m, d)

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

        # Calculate final metrics
        final_pos      = d.qpos[:3]
        final_distance = np.linalg.norm(final_pos[:2] - goal_pos[:2])
        completion     = self.calculate_completion(final_pos, start_pos, goal_pos)

        # Foot placement metrics from the gait generator
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
            rise=rise,
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
        )

        print(f"\n  Results:")
        print(f"    Success: {success}")
        print(f"    Completion: {completion:.1f}%")
        print(f"    Final distance to goal: {final_distance:.3f}m")
        print(f"    Episode length: {current_time:.2f}s")
        print(f"    Fell: {fell}")
        print(f"    Foot placements: {n_placements}  |  "
              f"2D: {mean_err_2d:.3f}±{std_err_2d:.3f}m  "
              f"3D: {mean_err_3d:.3f}±{std_err_3d:.3f}m  "
              f"accum={accum_err:.3f}m")

        return result

    def run_evaluation(self, rises: List[float], num_trials: int = 10,
                       headless: bool = True, max_time: float = 30.0,
                       randomize_orientation: bool = False, max_yaw_deg: float = 10.0,
                       record_video: bool = False) -> List[EvaluationResult]:
        """Run full evaluation across multiple rise values and trials"""
        print(f"\n{'='*60}")
        print(f"Starting Ramp Evaluation")
        print(f"Rise values: {rises}")
        print(f"Trials per rise: {num_trials}")
        print(f"Max time per trial: {max_time}s")
        if randomize_orientation:
            print(f"Orientation randomization: ±{max_yaw_deg}")
        print(f"{'='*60}\n")

        for rise in rises:
            for trial in range(num_trials):
                result = self.run_trial(rise, trial, max_time=max_time,
                                        headless=headless,
                                        randomize_orientation=randomize_orientation,
                                        max_yaw_deg=max_yaw_deg,
                                        record_video=record_video)
                self.results.append(result)

        self.print_summary()
        return self.results

    def print_summary(self):
        """Print summary statistics"""
        print(f"\n{'='*60}")
        print("RAMP EVALUATION SUMMARY")
        print(f"{'='*60}\n")

        rises = sorted(set(r.rise for r in self.results))

        for rise in rises:
            rise_results = [r for r in self.results if r.rise == rise]
            n_trials     = len(rise_results)
            n_success    = sum(1 for r in rise_results if r.success)
            success_rate = (n_success / n_trials) * 100 if n_trials > 0 else 0

            avg_completion = np.mean([r.completion_percentage for r in rise_results])
            avg_distance   = np.mean([r.final_distance for r in rise_results])
            n_fell         = sum(1 for r in rise_results if r.fell)
            n_timeout      = sum(1 for r in rise_results if r.timeout)
            avg_time       = np.mean([r.episode_length for r in rise_results])

            # Foot placement stats
            placed = [r for r in rise_results if r.n_placements > 0]
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

            print(f"Rise: {rise}m")
            print(f"  Success Rate: {success_rate:.1f}% ({n_success}/{n_trials})")
            print(f"  Avg Completion: {avg_completion:.1f}%")
            print(f"  Avg Final Distance: {avg_distance:.3f}m")
            print(f"  Falls: {n_fell}")
            print(f"  Timeouts: {n_timeout}")
            print(f"  Avg Episode Length: {avg_time:.2f}s")
            print(f"  Foot Placement Error  — "
                  f"2D: {avg_mean_err_2d:.3f}±{std_mean_err_2d:.3f}m  "
                  f"3D: {avg_mean_err_3d:.3f}±{std_mean_err_3d:.3f}m  "
                  f"accum: {avg_accum_err:.3f}±{std_accum_err:.3f}m")
            print()

    def save_results(self, output_path: str = "ramp_evaluation_results.json"):
        """Save results to JSON file"""
        results_dict = {
            "results": [
                {
                    "rise": r.rise,
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
    # Configuration — sweep over rise values
    RISES = [
        0.05,   # very easy
        0.10,   # very easy
        0.15,   # easy
        0.20,   # medium
        0.25,   # medium-hard
        0.30,   # hard
        0.35,   # very hard

    ]
    NUM_TRIALS  = 100
    MAX_TIME    = 30.0
    HEADLESS    = True
    OUTPUT_FILE = "fwd_ramp_evaluation_results.json"
    RECORD_VIDEO = False

    # Orientation randomization settings
    RANDOMIZE_ORIENTATION = True
    MAX_YAW_DEG = 30.0

    # Run evaluation
    evaluator = RampEvaluator(config)
    _ = evaluator.run_evaluation(
        rises=RISES,
        num_trials=NUM_TRIALS,
        headless=HEADLESS,
        max_time=MAX_TIME,
        randomize_orientation=RANDOMIZE_ORIENTATION,
        max_yaw_deg=MAX_YAW_DEG,
        record_video=RECORD_VIDEO,
    )

    # Save results
    evaluator.save_results(OUTPUT_FILE)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()