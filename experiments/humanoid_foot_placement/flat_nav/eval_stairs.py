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
from typing import List, Dict
import json
from pathlib import Path

from loco_mujoco.environments.utils import add_stairs_platform_stairs
from loco_mujoco.algorithms import PPOJax

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

# Import classes from sim2sim_stairs
from sim2sim_stairs import StairPlanGenerator, PlanFollowingGaitGenerator, LMJPolicy, quat_rotate_inverse, pd_control

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

class StairEvaluator:
    def __init__(self, config: DictConfig):
        self.config = config
        self.results: List[EvaluationResult] = []
        
    def check_success(self, robot_pos, goal_pos, threshold=0.5):
        """Check if robot reached the goal"""
        distance = np.linalg.norm(robot_pos[:2] - goal_pos[:2])  # XY distance only
        return distance < threshold
    
    def check_fallen(self, d, min_height=0.3):
        """Check if robot has fallen"""
        base_height = d.qpos[2]
        return base_height < min_height
    
    def calculate_completion(self, robot_pos, start_pos, goal_pos):
        """Calculate what percentage of the path was completed"""
        total_distance = np.linalg.norm(goal_pos[:2] - start_pos[:2])
        covered_distance = np.linalg.norm(robot_pos[:2] - start_pos[:2])
        return min(100.0, (covered_distance / total_distance) * 100.0)
    
    def run_trial(self, step_height: float, trial_num: int, max_time: float = 30.0, 
                  headless: bool = True, randomize_orientation: bool = False,
                  max_yaw_deg: float = 10.0) -> EvaluationResult:
        print(f"\n{'='*60}")
        print(f"Trial {trial_num + 1} - Step Height: {step_height}m")
        print(f"{'='*60}")
        
        # Setup parameters
        STEP_LEN = self.config["command"]["step_spacing"]
        N_STEPS = 5
        sign = self.config["command"]["direction"]
        STAIR_START_X = sign * 6 * STEP_LEN
        STEP_HEIGHT = step_height  # Use the parameter
        STEP_WIDTH = 2.0
        PLATFORM_LENGTH = STEP_LEN * 4
        FEET_DIST = float(self.config["command"]["feet_distance"])
        MAX_VERT = 0.4
        
        xml_path = self.config["xml_path"]
        simulation_dt = self.config["simulation_dt"]
        control_decimation = self.config["control_decimation"]
        agent_path = self.config["agent_path"]
        
        kps = np.array(self.config["lmj_kps"], dtype=np.float32)
        kds = np.array(self.config["lmj_kds"], dtype=np.float32)
        default_angles = np.array(self.config["default_angles"], dtype=np.float32)
        min_angles = np.array(self.config["min_angles"], dtype=np.float32)
        max_angles = np.array(self.config["max_angles"], dtype=np.float32)
        
        num_qj = len(default_angles)
        num_actions = self.config["num_actions"]
        cmd_params = self.config["command"]
        
        # Load Policy
        policy = LMJPolicy(policy_path=agent_path)
        
        # Setup MuJoCo
        spec = mujoco.MjSpec.from_file(xml_path)
        wb = spec.worldbody
        
        is_backward = STAIR_START_X < 0
        orientation_yaw = 180.0 if is_backward else 0.0
        yaw_off = cmd_params.get("yaw_off", 0.0)
        orientation_yaw += yaw_off
        first_step_center = [STAIR_START_X, 0.0, STEP_HEIGHT/2]
        
        wb = add_stairs_platform_stairs(
            world_body=wb,
            name="stair_1",
            first_step_coordinates=first_step_center,
            num_steps=N_STEPS,
            step_height=STEP_HEIGHT,
            step_length=STEP_LEN,
            step_width=STEP_WIDTH,
            orientation_yaw_deg=orientation_yaw,
            platform_length=PLATFORM_LENGTH,
            backend=np
        )
        
        # Generate plan
        planner = StairPlanGenerator(abs(STAIR_START_X), N_STEPS, STEP_LEN, STEP_HEIGHT, STEP_WIDTH, FEET_DIST, 
                                     platform_length=PLATFORM_LENGTH)
        l_plan, r_plan = planner.generate_plan()
        
        if is_backward:
            l_plan[:, 0] = -l_plan[:, 0]
            r_plan[:, 0] = -r_plan[:, 0]
        
        # Add visual markers for goal
        goal_pos = l_plan[-1]  # Last foothold position
        wb.add_site(name="goal_marker", pos=goal_pos, size=(0.1, 0.1, 0.1), 
                   rgba=(0, 1, 0, 0.8), group=2, type=mujoco.mjtGeom.mjGEOM_SPHERE)
        
        for geom in spec.geoms:
            if geom.name.endswith("_col"):
                geom.delete()
        
        m = spec.compile()
        d = mujoco.MjData(m)
        m.opt.timestep = simulation_dt
        
        # Initialize state
        try:
            initial_qpos = np.array(self.config["init_state_params"]["qpos_init"], dtype=np.float32)
            initial_qvel = np.array(self.config["init_state_params"]["qvel_init"], dtype=np.float32)
            
            # Randomize orientation if requested
            if randomize_orientation:
                random_yaw_deg = np.random.uniform(-max_yaw_deg, max_yaw_deg)
                random_yaw_rad = np.deg2rad(random_yaw_deg)
                
                # Create quaternion for random yaw rotation (w, x, y, z format for MuJoCo)
                random_quat = np_R.from_euler('z', random_yaw_rad).as_quat(scalar_first=True)
                
                # Replace the quaternion in initial_qpos (indices 3:7)
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
        gait_freq = float(self.config["command"]["gait_frequency"])
        planner_ctrl = PlanFollowingGaitGenerator(m, d, l_plan, r_plan, gait_freq, 
                                                  simulation_dt * control_decimation, max_vert=MAX_VERT, feet_dist=FEET_DIST, target_yaw=orientation_yaw)
        
        target_dof_pos = default_angles.copy()
        action = np.zeros(num_actions, dtype=np.float32)
        
        # Tracking variables
        start_time = time.time()
        max_distance_reached = 0.0
        success = False
        fell = False
        timeout = False
        
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
                
                # Check if fallen
                if self.check_fallen(d):
                    fell = True
                    print(f"  Robot fell at t={current_time:.2f}s")
                    break
                
                # Check success
                robot_pos = d.qpos[:3]
                max_distance_reached = max(max_distance_reached, 
                                          np.linalg.norm(robot_pos[:2] - start_pos[:2]))
                
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
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                base_ang_vel = d.qvel[3:6]
                projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
                cmd = np.concatenate([l_off, l_orn, r_off, r_orn, gait_info])
                
                obs_list = []
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
                action = emitted_action
                
                target_dof_pos = clipped_action[:num_qj] + default_angles[:num_qj]
                # target_dof_pos[0] = 0.0
                # target_dof_pos[1] = 0.0
                # target_dof_pos[3] = -1.2
                # target_dof_pos[7] = 1.2
                target_dof_pos = np.clip(target_dof_pos, min_angles, max_angles)
                
                if viewer is not None:
                    viewer.sync()
                
        finally:
            if viewer is not None:
                viewer.close()
        
        # Calculate final metrics
        final_pos = d.qpos[:3]
        final_distance = np.linalg.norm(final_pos[:2] - goal_pos[:2])
        completion = self.calculate_completion(final_pos, start_pos, goal_pos)
        
        result = EvaluationResult(
            step_height=step_height,
            trial_num=trial_num,
            success=success,
            final_distance=final_distance,
            completion_percentage=completion,
            episode_length=current_time,
            fell=fell,
            timeout=timeout,
            max_distance_reached=max_distance_reached
        )
        
        print(f"\n  Results:")
        print(f"    Success: {success}")
        print(f"    Completion: {completion:.1f}%")
        print(f"    Final distance to goal: {final_distance:.3f}m")
        print(f"    Episode length: {current_time:.2f}s")
        print(f"    Fell: {fell}")
        
        return result
    
    def run_evaluation(self, step_heights: List[float], num_trials: int = 10, 
                      headless: bool = True, max_time: float = 30.0,
                      randomize_orientation: bool = False, max_yaw_deg: float = 10.0):
        """Run full evaluation across multiple heights and trials
        
        Args:
            step_heights: List of step heights to test
            num_trials: Number of trials per height
            headless: Run without viewer
            max_time: Maximum time per trial in seconds
            randomize_orientation: Whether to randomize initial yaw orientation
            max_yaw_deg: Maximum random yaw deviation in degrees (±)
        """
        print(f"\n{'='*60}")
        print(f"Starting Evaluation")
        print(f"Step heights: {step_heights}")
        print(f"Trials per height: {num_trials}")
        print(f"Max time per trial: {max_time}s")
        if randomize_orientation:
            print(f"Orientation randomization: ±{max_yaw_deg}")
        print(f"{'='*60}\n")
        
        for height in step_heights:
            for trial in range(num_trials):
                result = self.run_trial(height, trial, max_time=max_time, 
                                       headless=headless,
                                       randomize_orientation=randomize_orientation,
                                       max_yaw_deg=max_yaw_deg)
                self.results.append(result)
                
                # Small delay between trials
                # time.sleep(0.5)
        
        self.print_summary()
        return self.results
    
    def print_summary(self):
        """Print summary statistics"""
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}\n")
        
        # Group by height
        heights = sorted(set(r.step_height for r in self.results))
        
        for height in heights:
            height_results = [r for r in self.results if r.step_height == height]
            n_trials = len(height_results)
            n_success = sum(1 for r in height_results if r.success)
            success_rate = (n_success / n_trials) * 100 if n_trials > 0 else 0
            
            avg_completion = np.mean([r.completion_percentage for r in height_results])
            avg_distance = np.mean([r.final_distance for r in height_results])
            n_fell = sum(1 for r in height_results if r.fell)
            n_timeout = sum(1 for r in height_results if r.timeout)
            avg_time = np.mean([r.episode_length for r in height_results])
            
            print(f"Step Height: {height}m")
            print(f"  Success Rate: {success_rate:.1f}% ({n_success}/{n_trials})")
            print(f"  Avg Completion: {avg_completion:.1f}%")
            print(f"  Avg Final Distance: {avg_distance:.3f}m")
            print(f"  Falls: {n_fell}")
            print(f"  Timeouts: {n_timeout}")
            print(f"  Avg Episode Length: {avg_time:.2f}s")
            print()
    
    def save_results(self, output_path: str = "evaluation_results.json"):
        """Save results to JSON file"""
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
                    "max_distance_reached": r.max_distance_reached
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
    # Configuration
    STEP_HEIGHTS = [
        # 0.01, 
        # 0.02,
        # 0.04, 
        # 0.08, 
        # 0.10, 
        0.11, 
        # 0.115,
        # 0.12, 
        # 0.125,
        # 0.13,
        # 0.14,
        # 0.15
    ]  
    NUM_TRIALS = 100  
    MAX_TIME = 30.0  
    HEADLESS = True  
    OUTPUT_FILE = "fwd_stair_evaluation_results_z.json"
    
    # Orientation randomization settings
    RANDOMIZE_ORIENTATION = True  # Set to True to enable randomization
    MAX_YAW_DEG = 30.0  
    
    # Run evaluation
    evaluator = StairEvaluator(config)
    _ = evaluator.run_evaluation(
        step_heights=STEP_HEIGHTS,
        num_trials=NUM_TRIALS,
        headless=HEADLESS,
        max_time=MAX_TIME,
        randomize_orientation=RANDOMIZE_ORIENTATION,
        max_yaw_deg=MAX_YAW_DEG
    )
    
    # Save results
    evaluator.save_results(OUTPUT_FILE)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()