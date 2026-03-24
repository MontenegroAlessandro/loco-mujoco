import time
import os
import sys
import mujoco
import mujoco.viewer
import numpy as np
import hydra
from omegaconf import DictConfig
import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation as np_R

from loco_mujoco.environments.utils import add_spiral_staircase
from loco_mujoco.algorithms import PPOJax

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

EPS = 0.15

# Global keyboard state
keyboard_state = {
    'paused': True,
    'reset_requested': False
}

def key_callback(keycode):
    """Callback for keyboard events"""
    if keycode == ord('P') or keycode == ord('p'):
        keyboard_state['paused'] = not keyboard_state['paused']
        print(f"{'Paused' if keyboard_state['paused'] else 'Running'}")
    elif keycode == ord('\\'):  # Backslash for reset
        keyboard_state['reset_requested'] = True
        print("Reset requested...")


# =====================================================================================================================
class CurvedStairPlanGenerator:
    """
    Generates a foot placement plan for a curved (chained) staircase.

    Mirrors exactly the geometry of add_spiral_staircase: each step's back edge
    is flush with the front edge of the previous step, and the heading rotates by
    `rotation_per_step_deg` at every step. No radius / fixed axis involved.

    The plan has three phases:
      1. Flat approach: straight walk from origin to the first step.
      2. Curved ascent: one foothold pair per step, feet straddling the step
         center along the lateral direction of that step's local frame.
      3. Landing platform: flat footholds on the top platform.

    Each foothold carries a `target_yaw` so the gait controller can smoothly
    track the changing heading around the curve.
    """

    def __init__(
            self,
            first_step_xy,              # (x, y) centre of the FIRST step (matches add_spiral_staircase)
            num_steps,
            step_height,
            step_length,
            step_width,
            feet_spacing,
            rotation_per_step_deg,
            initial_yaw_deg,
            n_approach_steps=4,
            n_platform_steps=3,
            platform_length=None,
        ):
        self.first_step_xy      = np.array(first_step_xy,  dtype=np.float64)
        self.num_steps          = num_steps
        self.step_height        = step_height
        self.step_length        = step_length
        self.step_width         = step_width
        self.feet_spacing       = feet_spacing
        self.rotation_per_step  = np.deg2rad(rotation_per_step_deg)
        self.initial_yaw        = np.deg2rad(initial_yaw_deg)
        self.n_approach         = n_approach_steps
        self.n_platform         = n_platform_steps
        self.platform_length    = (platform_length if platform_length is not None
                                   else step_length * n_platform_steps)

    # ------------------------------------------------------------------
    def _build_step_frames(self):
        """
        Reproduce the chaining logic from add_spiral_staircase and return
        a list of (center_3d, forward_dir_2d, lateral_dir_2d, yaw_rad) per step.
        """
        frames = []
        step_center = np.array([self.first_step_xy[0], self.first_step_xy[1], self.step_height])
        current_yaw = self.initial_yaw

        for _ in range(self.num_steps):
            fwd = np.array([np.cos(current_yaw), np.sin(current_yaw)])
            lat = np.array([-np.sin(current_yaw), np.cos(current_yaw)])  # 90° CCW = "left"
            frames.append((step_center.copy(), fwd, lat, current_yaw))

            # Advance: front edge of current step → back edge of next step
            next_yaw     = current_yaw + self.rotation_per_step
            next_fwd     = np.array([np.cos(next_yaw), np.sin(next_yaw)])
            front_edge   = step_center[:2] + fwd * (self.step_length / 2.0)
            next_center  = front_edge + next_fwd * (self.step_length / 2.0)

            step_center  = np.array([next_center[0], next_center[1],
                                     step_center[2] + self.step_height])
            current_yaw  = next_yaw

        # After the loop, step_center/current_yaw are "past-the-end".
        # Recover the last step's actual yaw and front edge so the platform
        # anchors flush against the last step with no gap.
        last_step_yaw    = current_yaw - self.rotation_per_step
        last_fwd         = np.array([np.cos(current_yaw), np.sin(current_yaw)])  # past-end fwd
        last_front_xy    = step_center[:2] - last_fwd * (self.step_length / 2.0)
        last_top_z       = step_center[2] - self.step_height  # top-face z of last step

        self._platform_front_xy = last_front_xy   # where platform back edge sits
        self._platform_z        = last_top_z
        self._platform_yaw      = last_step_yaw   # platform aligns with last step
        return frames

    # ------------------------------------------------------------------
    def generate_plan(self):
        """
        Returns:
            left_plan   : (N, 3) float32  world-frame left-foot targets
            right_plan  : (N, 3) float32  world-frame right-foot targets
            target_yaws : (N,)   float32  robot facing yaw per foothold [rad]
        """
        frames = self._build_step_frames()

        left_plan   = []
        right_plan  = []
        target_yaws = []

        # ---- Initial stance at origin ----
        left_plan.append ([0.0,  self.feet_spacing / 2, 0.0])
        right_plan.append([0.0, -self.feet_spacing / 2, 0.0])
        target_yaws.append(self.initial_yaw)

        # ---- Flat approach to the first step ----
        # Use step_length as the stride spacing so the robot never has to make
        # strides longer than one normal step. n_approach is derived from the
        # actual distance rather than being a fixed parameter.
        first_center, _, _, first_yaw = frames[0]
        approach_v   = first_center[:2]
        approach_d   = np.linalg.norm(approach_v)

        if approach_d > 1e-6:
            approach_dir = approach_v / approach_d
        else:
            approach_dir = np.array([np.cos(self.initial_yaw), np.sin(self.initial_yaw)])

        perp       = np.array([-approach_dir[1], approach_dir[0]])
        # n_approach = max(1, int(approach_d / self.step_length))
        n_approach = self.n_approach 

        for i in range(1, n_approach + 1):
            xy         = approach_dir * (i * self.step_length)
            interp_yaw = self.initial_yaw + (first_yaw - self.initial_yaw) * (i / n_approach)
            left_plan.append ([xy[0] + perp[0] * self.feet_spacing / 2,
                               xy[1] + perp[1] * self.feet_spacing / 2, 0.0])
            right_plan.append([xy[0] - perp[0] * self.feet_spacing / 2,
                               xy[1] - perp[1] * self.feet_spacing / 2, 0.0])
            target_yaws.append(interp_yaw)

        # ---- Curved steps ----
        for center, fwd, lat, yaw in frames:
            left_plan.append([
                center[0] + lat[0] * self.feet_spacing / 2,
                center[1] + lat[1] * self.feet_spacing / 2,
                center[2]
            ])
            right_plan.append([
                center[0] - lat[0] * self.feet_spacing / 2,
                center[1] - lat[1] * self.feet_spacing / 2,
                center[2]
            ])
            target_yaws.append(yaw)

        # ---- Landing platform ----
        # Spread n_platform footholds evenly from the front edge to the midpoint
        # so the robot walks onto the platform and settles near the centre
        # without reaching the far edge.
        plat_yaw = self._platform_yaw
        plat_fwd = np.array([np.cos(plat_yaw), np.sin(plat_yaw)])
        plat_lat = np.array([-np.sin(plat_yaw), np.cos(plat_yaw)])
        plat_z   = self._platform_z

        plat_step_d = (self.platform_length / 2.0) / max(self.n_platform, 1)
        for i in range(1, self.n_platform + 1):
            xy = self._platform_front_xy + plat_fwd * (i * plat_step_d)
            left_plan.append([
                xy[0] + plat_lat[0] * self.feet_spacing / 2,
                xy[1] + plat_lat[1] * self.feet_spacing / 2,
                plat_z
            ])
            right_plan.append([
                xy[0] - plat_lat[0] * self.feet_spacing / 2,
                xy[1] - plat_lat[1] * self.feet_spacing / 2,
                plat_z
            ])
            target_yaws.append(plat_yaw)

        return (
            np.array(left_plan,   dtype=np.float32),
            np.array(right_plan,  dtype=np.float32),
            np.array(target_yaws, dtype=np.float32),
        )


# =====================================================================================================================
class SpiralPlanFollowingGaitGenerator:
    """
    Gait controller that follows a curved-stair foot placement plan.

    Identical in interface to PlanFollowingGaitGenerator from sim2sim_stairs.py,
    with one addition: `target_yaws` is updated at every gait-phase switch so
    the foot orientation command tracks the staircase heading around the curve.
    """

    def __init__(
            self, model, data, left_plan, right_plan, target_yaws,
            gait_freq, policy_dt, max_vert: float = 0.4,
            feet_dist: float = 0.2, mode: str = "fast"):
        self.model       = model
        self.data        = data
        self.left_plan   = left_plan
        self.right_plan  = right_plan
        self.target_yaws = target_yaws

        self.gait_freq  = gait_freq
        self.policy_dt  = policy_dt
        self.gait_phase = 0.0
        self.max_vert   = max_vert
        self.feet_dist  = feet_dist
        self.l_idx      = 0
        self.r_idx      = 0

        assert mode in ["slow", "fast"]
        self.mode = mode

        self.prev_phase = 0.0

        self.left_foot_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        self.right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

        self._update_target_quat(0)
        self.reset(data)

    # ------------------------------------------------------------------
    def _update_target_quat(self, foothold_idx):
        idx      = min(foothold_idx, len(self.target_yaws) - 1)
        half_yaw = float(self.target_yaws[idx]) / 2.0
        self.target_quat = np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)])

    # ------------------------------------------------------------------
    def reset(self, data):
        self.l_idx = 0
        self.r_idx = 0
        self.placement_events = []  # list of dicts: {side, step_idx, target, actual, error_2d, error_3d}

        robot_quat = data.qpos[3:7]
        robot_yaw  = np_R.from_quat(
            [robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]]
        ).as_euler('xyz')[2]

        target_yaw = float(self.target_yaws[0])
        yaw_diff   = np.arctan2(np.sin(robot_yaw - target_yaw), np.cos(robot_yaw - target_yaw))

        # print(f"  Robot yaw: {np.rad2deg(robot_yaw):.2f}°, "
        #       f"Target yaw: {np.rad2deg(target_yaw):.2f}°, "
        #       f"Diff: {np.rad2deg(yaw_diff):.2f}°")

        self._update_target_quat(0)

        if yaw_diff < 0:
            self.gait_phase      = 0.0
            self.r_off_phase_cmd = np.zeros(3)
            self.r_orn_phase_cmd = np.array([1, 0, 0, 0])
            self.l_off_phase_cmd, self.l_orn_phase_cmd, _, _, _ = self.get_observation_cmd(data)
            # print("  Starting with LEFT foot swing (phase=0.0)")
        else:
            self.gait_phase      = 0.5
            self.l_off_phase_cmd = np.zeros(3)
            self.l_orn_phase_cmd = np.array([1, 0, 0, 0])
            _, _, self.r_off_phase_cmd, self.r_orn_phase_cmd, _ = self.get_observation_cmd(data)
            # print("  Starting with RIGHT foot swing (phase=0.5)")

        self.prev_phase = self.gait_phase
        self._clip_cmds()
        self.gp_off = self.policy_dt * self.gait_freq

        # print(f"  Controller reset: l_idx={self.l_idx}, r_idx={self.r_idx}, "
        #       f"phase={self.gait_phase}")
        # print(f"  Initial commands: L={self.l_off_phase_cmd}, R={self.r_off_phase_cmd}")

    # ------------------------------------------------------------------
    def _clip_cmds(self):
        self.l_off_phase_cmd[0] = np.clip(self.l_off_phase_cmd[0], -self.max_vert, self.max_vert)
        self.l_off_phase_cmd[1] = np.clip(self.l_off_phase_cmd[1],  0,             self.feet_dist)
        self.r_off_phase_cmd[0] = np.clip(self.r_off_phase_cmd[0], -self.max_vert, self.max_vert)
        self.r_off_phase_cmd[1] = np.clip(self.r_off_phase_cmd[1], -self.feet_dist, 0)

    # ------------------------------------------------------------------
    def _check_foot_reached_target(self, foot_pos, target_pos, tolerance=0.2):  
        """Check if foot is close enough to its target (XY distance)"""
        dist = np.linalg.norm(foot_pos[:2] - target_pos[:2])
        reached = dist <= tolerance
        # Debug output
        # if reached:
            # print(f"  Target reached! Distance: {dist:.3f}m")
        # input(f"\n\n[DEBUG] dist={dist:.3f}\npos={foot_pos[0]}\ntarget={target_pos[0]} \n\n")
        # print(f"\n\n[DEBUG] dist={dist:.3f}\npos={foot_pos}\ntarget={target_pos} \n\n")
        return reached

    def update(self, data):
        self.prev_phase = self.gait_phase
        self.gait_phase = (self.gait_phase + self.policy_dt * self.gait_freq) % 1.0
        
        # switching phase from right swing foot to left
        if (self.prev_phase > 0.5 and self.gait_phase < 0.5):
            # Right foot just finished swinging and becomes stance
            right_foot_pos = data.site_xpos[self.right_foot_id]
            right_target = self.right_plan[self.r_idx]

            # Record foot placement error
            err_3d = float(np.linalg.norm(right_foot_pos - right_target))
            err_2d = float(np.linalg.norm(right_foot_pos[:2] - right_target[:2]))
            self.placement_events.append({
                'side': 'right', 'step_idx': self.r_idx,
                'target': right_target.copy(), 'actual': right_foot_pos.copy(),
                'error_2d': err_2d, 'error_3d': err_3d,
            })

            # Advance both indices when target is reached
            tar_z = self.right_plan[self.r_idx][2]
            if self._check_foot_reached_target(right_foot_pos, right_target):
                # Advance right (which just became stance)
                self.r_idx = min(self.r_idx + 1, len(self.right_plan) - 1)
                # Maintain swing = stance + 1
                self.l_idx = min(self.r_idx, len(self.left_plan) - 1)
            

            self._update_target_quat(self.l_idx)

            self.r_off_phase_cmd = np.zeros(3)
            self.r_orn_phase_cmd = np.array([1, 0, 0, 0])
            self.l_off_phase_cmd, self.l_orn_phase_cmd = self.get_observation_cmd(data)[:2]
            self.l_off_phase_cmd[0] = np.clip(self.l_off_phase_cmd[0], -self.max_vert, self.max_vert)
            self.l_off_phase_cmd[1] = np.clip(self.l_off_phase_cmd[1],  EPS,           self.feet_dist)
            self.l_off_phase_cmd[2] = (self.left_plan[self.l_idx][2]- tar_z)
        
        # switching phase from left swing foot to right
        if (self.prev_phase <= 0.5 and self.gait_phase > 0.5):
            # Left foot just finished swinging and becomes stance
            left_foot_pos = data.site_xpos[self.left_foot_id]
            left_target = self.left_plan[self.l_idx]

            # Record foot placement error
            err_3d = float(np.linalg.norm(left_foot_pos - left_target))
            err_2d = float(np.linalg.norm(left_foot_pos[:2] - left_target[:2]))
            self.placement_events.append({
                'side': 'left', 'step_idx': self.l_idx,
                'target': left_target.copy(), 'actual': left_foot_pos.copy(),
                'error_2d': err_2d, 'error_3d': err_3d,
            })

            # Advance both indices when target is reached
            tar_z = self.left_plan[self.l_idx][2]
            if self._check_foot_reached_target(left_foot_pos, left_target):
                # Advance left (which just became stance)
                self.l_idx = min(self.l_idx + 1, len(self.left_plan) - 1)
                # Maintain swing = stance + 1
                self.r_idx = min(self.l_idx, len(self.right_plan) - 1)

            self._update_target_quat(self.r_idx)

            self.l_off_phase_cmd = np.zeros(3)
            self.l_orn_phase_cmd = np.array([1, 0, 0, 0])
            self.r_off_phase_cmd, self.r_orn_phase_cmd = self.get_observation_cmd(data)[2:4]
            self.r_off_phase_cmd[0] = np.clip(self.r_off_phase_cmd[0], -self.max_vert, self.max_vert)
            self.r_off_phase_cmd[1] = np.clip(self.r_off_phase_cmd[1], -self.feet_dist, -EPS)
            self.r_off_phase_cmd[2] = (self.right_plan[self.r_idx][2] - tar_z)
        
        # print(f"{self.l_idx}, {self.r_idx}")
    
    def update_old(self, data):
        self.prev_phase = self.gait_phase
        self.gait_phase = (self.gait_phase + self.policy_dt * self.gait_freq) % 1.0

        # Right was swinging → left becomes swing
        if self.prev_phase > 0.5 and self.gait_phase < 0.5:
            if self.mode == "fast":
                self.l_idx = min(self.r_idx + 1, len(self.left_plan) - 1)
            else:
                self.l_idx = min(self.l_idx + 1, len(self.left_plan) - 1)

            self._update_target_quat(self.l_idx)

            self.r_off_phase_cmd = np.zeros(3)
            self.r_orn_phase_cmd = np.array([1, 0, 0, 0])
            self.l_off_phase_cmd, self.l_orn_phase_cmd = self.get_observation_cmd(data)[:2]
            self.l_off_phase_cmd[0] = np.clip(self.l_off_phase_cmd[0], -self.max_vert, self.max_vert)
            self.l_off_phase_cmd[1] = np.clip(self.l_off_phase_cmd[1],  EPS,           self.feet_dist)
            self.l_off_phase_cmd[2] = (self.left_plan[self.l_idx][2]
                                       - self.right_plan[self.r_idx][2])

        # Left was swinging → right becomes swing
        if self.prev_phase <= 0.5 and self.gait_phase > 0.5:
            if self.mode == "fast":
                self.r_idx = min(self.l_idx + 1, len(self.right_plan) - 1)
            else:
                self.r_idx = min(self.r_idx + 1, len(self.right_plan) - 1)

            self._update_target_quat(self.r_idx)

            self.l_off_phase_cmd = np.zeros(3)
            self.l_orn_phase_cmd = np.array([1, 0, 0, 0])
            self.r_off_phase_cmd, self.r_orn_phase_cmd = self.get_observation_cmd(data)[2:4]
            self.r_off_phase_cmd[0] = np.clip(self.r_off_phase_cmd[0], -self.max_vert, self.max_vert)
            self.r_off_phase_cmd[1] = np.clip(self.r_off_phase_cmd[1], -self.feet_dist, -EPS)
            self.r_off_phase_cmd[2] = (self.right_plan[self.r_idx][2]
                                       - self.left_plan[self.l_idx][2])

    # ------------------------------------------------------------------
    def get_observation_cmd(self, data):
        is_left_swing = (self.gait_phase < 0.5)

        if is_left_swing:
            stance_id    = self.right_foot_id
            swing_target = self.left_plan[self.l_idx]
        else:
            stance_id    = self.left_foot_id
            swing_target = self.right_plan[self.r_idx]

        stance_pos  = data.site_xpos[stance_id]
        stance_mat  = data.site_xmat[stance_id].reshape(3, 3)
        stance_quat = np_R.from_matrix(stance_mat).as_quat(scalar_first=True)

        swing_rel_pos, swing_rel_quat = self._to_stance_frame(
            swing_target, self.target_quat, stance_pos, stance_mat, stance_quat
        )
        stance_rel_pos  = np.zeros(3)
        stance_rel_quat = np.array([1, 0, 0, 0])

        if is_left_swing:
            l_rel_pos, l_rel_quat = swing_rel_pos,  swing_rel_quat
            r_rel_pos, r_rel_quat = stance_rel_pos, stance_rel_quat
        else:
            r_rel_pos, r_rel_quat = swing_rel_pos,  swing_rel_quat
            l_rel_pos, l_rel_quat = stance_rel_pos, stance_rel_quat

        gait_info = np.array([
            np.cos(2 * np.pi * self.gait_phase),
            np.sin(2 * np.pi * self.gait_phase)
        ])
        return l_rel_pos, l_rel_quat, r_rel_pos, r_rel_quat, gait_info

    # ------------------------------------------------------------------
    def get_cmd(self):
        gait_info = np.array([
            np.cos(2 * np.pi * self.gait_phase),
            np.sin(2 * np.pi * self.gait_phase)
        ])
        return (self.l_off_phase_cmd, self.l_orn_phase_cmd,
                self.r_off_phase_cmd, self.r_orn_phase_cmd, gait_info)

    # ------------------------------------------------------------------
    def _to_stance_frame(self, t_pos, t_quat, s_pos, s_mat, s_quat_wxyz):
        rel_pos = s_mat.T @ (t_pos - s_pos)

        def wxyz_to_xyzw(q): return np.array([q[1], q[2], q[3], q[0]])
        def xyzw_to_wxyz(q): return np.array([q[3], q[0], q[1], q[2]])

        r_s      = np_R.from_quat(wxyz_to_xyzw(s_quat_wxyz))
        r_t      = np_R.from_quat(wxyz_to_xyzw(t_quat))
        rel_quat = (r_s.inv() * r_t).as_quat()
        return rel_pos, xyzw_to_wxyz(rel_quat)


# =====================================================================================================================
class LMJPolicy:
    def __init__(self, policy_path: str) -> None:
        agent_conf, agent_state = PPOJax.load_agent(policy_path)

        train_state = agent_state.train_state
        train_state.params["log_std"] = np.ones_like(train_state.params["log_std"]) * -np.inf

        key = jax.random.key(0)
        key, _rng = jax.random.split(key)

        self.agent_conf  = agent_conf
        self.train_state = train_state
        self._rng        = _rng

        self.network_apply      = agent_conf.network.apply
        self._jit_sample_action = jax.jit(self._sample_action,
                                          static_argnames=["network_apply"])
        print("Policy loaded and JIT function compiled.")

    @staticmethod
    def _sample_action(network_apply, params, run_stats, rng, obs):
        y, _ = network_apply(
            {"params": params, "run_stats": run_stats}, obs, mutable=["run_stats"]
        )
        pi, _ = y
        return jnp.atleast_2d(pi.mode())

    def predict_action(self, obs):
        return self._jit_sample_action(
            self.network_apply, self.train_state.params,
            self.train_state.run_stats, self._rng, obs
        )


# =====================================================================================================================
def quat_rotate_inverse(q, v):
    """Rotates a vector by the inverse of a quaternion (MuJoCo: [w, x, y, z])."""
    q_w   = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c


# =====================================================================================================================
@hydra.main(config_name="fp_config.yaml")
def main(config: DictConfig):

    # ------------------------------------------------------------------ params
    STEP_LEN          = config["command"]["step_spacing"]
    STEP_HEIGHT       = config["command"]["step_height"]
    STEP_WIDTH        = 1.0 # 0.6
    FEET_DIST         = float(config["command"]["feet_distance"])
    MAX_VERT          = 0.4
    N_STEPS           = 8
    ROTATION_PER_STEP = -10.0        # degrees per step; raise for tighter curve
    INITIAL_YAW_DEG   = 0.0         # first step faces +X
    FIRST_STEP_X      = 6 * STEP_LEN
    FIRST_STEP_Y      = 0.0
    PLATFORM_LENGTH   = STEP_LEN * 3
    N_APPROACH_STEPS  = 4
    N_PLATFORM_STEPS  = 3
    obs_dim           = 16
    mode              = config["command"].get("mode", "fast")

    xml_path           = config["xml_path"]
    simulation_dt      = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    agent_path         = config["agent_path"]

    kps            = np.array(config["lmj_kps"],        dtype=np.float32)
    kds            = np.array(config["lmj_kds"],        dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    min_angles     = np.array(config["min_angles"],     dtype=np.float32)
    max_angles     = np.array(config["max_angles"],     dtype=np.float32)

    num_qj      = len(default_angles)
    num_actions = config["num_actions"]

    # fixes
    approach_distance = np.sqrt(FIRST_STEP_X**2 + FIRST_STEP_Y**2)
    N_APPROACH_STEPS = max(1, int((approach_distance - 1e-6) / STEP_LEN))

    # ------------------------------------------------------------------ policy
    policy = LMJPolicy(policy_path=agent_path)

    total_obs = max(
        policy.agent_conf.network.actor_obs_ind.max(),
        policy.agent_conf.network.critic_obs_ind.max()
    ) + 1
    print("Warming up the policy network for JIT compilation...")
    for _ in range(500):
        _ = policy.predict_action(jnp.zeros((1, total_obs), dtype=np.float32))
    print("Warmup complete.")

    # ------------------------------------------------------------------ MuJoCo
    spec = mujoco.MjSpec.from_file(xml_path)
    wb   = spec.worldbody

    # Curved staircase — geometry matches CurvedStairPlanGenerator exactly
    wb = add_spiral_staircase(
        world_body=wb,
        name="curved_stair_1",
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

    # ------------------------------------------------------------------ foot plan
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

    # Visualise footholds
    for i, pos in enumerate(l_plan):
        wb.add_site(name=f"tgt_L_{i}", pos=pos,
                    size=(0.02, 0.02, 0.02), rgba=(0, 0, 1, 0.5), group=2)
    for i, pos in enumerate(r_plan):
        wb.add_site(name=f"tgt_R_{i}", pos=pos,
                    size=(0.02, 0.02, 0.02), rgba=(1, 0, 0, 0.5), group=2)

    for geom in spec.geoms:
        if geom.name.endswith("_col"):
            geom.delete()

    m = spec.compile()
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # ------------------------------------------------------------------ initial state
    initial_qpos = np.array(config["init_state_params"]["qpos_init"], dtype=np.float32)
    initial_qvel = np.array(config["init_state_params"]["qvel_init"], dtype=np.float32)

    random_yaw_deg = np.random.uniform(-15, 15)
    random_quat    = np_R.from_euler('z', np.deg2rad(random_yaw_deg)).as_quat(scalar_first=True)
    initial_qpos[3:7] = random_quat
    print(f"  Randomized initial yaw: {random_yaw_deg:.2f}°")

    if len(initial_qpos) == m.nq and len(initial_qvel) == m.nv:
        d.qpos[:] = initial_qpos
        d.qvel[:] = initial_qvel
        print("Initial state loaded from policy config.")
    else:
        print("Warning: init state length mismatch. Using default init.")

    mujoco.mj_forward(m, d)
    initial_qpos = d.qpos.copy()
    initial_qvel = d.qvel.copy()

    # ------------------------------------------------------------------ controller
    gait_freq    = float(config["command"]["gait_frequency"])
    planner_ctrl = SpiralPlanFollowingGaitGenerator(
        m, d,
        l_plan, r_plan, target_yaws,
        gait_freq,
        simulation_dt * control_decimation,
        max_vert=MAX_VERT,
        feet_dist=FEET_DIST,
        mode=mode,
    )

    # ------------------------------------------------------------------ sim state
    action         = np.zeros(num_actions,  dtype=np.float32)
    target_dof_pos = default_angles.copy()
    success        = False

    keyboard_state['paused']          = True
    keyboard_state['reset_requested'] = False

    print("\n=== Simulation Ready ===")
    print("Press 'P' to start/pause")
    print("Press '\\' (backslash) to reset\n")

    # ------------------------------------------------------------------ sim loop
    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            if np.linalg.norm(d.qpos[:2] - l_plan[-1][:2]) < 0.5:
                if not success:
                    print("SUCCESS – robot reached the landing platform!")
                success = True

            if keyboard_state['reset_requested']:
                print("Resetting...")
                d.qpos[:] = initial_qpos
                d.qvel[:] = initial_qvel
                mujoco.mj_forward(m, d)
                planner_ctrl.reset(d)
                action[:]         = 0.0
                target_dof_pos[:] = default_angles
                keyboard_state['paused']          = True
                keyboard_state['reset_requested'] = False
                success = False
                print("Reset complete – press 'P' to start")

            if keyboard_state['paused']:
                viewer.sync()
                time.sleep(0.01)
                continue

            # Physics
            for _ in range(control_decimation):
                tau = (target_dof_pos - d.qpos[7:]) * kps + (0.0 - d.qvel[6:]) * kds
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)

            planner_ctrl.update(d)
            l_off, l_orn, r_off, r_orn, gait_info = planner_ctrl.get_cmd()
            print(f"[DEBUG] LEFT: {l_off} – {l_orn}\t RIGHT: {r_off} – {r_orn}")

            # Observation
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
            obs = np.array([0.0] * critic_n_obs + obs_list, dtype=np.float32).reshape(1, -1)

            emitted_action = np.asarray(policy.predict_action(obs)).flatten()
            clipped_action = np.clip(emitted_action, -1.0, 1.0)
            action         = emitted_action

            target_dof_pos = clipped_action[:num_qj] + default_angles[:num_qj]
            target_dof_pos = np.clip(target_dof_pos, min_angles, max_angles)

            viewer.sync()

            time_until_next = m.opt.timestep * control_decimation - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)


if __name__ == "__main__":
    main()