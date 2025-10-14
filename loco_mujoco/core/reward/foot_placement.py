from types import ModuleType
from typing import Any, Dict, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
from jax._src.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R
import mujoco
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.reward.base import Reward
from loco_mujoco.core.utils import mj_jntname2qposid, mj_jntname2qvelid, mj_jntid2qposid, mj_check_collisions
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast

@struct.dataclass
class FootPlacementRewardState:
    """State for the FootPlacementReward function."""
    last_action: Union[np.ndarray, jax.Array]
    reward_components: Dict[str, Union[np.ndarray, jax.Array]]


class FootPlacementReward(Reward):
    """
    Improved FootPlacementReward aligned with CrispBoosterLocomotionReward conventions.

    Components:
    - Swing-foot tracking (pos + orn)
    - Torso height stability
    - Touchdown bonus (accurate landings)
    - Swing clearance shaping
    - Feet slip and impact penalties
    - Smooth action regularization

    All penalties are dt-scaled, following CrispBooster style.
    """

    def __init__(
        self,
        env,
        left_foot_site_name: str,
        right_foot_site_name: str,
        swing_pos_w: float = 6.0,
        swing_orn_w: float = 2.0,
        torso_height_w: float = 3.0,
        action_rate_w: float = 1e-2,
        sharp_pos: float = 60.0,
        sharp_orn: float = 400.0,
        sharp_height: float = 80.0,
        touchdown_w: float = 1.0,
        touchdown_sharp: float = 20.0,
        clearance_w: float = 0.5,
        clearance_sharp: float = 80.0,
        slip_w: float = 0.5,
        impact_w: float = 0.3,
        impact_threshold: float = 6.0,
        clearance_target: float = 0.03,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        self.goal_name = "GoalRandomFootPlacement"

        # Store weights and parameters
        self._swing_pos_w = swing_pos_w
        self._swing_orn_w = swing_orn_w
        self._torso_height_w = torso_height_w
        self._action_rate_w = action_rate_w
        self._touchdown_w = touchdown_w
        self._clearance_w = clearance_w
        self._clearance_sharp = clearance_sharp
        self._slip_w = slip_w
        self._impact_w = impact_w
        self._impact_threshold = impact_threshold
        self._clearance_target = clearance_target
        self._touchdown_sharp = touchdown_sharp

        self._sharp_pos = sharp_pos
        self._sharp_orn = sharp_orn
        self._sharp_height = sharp_height

        # Healthy height midpoint
        min_h, max_h = env.root_height_healthy_range
        self._target_height = (min_h + max_h) / 2.0

        # Model references
        model = env.model
        self._foot_site_id_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, left_foot_site_name)
        self._foot_site_id_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, right_foot_site_name)
        self._torso_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, env.root_body_name)
        self._floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        assert self._foot_site_id_left != -1 and self._foot_site_id_right != -1 and self._torso_body_id != -1

        # Try to get foot body and sensor ids (for slip)
        self._left_foot_body_id = model.site_bodyid[self._foot_site_id_left]
        self._right_foot_body_id = model.site_bodyid[self._foot_site_id_right]
        self._left_sensor_adr = None
        self._right_sensor_adr = None
        for name in getattr(env._model.sensor_names, [], []):
            if "left_foot_global_linvel" in name:
                sid = model.sensor(name).id
                self._left_sensor_adr = list(range(model.sensor_adr[sid], model.sensor_adr[sid] + model.sensor_dim[sid]))
            if "right_foot_global_linvel" in name:
                sid = model.sensor(name).id
                self._right_sensor_adr = list(range(model.sensor_adr[sid], model.sensor_adr[sid] + model.sensor_dim[sid]))

    def init_state(self, env, key, model, data, backend):
        reward_components = {
            "tracking/foot_position": 0.,
            "tracking/foot_orientation": 0.,
            "tracking/height": 0.,
            "bonus/touchdown": 0.,
            "tracking/clearance": 0.,
            "penalties/slip": 0.,
            "penalties/impact": 0.,
            "penalties/action_rate": 0.,
        }
        return FootPlacementRewardState(
            last_action=backend.zeros(env.info.action_space.shape[0]),
            reward_components=reward_components,
        )

    def __call__(self, state, action, next_state, absorbing, info, env, model, data, carry, backend):
        R = np_R if backend == np else jnp_R
        reward_state = carry.reward_state
        goal_state = getattr(carry.observation_states, self.goal_name)

        # Goal info
        swing_target_pos = goal_state.swing_target_pos
        swing_target_orn = goal_state.swing_target_orn
        swing_foot_idx = goal_state.swing_foot_idx

        # Foot IDs
        swing_site_id = jax.lax.select((swing_foot_idx == 1), self._foot_site_id_right, self._foot_site_id_left)
        swing_body_id = jax.lax.select((swing_foot_idx == 1), self._right_foot_body_id, self._left_foot_body_id)

        # Current pose
        swing_pos = data.site_xpos[swing_site_id]
        swing_orn = R.from_matrix(data.site_xmat[swing_site_id].reshape(3, 3)).as_quat(scalar_first=True)

        # --- 1. Swing-foot tracking (pos + orn)
        pos_err_sq = backend.sum(backend.square(swing_pos - swing_target_pos))
        dot_q = backend.clip(backend.sum(swing_target_orn * swing_orn), -1.0, 1.0)
        orn_err = 1.0 - backend.square(dot_q)
        swing_pos_reward = self._swing_pos_w * backend.exp(-self._sharp_pos * pos_err_sq)
        swing_orn_reward = self._swing_orn_w * backend.exp(-self._sharp_orn * orn_err)

        # --- 2. Torso height tracking
        torso_z = data.xpos[self._torso_body_id][2]
        height_err_sq = backend.square(torso_z - self._target_height)
        torso_height_reward = self._torso_height_w * backend.exp(-self._sharp_height * height_err_sq)

        # --- 3. Clearance reward 
        clearance_reward = self._clearance_w * backend.exp(
            -self._clearance_sharp * backend.square(backend.maximum(0.0, self._clearance_target - swing_pos[2]))
        ) * env.dt

        # --- 4. Touchdown bonus (based on contact + accuracy)
        foot_on_ground = mj_check_collisions(swing_body_id, self._floor_id, data, backend)
        cfrc = backend.linalg.norm(backend.array(data.cfrc_ext[swing_body_id, :3]))
        touchdown_reward = backend.where(
            backend.logical_and(foot_on_ground, cfrc > self._impact_threshold),
            self._touchdown_w * backend.exp(-self._touchdown_sharp * pos_err_sq) * env.dt,
            0.0,
        )

        # --- 5. Slip penalty
        if (swing_foot_idx == 0 and self._left_sensor_adr is not None) or (swing_foot_idx == 1 and self._right_sensor_adr is not None):
            foot_vel = (
                data.sensordata[self._left_sensor_adr]
                if swing_foot_idx == 0
                else data.sensordata[self._right_sensor_adr]
            )
        else:
            foot_vel = data.site_xvelp[swing_site_id]
        slip_penalty = self._slip_w * backend.square(backend.linalg.norm(foot_vel[:2])) * env.dt

        # --- 6. Impact penalty
        impact_penalty = self._impact_w * backend.maximum(0.0, cfrc - self._impact_threshold) * env.dt

        # --- 7. Smoothness penalty
        action_rate_penalty = self._action_rate_w * backend.sum(backend.square(action - reward_state.last_action)) * env.dt

        # --- Combine and clip
        total_reward = (
            swing_pos_reward
            + swing_orn_reward
            + torso_height_reward
            + clearance_reward
            + touchdown_reward
            - (slip_penalty + impact_penalty + action_rate_penalty)
        )
        total_reward = backend.maximum(total_reward, 0.0)

        # --- Update components
        comps = {
            "tracking/foot_position": swing_pos_reward,
            "tracking/foot_orientation": swing_orn_reward,
            "tracking/height": torso_height_reward,
            "bonus/touchdown": touchdown_reward,
            "tracking/clearance": clearance_reward,
            "penalties/slip": slip_penalty,
            "penalties/impact": impact_penalty,
            "penalties/action_rate": action_rate_penalty,
        }

        new_state = reward_state.replace(last_action=action, reward_components=comps)
        carry = carry.replace(reward_state=new_state)
        return total_reward, carry



@struct.dataclass
class FootPlacementLocomotionRewardState:
    """
    Combined state for FootPlacement + Locomotion regularizers.
    """
    last_qvel: Union[np.ndarray, jax.Array]
    last_action: Union[np.ndarray, jax.Array]
    time_since_last_touchdown: Union[np.ndarray, jax.Array]


class FootPlacementLocomotionReward(Reward):
    """
    Encourage accurate swing-foot placement + natural, stable walking.
    - Foot placement: track swing foot (pos + orn) toward GoalRandomFootPlacement.
    - Natural gait: Locomotion-style regularizers (torso stability, joint limits, torque/energy, air-time, etc.).
    - Optional (off by default): velocity-goal tracking like TargetVelocityGoalReward if GoalRandomRootVelocity is present.

    Notes:
    - Does NOT require GoalRandomRootVelocity (tracking is optional).
    - Uses the same conventions/utilities as the existing rewards in this file.
    """

    def __init__(
            self,
            env: Any,
            left_foot_site_name: str,
            right_foot_site_name: str,
            # --- Foot-placement weights ---
            swing_pos_w: float = 6.0,
            swing_orn_w: float = 2.0,
            torso_height_w: float = 3.0,
            action_rate_w: float = 1e-2,
            sharp_pos: float = 60.0,
            sharp_orn: float = 400.0,
            sharp_height: float = 80.0,
            # --- Locomotion-style regularizers (same defaults as your LocomotionReward) ---
            z_vel_coeff: float = 2.0,
            roll_pitch_vel_coeff: float = 5e-2,
            roll_pitch_pos_coeff: float = 2e-1,
            nominal_joint_pos_coeff: float = 0.0,
            nominal_joint_pos_names: Union[None, list] = None,
            joint_position_limit_coeff: float = 10.0,
            joint_vel_coeff: float = 0.0,
            joint_acc_coeff: float = 2e-7,
            joint_torque_coeff: float = 2e-5,
            air_time_max: float = 0.0,
            air_time_coeff: float = 0.0,
            symmetry_air_coeff: float = 0.0,  # keep 0.0 for humanoids unless you define a pairing
            energy_coeff: float = 0.0,
            # --- Optional velocity tracking (like TargetVelocityGoalReward) ---
            include_velocity_tracking: bool = False,
            tracking_w_exp_xy: float = 10.0,
            tracking_w_exp_yaw: float = 10.0,
            tracking_w_sum_xy: float = 1.0,
            tracking_w_sum_yaw: float = 1.0,
            # --- Scale for all locomotion regularizers (acts as a single knob) ---
            locomotion_scale: float = 0.3,
            **kwargs
        ):
        super().__init__(env, **kwargs)

        self.goal_name = "GoalRandomFootPlacement"

        # Foot-placement weights/sharpness
        self._swing_pos_w = swing_pos_w
        self._swing_orn_w = swing_orn_w
        self._torso_height_w = torso_height_w
        self._action_rate_w = action_rate_w
        self._sharp_pos = sharp_pos
        self._sharp_orn = sharp_orn
        self._sharp_height = sharp_height

        # Healthy target torso height (midpoint of your env range)
        min_h, max_h = env.root_height_healthy_range
        self._target_height = (min_h + max_h) / 2.0

        # Sites for swing foot tracking
        self._foot_site_id_left = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, left_foot_site_name)
        self._foot_site_id_right = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, right_foot_site_name)
        self._torso_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, env.root_body_name)
        assert self._foot_site_id_left != -1 and self._foot_site_id_right != -1 and self._torso_body_id != -1

        # Locomotion-style setup (match your LocomotionReward wiring)
        model = env._model
        self._free_joint_qpos_ind = np.array(mj_jntname2qposid(self._info_props["root_free_joint_xml_name"], model))
        self._free_joint_qvel_ind = np.array(mj_jntname2qvelid(self._info_props["root_free_joint_xml_name"], model))
        self._free_joint_qpos_mask = np.zeros(model.nq, dtype=bool)
        self._free_joint_qpos_mask[self._free_joint_qpos_ind] = True
        self._free_joint_qvel_mask = np.zeros(model.nv, dtype=bool)
        self._free_joint_qvel_mask[self._free_joint_qvel_ind] = True

        self._floor_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        # use the same list the repo already trusts for contacts
        self._foot_names = self._info_props["foot_geom_names"]
        self._foot_ids = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in self._foot_names]

        # Coefficients
        self._z_vel_coeff = z_vel_coeff
        self._roll_pitch_vel_coeff = roll_pitch_vel_coeff
        self._roll_pitch_pos_coeff = roll_pitch_pos_coeff
        self._nominal_joint_pos_coeff = nominal_joint_pos_coeff
        self._nominal_joint_pos_names = nominal_joint_pos_names
        self._joint_position_limit_coeff = joint_position_limit_coeff
        self._joint_vel_coeff = joint_vel_coeff
        self._joint_acc_coeff = joint_acc_coeff
        self._joint_torque_coeff = joint_torque_coeff
        self._air_time_max = air_time_max
        self._air_time_coeff = air_time_coeff
        self._symmetry_air_coeff = symmetry_air_coeff
        self._energy_coeff = energy_coeff
        self._locomotion_scale = locomotion_scale

        # Optional velocity tracking (only used if include_velocity_tracking=True AND the goal is present)
        self._include_vel_tracking = include_velocity_tracking
        self._tracking_w_exp_xy = tracking_w_exp_xy
        self._tracking_w_exp_yaw = tracking_w_exp_yaw
        self._tracking_w_sum_xy = tracking_w_sum_xy
        self._tracking_w_sum_yaw = tracking_w_sum_yaw
        self._free_jnt_name = self._info_props["root_free_joint_xml_name"]
        self._vel_idx = np.array(mj_jntname2qvelid(self._free_jnt_name, model))

        # Joint limit / nominal pose data
        self._limited_joints = np.array(model.jnt_limited, dtype=bool)
        self._limited_joints_qpos_id = model.jnt_qposadr[np.where(self._limited_joints)]
        self._joint_ranges = model.jnt_range[self._limited_joints]
        self._nominal_joint_qpos = env._model.qpos0
        if self._nominal_joint_pos_names is None:
            self._nominal_joint_qpos_id = self._limited_joints_qpos_id
        else:
            self._nominal_joint_qpos_id = np.concatenate([mj_jntname2qposid(name, model)
                                                          for name in self._nominal_joint_pos_names])

    def init_state(self, env: Any, key: Any, model: Union[MjModel, Model], data: Union[MjData, Data], backend: ModuleType):
        return FootPlacementLocomotionRewardState(
            last_qvel=data.qvel,
            last_action=backend.zeros(env.info.action_space.shape[0]),
            time_since_last_touchdown=backend.zeros(len(self._foot_ids))
        )

    def reset(self, env: Any, model: Union[MjModel, Model], data: Union[MjData, Data], carry: Any, backend: ModuleType):
        reward_state = self.init_state(env, None, model, data, backend)
        carry = carry.replace(reward_state=reward_state)
        return data, carry

    def __call__(self,
                 state: Union[np.ndarray, jnp.ndarray],
                 action: Union[np.ndarray, jnp.ndarray],
                 next_state: Union[np.ndarray, jnp.ndarray],
                 absorbing: bool,
                 info: Dict[str, Any],
                 env: Any,
                 model: Union[MjModel, Model],
                 data: Union[MjData, Data],
                 carry: Any,
                 backend: ModuleType) -> Tuple[float, Any]:

        R = np_R if backend == np else jnp_R
        reward_state = carry.reward_state

        # -------------------------------
        # Foot-placement tracking terms
        # -------------------------------
        goal_state = getattr(carry.observation_states, self.goal_name)
        swing_target_pos = goal_state.swing_target_pos
        swing_target_orn = goal_state.swing_target_orn  # (w,x,y,z) world quat
        swing_foot_idx = goal_state.swing_foot_idx
        swing_id = jax.lax.select((swing_foot_idx == 1), self._foot_site_id_right, self._foot_site_id_left)

        swing_pos = data.site_xpos[swing_id]
        swing_orn = R.from_matrix(data.site_xmat[swing_id].reshape(3, 3)).as_quat(scalar_first=True)

        # position tracking (exp(-sharp * ||pos_err||^2))
        pos_err_sq = backend.sum(backend.square(swing_pos - swing_target_pos))
        swing_pos_reward = self._swing_pos_w * backend.exp(-self._sharp_pos * pos_err_sq)

        # orientation tracking via quaternion dot (invariant to +/-)
        dot_q = backend.sum(swing_target_orn * swing_orn)
        dot_q = backend.clip(dot_q, -1.0, 1.0)
        orn_err = 1.0 - backend.square(dot_q)  # in [0,1]
        swing_orn_reward = self._swing_orn_w * backend.exp(-self._sharp_orn * orn_err)

        # torso height
        torso_z = data.xpos[self._torso_body_id][2]
        height_err_sq = backend.square(torso_z - self._target_height)
        torso_height_reward = self._torso_height_w * backend.exp(-self._sharp_height * height_err_sq)

        # action-rate smoothness (we keep ONLY this to avoid double-penalizing with LocomotionReward's action rate)
        action_rate_penalty = self._action_rate_w * backend.sum(backend.square(action - reward_state.last_action))

        # -------------------------------
        # Locomotion-style regularizers
        # -------------------------------
        global_pose_root = data.qpos[self._free_joint_qpos_ind]
        global_pos_root = global_pose_root[:3]
        global_quat_root = global_pose_root[3:]
        global_rot = R.from_quat(quat_scalarfirst2scalarlast(global_quat_root))

        global_vel_root = data.qvel[self._free_joint_qvel_ind]
        local_vel_root_lin = global_rot.inv().apply(global_vel_root[:3])
        local_vel_root_ang = global_rot.inv().apply(global_vel_root[3:])

        # velocity-based posture penalties
        z_vel_reward = (self._z_vel_coeff * -(backend.square(local_vel_root_lin[2]))) if self._z_vel_coeff > 0.0 else 0.0
        roll_pitch_vel_reward = (self._roll_pitch_vel_coeff * -backend.square(local_vel_root_ang[:2]).sum()) \
            if self._roll_pitch_vel_coeff > 0.0 else 0.0

        # torso roll/pitch posture
        roll_pitch_reward = (self._roll_pitch_pos_coeff * -backend.square(global_rot.as_euler("xyz")[:2]).sum()) \
            if self._roll_pitch_pos_coeff > 0.0 else 0.0

        # nominal joint pose (around qpos0 or a subset)
        if self._nominal_joint_pos_coeff > 0.0:
            joint_qpos_reward = (self._nominal_joint_pos_coeff *
                                 -backend.square(data.qpos[self._nominal_joint_qpos_id]
                                                 - self._nominal_joint_qpos[self._nominal_joint_qpos_id]).sum())
        else:
            joint_qpos_reward = 0.0

        # joint limits penalty (outside range)
        if self._joint_position_limit_coeff > 0.0:
            joint_positions = backend.array(data.qpos[self._limited_joints_qpos_id])
            lower_limit_penalty = -backend.minimum(joint_positions - self._joint_ranges[:, 0], 0.0).sum()
            upper_limit_penalty = backend.maximum(joint_positions - self._joint_ranges[:, 1], 0.0).sum()
            joint_position_limit_reward = self._joint_position_limit_coeff * -(lower_limit_penalty + upper_limit_penalty)
        else:
            joint_position_limit_reward = 0.0

        # joint velocity / acceleration / torque / energy
        joint_vel = data.qvel[~self._free_joint_qvel_mask]
        joint_vel_reward = self._joint_vel_coeff * -backend.square(joint_vel).sum() if self._joint_vel_coeff > 0.0 else 0.0

        if self._joint_acc_coeff > 0.0:
            last_joint_vel = reward_state.last_qvel[~self._free_joint_qvel_mask]
            acceleration_norm = backend.sum(backend.square(joint_vel - last_joint_vel) / env.dt)
            acceleration_reward = self._joint_acc_coeff * -acceleration_norm
        else:
            acceleration_reward = 0.0

        if self._joint_torque_coeff > 0.0:
            torque_norm = backend.sum(backend.square(data.qfrc_actuator[~self._free_joint_qvel_mask]))
            torque_reward = self._joint_torque_coeff * -torque_norm
        else:
            torque_reward = 0.0

        if self._energy_coeff > 0.0:
            energy = backend.sum(backend.abs(joint_vel) * backend.abs(data.qfrc_actuator[~self._free_joint_qvel_mask]))
            energy_reward = self._energy_coeff * -energy
        else:
            energy_reward = 0.0

        # air time (time since last touchdown vs. desired)
        if self._air_time_coeff > 0.0 or self._symmetry_air_coeff > 0.0:
            air_time_reward = 0.0
            foots_on_ground = backend.zeros(len(self._foot_ids))
            tslt = reward_state.time_since_last_touchdown.copy()
            for i, f_id in enumerate(self._foot_ids):
                foot_on_ground = mj_check_collisions(f_id, self._floor_id, data, backend)
                if backend == np:
                    foots_on_ground[i] = foot_on_ground
                else:
                    foots_on_ground = foots_on_ground.at[i].set(foot_on_ground)

                if backend == np:
                    if foot_on_ground:
                        air_time_reward += (tslt[i] - self._air_time_max)
                        tslt[i] = 0.0
                    else:
                        tslt[i] += env.dt
                else:
                    tslt_i, air_time_reward = jax.lax.cond(foot_on_ground,
                                                           lambda: (0.0, air_time_reward + tslt[i] - self._air_time_max),
                                                           lambda: (tslt[i] + env.dt, air_time_reward))
                    tslt = tslt.at[i].set(tslt_i)

            air_time_reward = self._air_time_coeff * air_time_reward
        else:
            tslt = reward_state.time_since_last_touchdown.copy()
            air_time_reward = 0.0

        # symmetry (keep 0.0 for humanoids unless you define pairs)
        if self._symmetry_air_coeff > 0.0:
            symmetry_air_violations = 0.0
            if backend == np:
                if len(self._foot_ids) >= 2 and (not foots_on_ground[0] and not foots_on_ground[1]):
                    symmetry_air_violations += 1
                if len(self._foot_ids) >= 4 and (not foots_on_ground[2] and not foots_on_ground[3]):
                    symmetry_air_violations += 1
            else:
                if len(self._foot_ids) >= 2:
                    symmetry_air_violations = jax.lax.cond(jnp.logical_and(jnp.logical_not(foots_on_ground[0]),
                                                                           jnp.logical_not(foots_on_ground[1])),
                                                           lambda: symmetry_air_violations + 1,
                                                           lambda: symmetry_air_violations)
                if len(self._foot_ids) >= 4:
                    symmetry_air_violations = jax.lax.cond(jnp.logical_and(jnp.logical_not(foots_on_ground[2]),
                                                                           jnp.logical_not(foots_on_ground[3])),
                                                           lambda: symmetry_air_violations + 1,
                                                           lambda: symmetry_air_violations)
            symmetry_air_reward = self._symmetry_air_coeff * -symmetry_air_violations
        else:
            symmetry_air_reward = 0.0

        locomotion_penalties = (
            z_vel_reward + roll_pitch_vel_reward + roll_pitch_reward + joint_qpos_reward
            + joint_position_limit_reward + joint_vel_reward + acceleration_reward
            + torque_reward + air_time_reward + symmetry_air_reward + energy_reward
        )

        # Optional: velocity-goal tracking (if user enables and goal exists)
        tracking_reward = 0.0
        if self._include_vel_tracking and ("GoalRandomRootVelocity" in env.obs_container):
            goal_state_vel = getattr(carry.observation_states, "GoalRandomRootVelocity")
            # local planar velocities + yaw rate (same as TargetVelocityGoalReward)
            lin_vel_global = backend.squeeze(data.qvel[self._vel_idx])[:3]
            ang_vel_global = backend.squeeze(data.qvel[self._vel_idx])[3:]
            lin_vel_local = global_rot.as_matrix().T @ lin_vel_global
            vel_local = backend.concatenate([lin_vel_local[:2], backend.atleast_1d(ang_vel_global[2])])

            goal_vel = backend.array([goal_state_vel.goal_vel_x, goal_state_vel.goal_vel_y, goal_state_vel.goal_vel_yaw])
            tracking_xy = backend.exp(-self._tracking_w_exp_xy * backend.mean(backend.square(vel_local[:2] - goal_vel[:2])))
            tracking_yaw = backend.exp(-self._tracking_w_exp_yaw * backend.mean(backend.square(vel_local[2] - goal_vel[2])))
            tracking_reward = self._tracking_w_sum_xy * tracking_xy + self._tracking_w_sum_yaw * tracking_yaw

        # -------------------------------
        # Total reward (clamped >= 0)
        # -------------------------------
        placement_reward = swing_pos_reward + swing_orn_reward + torso_height_reward
        total_reward = placement_reward \
                       - action_rate_penalty \
                       + self._locomotion_scale * (locomotion_penalties + tracking_reward)

        total_reward = backend.maximum(total_reward, 0.0)

        # Update state
        new_state = reward_state.replace(last_qvel=data.qvel, last_action=action, time_since_last_touchdown=tslt)
        carry = carry.replace(reward_state=new_state)
        return total_reward, carry