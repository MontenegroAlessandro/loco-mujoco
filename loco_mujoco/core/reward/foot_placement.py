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
        # self._floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # assert self._foot_site_id_left != -1 and self._foot_site_id_right != -1 and self._torso_body_id != -1

        # Try to get foot body and sensor ids (for slip)
        self._left_foot_body_id = model.site_bodyid[self._foot_site_id_left]
        self._right_foot_body_id = model.site_bodyid[self._foot_site_id_right]
        # self._left_sensor_adr = None
        # self._right_sensor_adr = None
        # for name in getattr(env._model.sensor_names, [], []):
        #     if "left_foot_global_linvel" in name:
        #         sid = model.sensor(name).id
        #         self._left_sensor_adr = list(range(model.sensor_adr[sid], model.sensor_adr[sid] + model.sensor_dim[sid]))
        #     if "right_foot_global_linvel" in name:
        #         sid = model.sensor(name).id
        #         self._right_sensor_adr = list(range(model.sensor_adr[sid], model.sensor_adr[sid] + model.sensor_dim[sid]))

    def init_state(self, env, key, model, data, backend):
        reward_components = {
            "tracking/foot_position": 0.,
            "tracking/foot_orientation": 0.,
            "tracking/height": 0.,
            # "bonus/touchdown": 0.,
            # "tracking/clearance": 0.,
            # "penalties/slip": 0.,
            # "penalties/impact": 0.,
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
        # swing_body_id = jax.lax.select((swing_foot_idx == 1), self._right_foot_body_id, self._left_foot_body_id)

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
        # clearance_reward = self._clearance_w * backend.exp(
        #     -self._clearance_sharp * backend.square(backend.maximum(0.0, self._clearance_target - swing_pos[2]))
        # ) * env.dt

        # --- 4. Touchdown bonus (based on contact + accuracy)
        # foot_on_ground = mj_check_collisions(swing_body_id, self._floor_id, data, backend)
        # cfrc = backend.linalg.norm(backend.array(data.cfrc_ext[swing_body_id, :3]))
        # touchdown_reward = backend.where(
        #     backend.logical_and(foot_on_ground, cfrc > self._impact_threshold),
        #     self._touchdown_w * backend.exp(-self._touchdown_sharp * pos_err_sq) * env.dt,
        #     0.0,
        # )

        # --- 5. Slip penalty
        # if (swing_foot_idx == 0 and self._left_sensor_adr is not None) or (swing_foot_idx == 1 and self._right_sensor_adr is not None):
        #     foot_vel = (
        #         data.sensordata[self._left_sensor_adr]
        #         if swing_foot_idx == 0
        #         else data.sensordata[self._right_sensor_adr]
        #     )
        # else:
        #     foot_vel = data.site_xvelp[swing_site_id]
        # slip_penalty = self._slip_w * backend.square(backend.linalg.norm(foot_vel[:2])) * env.dt

        # --- 6. Impact penalty
        # impact_penalty = self._impact_w * backend.maximum(0.0, cfrc - self._impact_threshold) * env.dt

        # --- 7. Smoothness penalty
        action_rate_penalty = self._action_rate_w * backend.sum(backend.square(action - reward_state.last_action)) * env.dt

        # --- Combine and clip
        total_reward = (
            swing_pos_reward
            + swing_orn_reward
            + torso_height_reward
            # + clearance_reward
            # + touchdown_reward
            # - slip_penalty 
            # - impact_penalty 
            - action_rate_penalty
        )
        total_reward = backend.maximum(total_reward, 0.0)

        # --- Update components
        comps = {
            "tracking/foot_position": swing_pos_reward,
            "tracking/foot_orientation": swing_orn_reward,
            "tracking/height": torso_height_reward,
            # "bonus/touchdown": touchdown_reward,
            # "tracking/clearance": clearance_reward,
            # "penalties/slip": slip_penalty,
            # "penalties/impact": impact_penalty,
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
    

@struct.dataclass
class CrispBoosterLocomotionRewardFootPlacementState:
    """
    State of LocomotionReward.
    """
    gait_process: float
    last_qvel: Union[np.ndarray, jax.Array]
    last_action: Union[np.ndarray, jax.Array]
    time_since_last_touchdown: Union[np.ndarray, jax.Array]
    reward_components: Dict[str, Union[np.ndarray, jax.Array]]

class CrispBoosterLocomotionFootPlacementReward(Reward):
    """
    Reward function extending the FootPlacementReward with typical additional penalties
    and regularization terms for locomotion. This reward is stateful: LocomotionRewardState
    """

    def __init__(self, env: Any, **kwargs):
        """
        Initialize the reward function.

        Args:
            env (Any): The environment instance.
            **kwargs (Any): Additional keyword arguments.
        """
        super().__init__(env, **kwargs)

        model = env._model
        self._free_jnt_name = self._info_props["root_free_joint_xml_name"]

        # Initialize joint indices and masks
        self._free_joint_qpos_ind = np.array(mj_jntname2qposid(self._free_jnt_name, model))
        self._free_joint_qvel_ind = np.array(mj_jntname2qvelid(self._free_jnt_name, model))
        
        self._free_joint_qpos_mask = np.zeros(model.nq, dtype=bool)
        self._free_joint_qpos_mask[self._free_joint_qpos_ind] = True
        
        self._free_joint_qvel_mask = np.zeros(model.nv, dtype=bool)
        self._free_joint_qvel_mask[self._free_joint_qvel_ind] = True

        # Initialize floor and foot geometry IDs
        self._floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        foot_names = self._info_props["foot_geom_names"]
        
        # Get left and right foot names and IDs
        self._left_foot_names = [name for name in foot_names if "left" in name]
        self._right_foot_names = [name for name in foot_names if "right" in name]
        
        self._left_foot_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) 
            for name in self._left_foot_names
        ]
        self._right_foot_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) 
            for name in self._right_foot_names
        ]
        
        self._left_foot_body_ids = [model.geom_bodyid[foot_id] for foot_id in self._left_foot_ids]
        self._right_foot_body_ids = [model.geom_bodyid[foot_id] for foot_id in self._right_foot_ids]
        
        # Initialize foot sensor addresses
        # Adapted from: https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/h1/joystick_gait_tracking.py
        foot_sensor_adrs = []
        for foot_sensor in ['left_foot_global_linvel', 'right_foot_global_linvel']:
            sensor_id = model.sensor(foot_sensor).id
            sensor_adr = model.sensor_adr[sensor_id]
            sensor_dim = model.sensor_dim[sensor_id]
            foot_sensor_adrs.append(list(range(sensor_adr, sensor_adr + sensor_dim)))
        
        self._left_foot_sensor_adr = np.array(foot_sensor_adrs[0])
        self._right_foot_sensor_adr = np.array(foot_sensor_adrs[1])

        # Initialize foot site IDs
        self._left_foot_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        self._right_foot_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

        # Extract reward coefficients from kwargs
        self._survival = kwargs.get("survival", 0.0)

        # feet tracking weights and coefficients
        self._tracking_swing_pos_w = kwargs.get("swing_pos_w", 0.0)
        self._tracking_swing_orn_w = kwargs.get("swing_orn_w", 0.0)
        self._tracking_swing_pos_sharp = kwargs.get("sharp_pos", 0.0)
        self._tracking_swing_orn_sharp = kwargs.get("sharp_orn", 0.0)

        self._tracking_stance_pos_w = kwargs.get("stance_pos_w", 0.0)
        self._tracking_stance_orn_w = kwargs.get("stance_orn_w", 0.0)
        self._tracking_stance_pos_sharp = kwargs.get("stance_sharp_pos", 0.0)
        self._tracking_stance_orn_sharp = kwargs.get("stance_sharp_orn", 0.0)

        self._tracking_swing_pos_threshold = kwargs.get("swing_pos_threshold", 0.0)

        # Nominal posture tracking weights and coefficients
        self._nominal_joint_pos_exp = kwargs.get("tracking_nominal_joint_pos_exp", 0.0)
        self._nominal_joint_pos_coeff = kwargs.get("tracking_nominal_joint_pos_coeff", 0.0)
        self._nominal_joint_pos_names = kwargs.get("tracking_nominal_joint_pos_names", None)

        self._joint_deviation_l1_coeff = kwargs.get("joint_deviation_l1_coeff", 0.0)   
        self._base_height_coeff = kwargs.get("base_height_coeff", 0.0)
        self._base_height_target = kwargs.get("base_height_target", 0.0)
        self.orientation_coeff = kwargs.get("orientation_coeff", 0.0)

        # Torque and energy coefficients
        self._joint_torque_coeff = kwargs.get("joint_torque_coeff", 0.0)
        self._torque_tiredness_coeff = kwargs.get("torque_tiredness_coeff", 0.0)
        self._energy_coeff = kwargs.get("energy_coeff", 0.0)

        # Velocity and acceleration penalties
        self._z_vel_coeff = kwargs.get("z_vel_coeff", 0.0)
        self._roll_pitch_vel_coeff = kwargs.get("roll_pitch_vel_coeff", 0.0)
        self._joint_vel_coeff = kwargs.get("joint_vel_coeff", 0.0)
        self._joint_acc_coeff = kwargs.get("joint_acc_coeff", 0.0)
        self._root_acc_coeff = kwargs.get("root_acc_coeff", 0.0)
        self._action_rate_coeff = kwargs.get("action_rate_coeff", 0.0)
        self._low_gains_coeff = kwargs.get("low_gains_coeff", 0.0)

        # Joint position limit coefficients
        self._joint_position_limit_scale = kwargs.get("joint_position_limit_scale", 1.0)
        self._joint_position_limit_coeff = kwargs.get("joint_position_limit_coeff", 0.0)

        # Feet-related coefficients
        self._feet_slip_coeff = kwargs.get("feet_slip_coeff", 0.0)
        self._feet_yaw_diff_coeff = kwargs.get("feet_yaw_diff_coeff", 0.0)
        self._feet_yaw_mean_coeff = kwargs.get("feet_yaw_mean_coeff", 0.0)
        self._feet_roll_coeff = kwargs.get("feet_roll_coeff", 0.0)
        self._feet_distance_coeff = kwargs.get("feet_distance_coeff", 0.0)
        self._feet_distance_target = kwargs.get("feet_distance_target", 0.0)
        self._feet_swing_coeff = kwargs.get("feet_swing_coeff", 0.0)
        self._feet_swing_period = kwargs.get("feet_swing_period", 0.2)
        self._gait_height_sharp = kwargs.get("gait_height_sharp", 0.0)
        self._gait_height_coeff = kwargs.get("gait_height_coeff", 0.0)

        # Air time and impact coefficients
        self._air_time_max = kwargs.get("air_time_max", 0.0)
        self._air_time_coeff = kwargs.get("air_time_coeff", 0.0)
        self._no_fly_coeff = kwargs.get("no_fly_coeff", 0.0)
        self._symmetry_air_coeff = kwargs.get("symmetry_air_coeff", 0.0)
        self._impact_threshold = kwargs.get("impact_threshold", 0.0)
        self._impact_coeff = kwargs.get("impact_coeff", 0.0)

        # Initialize joint limits and nominal positions
        self._limited_joints = np.array(model.jnt_limited, dtype=bool)
        self._limited_joints_qpos_id = model.jnt_qposadr[np.where(self._limited_joints)]
        self._joint_ranges = model.jnt_range[self._limited_joints]
        self._nominal_joint_qpos = env._init_state_handler.qpos_init
        
        if self._nominal_joint_pos_names is None:
            # Take all limited joints
            self._nominal_joint_qpos_id = self._limited_joints_qpos_id
        else:
            self._nominal_joint_qpos_id = np.concatenate([
                mj_jntname2qposid(name, model) for name in self._nominal_joint_pos_names
            ])
        
        # Goal class name
        self._goal_name = kwargs.get("goal_name", "GoalRandomFootPlacement")

    def init_state(self, env: Any, key: Any, model: Union[MjModel, Model], 
                   data: Union[MjData, Data], backend: ModuleType):
        """
        Initialize the reward state.

        Args:
            env (Any): The environment instance.
            key (Any): Key for the reward state.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            LocomotionRewardState: The initialized reward state.
        """
        reward_components = {
            "survival_reward": 0.,
            "tracking/tracking_swing_position": 0.,
            "tracking/tracking_swing_orientation": 0.,
            "tracking/tracking_stance_position": 0.,
            "tracking/tracking_stance_orientation": 0.,
            "tracking/joint_qpos_reward": 0.,
            "tracking/feet_swing_reward": 0.,
            "penalties/joint_deviation_l1_penalty": 0.,
            "penalties/base_height_reward": 0.,
            "penalties/orientation_reward": 0.,
            "penalties/torque_reward": 0.,
            "penalties/torque_tiredness_reward": 0.,
            "penalties/energy_reward": 0.,
            "penalties/z_vel_reward": 0.,
            "penalties/roll_pitch_vel_reward": 0.,
            "penalties/joint_vel_reward": 0.,
            "penalties/acceleration_reward": 0.,
            "penalties/root_acceleration_reward": 0.,
            "penalties/action_rate_reward": 0.,
            "penalties/low_gains_reward": 0.,
            "penalties/joint_position_limit_reward": 0.,
            "penalties/feet_slip_reward": 0.,
            "penalties/feet_yaw_diff_reward": 0.,
            "penalties/feet_yaw_mean_reward": 0.,
            "penalties/feet_roll_reward": 0.,
            "penalties/feet_distance_reward": 0.,
            "penalties/air_time_reward": 0.,
            "penalties/no_fly_reward": 0.,
            "penalties/impact_reward": 0.,
        }

        return CrispBoosterLocomotionRewardFootPlacementState(
            gait_process=0.0,
            last_qvel=data.qvel, 
            last_action=backend.zeros(env.info.action_space.shape[0]),
            time_since_last_touchdown=backend.zeros(2, dtype=backend.float32),
            reward_components=reward_components
        )

    def reset(self, env: Any, model: Union[MjModel, Model], data: Union[MjData, Data], 
              carry: Any, backend: ModuleType):
        """
        Reset the reward state.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated data and carry.
        """
        reward_state = self.init_state(env, None, model, data, backend)
        carry = carry.replace(reward_state=reward_state)
        return data, carry

    def __call__(self, state: Union[np.ndarray, jnp.ndarray], action: Union[np.ndarray, jnp.ndarray],
                 next_state: Union[np.ndarray, jnp.ndarray], absorbing: bool, info: Dict[str, Any],
                 env: Any, model: Union[MjModel, Model], data: Union[MjData, Data], 
                 carry: Any, backend: ModuleType) -> Tuple[float, Any]:
        """
        Based on the tracking reward, this reward function adds typical penalties and regularization terms
        for locomotion.

        Args:
            state (Union[np.ndarray, jnp.ndarray]): Last state.
            action (Union[np.ndarray, jnp.ndarray]): Applied action.
            next_state (Union[np.ndarray, jnp.ndarray]): Current state.
            absorbing (bool): Whether the state is absorbing.
            info (Dict[str, Any]): Additional information.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[float, Any]: The reward for the current transition and the updated carry.
        """
        # Select rotation backend
        if backend == np:
            R = np_R
        else:
            R = jnp_R

        # Get current states
        reward_state = carry.reward_state
        goal_state = getattr(carry.observation_states, self._goal_name)

        # Extract global pose and velocity information
        global_pose_root = data.qpos[self._free_joint_qpos_ind]
        global_pos_root = global_pose_root[:3]
        global_quat_root = global_pose_root[3:]
        global_rot = R.from_quat(quat_scalarfirst2scalarlast(global_quat_root))
        global_vel_root = data.qvel[self._free_joint_qvel_ind]

        # Transform to local coordinates
        local_vel_root_lin = global_rot.inv().apply(global_vel_root[:3])
        local_vel_root_ang = global_rot.inv().apply(global_vel_root[3:])
        global_vel_root_ang = global_vel_root[3:]

        # ==================== REWARD COMPONENTS ====================
        
        # Survival reward
        survival_reward = 1.0

        # Goal tracking rewards
        # Goal info
        swing_foot_idx = goal_state.swing_foot_idx
        if self._goal_name in ["GoalDoubleFootPlacement", "GoalFootPlacementFromVelocity"]:
            gait_process = goal_state.gait_process
            
            # if we are in the right phase (gait_process >= 0.5) we remove 0.5
            # the quantity we consider is always in 2 * [0, 0.5]
            gait_sharpness = 2 * (gait_process - backend.where(gait_process >= 0.5, 0.5, 0))
            
            # retrieve tragets for the swing and the stance feet
            swing_target_pos, swing_target_orn, stance_target_pos, stance_target_orn = jax.lax.cond(
                swing_foot_idx == 0, # left foot swing
                lambda: (goal_state.left_foot_target_pos, goal_state.left_foot_target_orn, goal_state.right_foot_target_pos, goal_state.right_foot_target_orn),
                lambda: (goal_state.right_foot_target_pos, goal_state.right_foot_target_orn, goal_state.left_foot_target_pos, goal_state.left_foot_target_orn),
            )
            
            # retrieve current pose
            swing_curr_pos, swing_curr_orn, stance_curr_pos, stance_curr_orn = jax.lax.cond(
                swing_foot_idx == 0, # left foot swing
                lambda: (data.site_xpos[self._left_foot_site_id], data.site_xmat[self._left_foot_site_id], data.site_xpos[self._right_foot_site_id], data.site_xmat[self._right_foot_site_id]),
                lambda: (data.site_xpos[self._right_foot_site_id], data.site_xmat[self._right_foot_site_id], data.site_xpos[self._left_foot_site_id], data.site_xmat[self._left_foot_site_id]),
            )
            
            # take just the xy components for the swing positions
            swing_target_pos = swing_target_pos[:2]
            swing_curr_pos = swing_curr_pos[:2]
            
            # post-process the orientations knowing that:
            # 1. the goal is providing (w,x,y,z) 
            # 2. the data is a flattened matrix
            # 3. helper functions wants quaternions with scalar last
            # 4. we want to treat just the yaw error
            swing_target_orn = (R.from_quat(quat_scalarfirst2scalarlast(swing_target_orn))).as_euler('xyz')[2]
            stance_target_orn = (R.from_quat(quat_scalarfirst2scalarlast(stance_target_orn))).as_euler('xyz')[2]
            swing_curr_orn = (R.from_matrix(swing_curr_orn.reshape(3,3))).as_euler('xyz')[2]
            stance_curr_orn = (R.from_matrix(stance_curr_orn.reshape(3,3))).as_euler('xyz')[2]
            
            # compute errors
            swing_pos_error_sq = backend.sum(backend.square(swing_curr_pos - swing_target_pos))
            stance_pos_error_sq = backend.sum(backend.square(stance_curr_pos - stance_target_pos))
            
            def _wrap_to_pi(angle):
                return (angle + backend.pi) % (2 * backend.pi) - backend.pi
            swing_orn_error = backend.square(_wrap_to_pi(swing_target_orn - swing_curr_orn))
            stance_orn_error = backend.square(_wrap_to_pi(stance_target_orn - stance_curr_orn))
            
            # FIXME: old code start
            # Retrieve targets
            # left_target_pos = jax.lax.select(swing_foot_idx==0, goal_state.left_foot_target_pos, goal_state.left_foot_target_pos) # (x,y,z)
            # left_target_orn = goal_state.left_foot_target_orn
            # right_target_pos = jax.lax.select(swing_foot_idx==1, goal_state.right_foot_target_pos, goal_state.right_foot_target_pos) # (x,y,z)
            # right_target_orn = goal_state.right_foot_target_orn

            # # Current pose
            # left_pos = jax.lax.select(swing_foot_idx==0, data.site_xpos[self._left_foot_site_id], data.site_xpos[self._left_foot_site_id]) # (x,y,z)
            # left_orn = R.from_matrix(data.site_xmat[self._left_foot_site_id].reshape(3, 3)).as_quat(scalar_first=True)
            # right_pos = jax.lax.select(swing_foot_idx==0, data.site_xpos[self._right_foot_site_id], data.site_xpos[self._right_foot_site_id]) # (x,y,z)
            # right_orn = R.from_matrix(data.site_xmat[self._right_foot_site_id].reshape(3, 3)).as_quat(scalar_first=True)

            # # Position tracking error
            # lpos_err_sq = backend.sum(backend.square(left_pos - left_target_pos))
            # rpos_err_sq = backend.sum(backend.square(right_pos - right_target_pos))

            # # Orientation tracking error
            # # NOTE: we are going to consider just the yaw misalignment, since we accept the possibility of having 
            # # NOTE: uneven terrains
            # # target yaws
            # l_targ_yaw = (R.from_quat(quat_scalarfirst2scalarlast(left_target_orn))).as_euler('xyz')[2]
            # r_targ_yaw = (R.from_quat(quat_scalarfirst2scalarlast(right_target_orn))).as_euler('xyz')[2]
            # # current feet yaws
            # l_cur_yaw = (R.from_quat(quat_scalarfirst2scalarlast(left_orn))).as_euler('xyz')[2]
            # r_cur_yaw = R.from_quat(quat_scalarfirst2scalarlast(right_orn)).as_euler('xyz')[2]
            # # get the yaw error
            # l_yaw_err = (l_targ_yaw - l_cur_yaw + backend.pi) % (2 * backend.pi) - backend.pi
            # lorn_err = backend.square(l_yaw_err)
            # r_yaw_err = (r_targ_yaw - r_cur_yaw + backend.pi) % (2 * backend.pi) - backend.pi
            # rorn_err = backend.square(r_yaw_err)

            # # discrimate left and right foot basing on the swing idx
            # swing_left = (swing_foot_idx == 0)
            # (swing_pos_error_sq, swing_orn_error, stance_pos_error_sq, stance_orn_error) = jax.lax.cond(
            #     swing_left,
            #     lambda: (lpos_err_sq, lorn_err, rpos_err_sq, rorn_err),
            #     lambda: (rpos_err_sq, rorn_err, lpos_err_sq, lorn_err)
            # )
            # FIXME: old code end

            # NOTE: adaptive sharpness is just for the swing targets
            swing_pos_reward = self._tracking_swing_pos_w * backend.exp(-self._tracking_swing_pos_sharp * swing_pos_error_sq * gait_sharpness)
            stance_pos_reward = self._tracking_stance_pos_w * backend.exp(-self._tracking_stance_pos_sharp * stance_pos_error_sq)

            swing_orn_reward = self._tracking_swing_orn_w * backend.exp(-self._tracking_swing_orn_sharp * swing_orn_error * gait_sharpness) 
            stance_orn_reward = self._tracking_stance_orn_w * backend.exp(-self._tracking_stance_orn_sharp * stance_orn_error)
        else:
            # swing_target_pos = goal_state.swing_target_pos[:2] # just (x,y)
            swing_target_pos = goal_state.swing_target_pos # (x,y,z)
            swing_target_orn = goal_state.swing_target_orn

            # Foot IDs
            swing_site_id = jax.lax.select((swing_foot_idx == 1), self._right_foot_site_id, self._left_foot_site_id)

            # Current pose
            # swing_pos = data.site_xpos[swing_site_id][:2] # just (x,y)
            swing_pos = data.site_xpos[swing_site_id] # (x,y,z)
            swing_orn = R.from_matrix(data.site_xmat[swing_site_id].reshape(3, 3)).as_quat(scalar_first=True)

            # FIXME: track only the yaw and also the z!
            # Swing-foot tracking (pos + orn)
            pos_err_sq = backend.sum(backend.square(swing_pos - swing_target_pos))
            dot_q = backend.clip(backend.sum(swing_target_orn * swing_orn), -1.0, 1.0)
            orn_err = 1.0 - backend.square(dot_q)
            swing_pos_reward = self._tracking_swing_pos_w * backend.exp(-self._tracking_swing_pos_sharp * pos_err_sq)
            swing_orn_reward = self._tracking_swing_orn_w * backend.exp(-self._tracking_swing_orn_sharp * orn_err)
            # check if within threshold
            position_reached = (pos_err_sq <= self._tracking_swing_pos_threshold**2)
            if backend == np:
                position_reached = int(not position_reached)
            else:
                position_reached = jnp.where(position_reached, 0, 1)
            
            # for compatibility
            stance_pos_reward = 0
            stance_orn_reward = 0

        # Base height reward
        base_height_target = goal_state.goal_height
        base_height = global_pos_root[2] - env._terrain.get_height_at_xy(carry.terrain_state, global_pos_root[:2], backend)  # Assuming flat ground at z=0
        base_height_reward = backend.square(base_height - base_height_target)

        # Orientation reward
        projected_gravity = global_rot.inv().apply(backend.array([0, 0, -1]))
        orientation_reward = backend.sum(backend.square(projected_gravity[:2]))  # Penalize deviation from vertical

        # Joint torque reward
        torque_reward = backend.sum(backend.square(data.qfrc_actuator[~self._free_joint_qvel_mask]))

        # Torque tiredness reward
        torques = data.qfrc_actuator[~self._free_joint_qvel_mask]
        torque_tiredness_reward = 0.

        # Energy reward
        energy_reward = backend.sum(backend.clip(
            data.qvel[~self._free_joint_qvel_mask] * data.qfrc_actuator[~self._free_joint_qvel_mask], 
            a_min=0.0
        ))

        # Velocity penalties
        z_vel_reward = backend.square(local_vel_root_lin[2])
        roll_pitch_vel_reward = backend.square(local_vel_root_ang[:2]).sum()

        # Joint motion penalties
        joint_vel = data.qvel[~self._free_joint_qvel_mask]
        joint_vel_reward = backend.square(joint_vel).sum()

        last_joint_vel = reward_state.last_qvel[~self._free_joint_qvel_mask]
        acceleration_reward = (backend.square((joint_vel - last_joint_vel) / env.dt)).sum()

        # Root acceleration penalty
        root_acceleration_reward = backend.square(
            (global_vel_root - reward_state.last_qvel[self._free_joint_qvel_ind]) / env.dt
        ).sum()

        # Action rate penalty
        action_rate_reward = (backend.square(action - reward_state.last_action)).sum()

        # Low gains reward (incentivize gains to be close to -1)
        low_gains_reward = 0.0
        if len(action) == 2 * len(self._limited_joints_qpos_id):
            gains = action[len(self._limited_joints_qpos_id):]
            low_gains_reward = backend.sum(backend.square(gains + 1.0))

        # Joint position limit penalty
        joint_positions = backend.array(data.qpos[self._limited_joints_qpos_id])
        scale_factor = 0.5 * (1 - self._joint_position_limit_scale)
        range_diff = self._joint_ranges[:, 1] - self._joint_ranges[:, 0]
        
        lower = self._joint_ranges[:, 0] + scale_factor * range_diff
        upper = self._joint_ranges[:, 1] - scale_factor * range_diff
        joint_position_limit_reward = ((joint_positions < lower) + (joint_positions > upper)).sum() * 1.0

        # ==================== FEET-RELATED REWARDS ====================
        
        def get_feet_contact_states():
            """Check if the foot is in contact with the floor."""
            left_contacts = [
                mj_check_collisions(f_id, self._floor_id, data, backend) 
                for f_id in self._left_foot_ids
            ]
            right_contacts = [
                mj_check_collisions(f_id, self._floor_id, data, backend) 
                for f_id in self._right_foot_ids
            ]
            
            if backend == np:
                left_foot_on_ground = any(left_contacts)
                right_foot_on_ground = any(right_contacts)
                foots_on_ground = np.array([left_foot_on_ground, right_foot_on_ground])
            else:
                # JAX-compatible version
                left_foot_on_ground = (
                    jnp.logical_or.reduce(jnp.array(left_contacts)) if left_contacts 
                    else jnp.array(False)
                )
                right_foot_on_ground = (
                    jnp.logical_or.reduce(jnp.array(right_contacts)) if right_contacts 
                    else jnp.array(False)
                )
                foots_on_ground = jnp.array([left_foot_on_ground, right_foot_on_ground])
            
            return foots_on_ground

        # Feet slip reward
        left_foot_body_id = self._left_foot_body_ids[0]
        right_foot_body_id = self._right_foot_body_ids[0]
        
        left_foot_vel = data.sensordata[self._left_foot_sensor_adr]
        right_foot_vel = data.sensordata[self._right_foot_sensor_adr]
        feet_on_ground = get_feet_contact_states()
        
        feet_slip_reward = (
            backend.square(left_foot_vel[:3] * feet_on_ground[0]) + 
            backend.square(right_foot_vel[:3] * feet_on_ground[1])
        ).sum()

        # Feet yaw difference reward
        left_foot_yaw = R.from_matrix(data.site_xmat[self._left_foot_site_id]).as_euler('xyz')[2]
        left_foot_yaw = (left_foot_yaw + backend.pi) % (2 * backend.pi) - backend.pi
        
        right_foot_yaw = R.from_matrix(data.site_xmat[self._right_foot_site_id]).as_euler('xyz')[2]
        right_foot_yaw = (right_foot_yaw + backend.pi) % (2 * backend.pi) - backend.pi
        
        feet_yaw_diff_reward = backend.square(
            (left_foot_yaw - right_foot_yaw + backend.pi) % (2 * backend.pi) - backend.pi
        )

        # Feet yaw mean reward
        feet_yaw_mean = (
            (left_foot_yaw * 0.5 + right_foot_yaw * 0.5) +
            backend.pi * (backend.abs(left_foot_yaw - right_foot_yaw) > backend.pi)
        )
        base_yaw = global_rot.as_euler('xyz')[2]
        feet_yaw_mean_reward = backend.square(
            (base_yaw - feet_yaw_mean + backend.pi) % (2 * backend.pi) - backend.pi
        )

        # Feet roll reward
        left_foot_roll = R.from_matrix(data.site_xmat[self._left_foot_site_id]).as_euler('xyz')[0]
        left_foot_roll = (left_foot_roll + backend.pi) % (2 * backend.pi) - backend.pi
        
        right_foot_roll = R.from_matrix(data.site_xmat[self._right_foot_site_id]).as_euler('xyz')[0]
        right_foot_roll = (right_foot_roll + backend.pi) % (2 * backend.pi) - backend.pi
        
        feet_roll_reward = backend.square(left_foot_roll) + backend.square(right_foot_roll)

        # Feet distance reward
        left_foot_pos = data.site_xpos[self._left_foot_site_id]
        right_foot_pos = data.site_xpos[self._right_foot_site_id]
        
        feet_distance = (
            backend.cos(base_yaw) * (left_foot_pos[1] - right_foot_pos[1]) -
            backend.sin(base_yaw) * (left_foot_pos[0] - right_foot_pos[0])
        )
        feet_distance_reward = backend.clip(self._feet_distance_target - feet_distance, 0.0, 0.1)

        # =================================================FEET SWING=================================================
        # Feet swing reward
        gait_frequency = goal_state.gait_frequency
        
        # discriminate whether the gait porcess has to be computed or taken form the goal state
        if self._goal_name in ["GoalRandomChangingFootPlacement", "GoalDoubleFootPlacement", "GoalFootPlacementFromVelocity"]:
            # if we use the changing target, we have to sync to the goal gait phase
            gait_process = goal_state.gait_process
        else:
            gait_process = backend.fmod(reward_state.gait_process + env.dt * gait_frequency, 1.0)
            
        # swinging conditions
        left_swing = (
            (backend.abs(gait_process - 0.25) < 0.5 * self._feet_swing_period) & 
            (gait_frequency > 1.0e-8)
        )
        right_swing = (
            (backend.abs(gait_process - 0.75) < 0.5 * self._feet_swing_period) & 
            (gait_frequency > 1.0e-8)
        )
        
        # left_gait_height_sharpness = 1.0 # by default
        # right_gait_height_sharpness = 1.0 # by default
        if self._goal_name in ["GoalDoubleFootPlacement"]:
            # FIXME: old code start
            # # compute the exponential sharpness of the gait height tracking
            # # get feet positions
            # stance_foot_pos = jax.lax.select(
            #     gait_process < 0.5, # left swinging
            #     right_foot_pos,
            #     left_foot_pos
            # )
            # # get offset wrt world
            # swing_target_pos = jax.lax.select(
            #     gait_process < 0.5, # left swinging
            #     goal_state.left_foot_target_pos,
            #     goal_state.right_foot_target_pos
            # )
            # # desired_gait_height = env._terrain.get_height_at_xy(carry.terrain_state, stance_foot_pos[:2], backend) + goal_state.gait_height
            # desired_gait_height = swing_target_pos[2] + goal_state.gait_height
            # # compute exponential
            # left_gait_height_sharpness = self._gait_height_coeff * backend.exp(-self._gait_height_sharp * backend.square(desired_gait_height - left_foot_pos[2]))
            # right_gait_height_sharpness = self._gait_height_coeff * backend.exp(-self._gait_height_sharp * backend.square(desired_gait_height - right_foot_pos[2]))
            # FIXME: old code end
            
            # generate the conditions for the swing foot to be above the desired height
            swing_desired_gait_height = swing_target_pos[2] + goal_state.gait_height
            stance_desired_gait_height = env._terrain.get_height_at_xy(carry.terrain_state, stance_curr_pos[:2], backend)
            swing_above_desired_height = (swing_curr_pos[2] >= swing_desired_gait_height)
            stance_on_ground = (stance_curr_pos[2] <= (stance_desired_gait_height + goal_state.gait_height))
            
            l_foot_cond, r_foot_cond = jax.lax.cond(
                swing_foot_idx == 0,
                lambda: (swing_above_desired_height, stance_on_ground),
                lambda: (stance_on_ground, swing_above_desired_height)
            )
            
            feet_swing_reward = (
                (left_swing & l_foot_cond).astype(backend.float32) +
                (right_swing & r_foot_cond).astype(backend.float32)
            )
        else:
            # feet_swing_reward = (
            #     (left_swing & ~feet_on_ground[0]).astype(backend.float32) * left_gait_height_sharpness +
            #     (right_swing & ~feet_on_ground[1]).astype(backend.float32) * right_gait_height_sharpness
            # ) 
            feet_swing_reward = (
                (left_swing & ~feet_on_ground[0]).astype(backend.float32) +
                (right_swing & ~feet_on_ground[1]).astype(backend.float32)
            ) 
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # Nominal joint position rewards
        joint_qpos_reward = backend.exp(
            -1 * self._nominal_joint_pos_exp *
            backend.square(
                data.qpos[self._nominal_joint_qpos_id] - 
                self._nominal_joint_qpos[self._nominal_joint_qpos_id]
            ).sum()
        )

        joint_deviation_l1_penalty = backend.sum(backend.abs(
            data.qpos[self._nominal_joint_qpos_id] - 
            self._nominal_joint_qpos[self._nominal_joint_qpos_id]
        ))

        # ==================== AIR TIME AND IMPACT REWARDS ====================
        
        # Air time reward
        air_time_reward = 0.0
        tslt = reward_state.time_since_last_touchdown.copy()
        
        for i, _ in enumerate(["left", "right"]):
            foot_on_ground = feet_on_ground[i]
            if backend == np:
                if foot_on_ground:
                    if tslt[i] > 1e-6:  # > 0, to avoid numerical issues
                        air_time_reward += (tslt[i] - self._air_time_max)
                    tslt[i] = 0.0
                else:
                    tslt[i] += env.dt
            else:
                tslt_i, air_time_reward = jax.lax.cond(
                    foot_on_ground,
                    lambda: (0.0, air_time_reward + (tslt[i] - self._air_time_max) * (tslt[i] > 1e-6)),
                    lambda: (tslt[i] + env.dt, air_time_reward)
                )
                tslt = tslt.at[i].set(tslt_i)

        # No fly reward (penalize when both feet are off the ground)
        flying = backend.logical_and(tslt[0] > 0.0, tslt[1] > 0.0)
        no_fly_reward = flying * 1.0

        # Impact reward (penalize high impact forces at the feet)
        left_foot_contact_forces = data.cfrc_ext[self._left_foot_body_ids, :3]
        right_foot_contact_forces = data.cfrc_ext[self._right_foot_body_ids, :3]
        
        left_foot_contact_force_norm = backend.linalg.norm(left_foot_contact_forces, axis=1)
        right_foot_contact_force_norm = backend.linalg.norm(right_foot_contact_forces, axis=1)
        
        left_foot_impact = left_foot_contact_force_norm > self._impact_threshold
        right_foot_impact = right_foot_contact_force_norm > self._impact_threshold
        
        impact_reward = left_foot_impact * 1.0 + right_foot_impact * 1.0
        impact_reward = backend.mean(impact_reward)

        # Symmetry air reward (currently unused)
        symmetry_air_reward = 0.0

        # ==================== SCALE REWARDS BY COEFFICIENTS ====================
        
        survival_reward *= (self._survival * env.dt)
        swing_pos_reward *= env.dt
        swing_orn_reward *= env.dt
        stance_pos_reward *= env.dt
        stance_orn_reward *= env.dt
        joint_qpos_reward *= (self._nominal_joint_pos_coeff * env.dt)
        joint_deviation_l1_penalty *= (self._joint_deviation_l1_coeff * env.dt)
        base_height_reward *= (self._base_height_coeff * env.dt)
        orientation_reward *= (self.orientation_coeff * env.dt)
        torque_reward *= (self._joint_torque_coeff * env.dt)
        torque_tiredness_reward *= (self._torque_tiredness_coeff * env.dt)
        energy_reward *= (self._energy_coeff * env.dt)
        z_vel_reward *= (self._z_vel_coeff * env.dt)
        roll_pitch_vel_reward *= (self._roll_pitch_vel_coeff * env.dt)
        joint_vel_reward *= (self._joint_vel_coeff * env.dt)
        acceleration_reward *= (self._joint_acc_coeff * env.dt)
        root_acceleration_reward *= (self._root_acc_coeff * env.dt)
        action_rate_reward *= (self._action_rate_coeff * env.dt)
        low_gains_reward *= (self._low_gains_coeff * env.dt)
        joint_position_limit_reward *= (self._joint_position_limit_coeff * env.dt)
        feet_slip_reward *= (self._feet_slip_coeff * env.dt)
        feet_yaw_diff_reward *= (self._feet_yaw_diff_coeff * env.dt)
        feet_yaw_mean_reward *= (self._feet_yaw_mean_coeff * env.dt)
        feet_roll_reward *= (self._feet_roll_coeff * env.dt)
        feet_distance_reward *= (self._feet_distance_coeff * env.dt)
        feet_swing_reward *= (self._feet_swing_coeff * env.dt)
        air_time_reward *= (self._air_time_coeff * env.dt)
        no_fly_reward *= (self._no_fly_coeff * env.dt)
        impact_reward *= (self._impact_coeff * env.dt)

        # ==================== COMBINE REWARDS ====================
        
        tracking_reward = (
            swing_pos_reward + swing_orn_reward +
            stance_pos_reward + stance_orn_reward +
            joint_qpos_reward + feet_swing_reward
        )
        
        penalty_rewards = (
            base_height_reward + orientation_reward + torque_reward + torque_tiredness_reward +
            energy_reward + z_vel_reward + roll_pitch_vel_reward + joint_vel_reward +
            acceleration_reward + root_acceleration_reward + action_rate_reward + 
            joint_position_limit_reward + low_gains_reward + feet_slip_reward + 
            feet_yaw_diff_reward + feet_yaw_mean_reward + feet_roll_reward +
            feet_distance_reward + air_time_reward + no_fly_reward + impact_reward + 
            joint_deviation_l1_penalty
        )
        
        total_reward = survival_reward + tracking_reward + penalty_rewards
        
        # Handle NaN values
        total_reward = backend.nan_to_num(total_reward, nan=0.0)

        # ==================== UPDATE REWARD STATE ====================
        
        # Update reward state with new values
        reward_state = reward_state.replace(
            gait_process=gait_process,
            last_qvel=data.qvel, 
            last_action=action, 
            time_since_last_touchdown=tslt
        )
        
        # Update reward components dictionary
        updated_reward_components = {
            "survival_reward": survival_reward,
            "tracking/tracking_swing_position": swing_pos_reward,
            "tracking/tracking_swing_orientation": swing_orn_reward,
            "tracking/tracking_stance_position": stance_pos_reward,
            "tracking/tracking_stance_orientation": stance_orn_reward,
            "tracking/joint_qpos_reward": joint_qpos_reward,
            "tracking/feet_swing_reward": feet_swing_reward,
            "penalties/base_height_reward": base_height_reward,
            "penalties/joint_deviation_l1_penalty": joint_deviation_l1_penalty,
            "penalties/orientation_reward": orientation_reward,
            "penalties/torque_reward": torque_reward,
            "penalties/torque_tiredness_reward": torque_tiredness_reward,
            "penalties/energy_reward": energy_reward,
            "penalties/z_vel_reward": z_vel_reward,
            "penalties/roll_pitch_vel_reward": roll_pitch_vel_reward,
            "penalties/joint_vel_reward": joint_vel_reward,
            "penalties/acceleration_reward": acceleration_reward,
            "penalties/root_acceleration_reward": root_acceleration_reward,
            "penalties/action_rate_reward": action_rate_reward,
            "penalties/low_gains_reward": low_gains_reward,
            "penalties/joint_position_limit_reward": joint_position_limit_reward,
            "penalties/feet_slip_reward": feet_slip_reward,
            "penalties/feet_yaw_diff_reward": feet_yaw_diff_reward,
            "penalties/feet_yaw_mean_reward": feet_yaw_mean_reward,
            "penalties/feet_roll_reward": feet_roll_reward,
            "penalties/feet_distance_reward": feet_distance_reward,
            "penalties/air_time_reward": air_time_reward,
            "penalties/no_fly_reward": no_fly_reward,
            "penalties/impact_reward": impact_reward,
        }
        
        reward_state = reward_state.replace(reward_components=updated_reward_components)
        carry = carry.replace(reward_state=reward_state)
        
        return total_reward, carry