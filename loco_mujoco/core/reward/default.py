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


class NoReward(Reward):
    """
    A reward function that returns always 0.

    """

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
        """
        Return zero.

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
        return 0.0, carry


class TargetXVelocityReward(Reward):
    """
    Reward function that computes the reward based on the deviation from the root's
    target velocity in the x-direction.

    """
    def __init__(self, env: Any, target_velocity: float, **kwargs):
        """
        Initialize the reward function.

        Args:
            env (Any): The environment instance.
            target_velocity (float): The target velocity.
            **kwargs (Any): Additional keyword arguments.

        """
        super().__init__(env, **kwargs)
        self._target_vel = target_velocity
        root_free_joint_xml_name = self._info_props["root_free_joint_xml_name"]
        self._x_vel_idx = mj_jntname2qvelid(root_free_joint_xml_name, env._model)[0]

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
        """
        Compute the reward based on deviation from target velocity in x-direction.

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
        x_vel = backend.squeeze(data.qvel[self._x_vel_idx])
        return backend.exp(-backend.square(x_vel - self._target_vel)), carry


class TargetVelocityGoalReward(Reward):
    """
    Reward function that computes the reward based on the deviation from the goal velocity. The goal velocity is
    provided as an observation in the environment. The reward is computed as the negative exponential of the squared
    difference between the current velocity and the goal velocity. The reward is computed for the x, y, and yaw
    velocities of the root.

    """

    def __init__(self, env: Any, tracking_w_exp_xy=10.0, tracking_w_exp_yaw=10.0,
                 tracking_w_sum_xy=1.0, tracking_w_sum_yaw=1.0, **kwargs):
        """
        Initialize the reward function.

        Args:
            env (Any): The environment instance.
            tracking_w_exp_xy (float, optional): The exponential weight for xy-tracking reward.
            tracking_w_exp_yaw (float, optional): The exponential weight for yaw-tracking reward.
            **kwargs (Any): Additional keyword arguments.

        """

        super().__init__(env, **kwargs)

        self._free_jnt_name = self._info_props["root_free_joint_xml_name"]
        self._vel_idx = np.array(mj_jntname2qvelid(self._free_jnt_name, env._model))
        self._w_exp_xy = tracking_w_exp_xy
        self._w_exp_yaw = tracking_w_exp_yaw
        self._w_sum_xy = tracking_w_sum_xy
        self._w_sum_yaw = tracking_w_sum_yaw

        # find the goal velocity observation
        assert "GoalRandomRootVelocity" in env.obs_container, \
            f"GoalRandomRootVelocity is the required goal for the reward for{self.__class__.__name__}"

        super().__init__(env, **kwargs)

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
        """
        Computes a tracking reward based on the deviation from the goal velocity.Tracking is done on the x, y, and yaw
        velocities of the root.

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
        if backend == np:
            R = np_R
        else:
            R = jnp_R

        goal_state = getattr(carry.observation_states, "GoalRandomRootVelocity")

        # get root orientation
        root_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self._free_jnt_name)

        assert root_jnt_id != -1, f"Joint {self._free_jnt_name} not found in the model."
        root_jnt_qpos_start_id = model.jnt_qposadr[root_jnt_id]
        root_qpos = backend.squeeze(data.qpos[root_jnt_qpos_start_id:root_jnt_qpos_start_id+7])
        root_quat = R.from_quat(quat_scalarfirst2scalarlast(root_qpos[3:7]))

        # get current local vel of root
        lin_vel_global = backend.squeeze(data.qvel[self._vel_idx])[:3]
        ang_vel_global = backend.squeeze(data.qvel[self._vel_idx])[3:]
        lin_vel_local = root_quat.as_matrix().T @ lin_vel_global
        vel_local = backend.concatenate([lin_vel_local[:2], backend.atleast_1d(ang_vel_global[2])]) # construct vel, x, y and yaw

        # calculate tracking reward
        goal_vel = backend.array([goal_state.goal_vel_x, goal_state.goal_vel_y, goal_state.goal_vel_yaw])
        tracking_reward_xy = backend.exp(-self._w_exp_xy * backend.mean(backend.square(vel_local[:2] - goal_vel[:2])))
        tracking_reward_yaw = backend.exp(-self._w_exp_yaw * backend.mean(backend.square(vel_local[2] - goal_vel[2])))
        total_tracking = self._w_sum_xy * tracking_reward_xy + self._w_sum_yaw * tracking_reward_yaw

        return total_tracking, carry


@struct.dataclass
class LocomotionRewardState:
    """
    State of LocomotionReward.
    """
    last_qvel: Union[np.ndarray, jax.Array]
    last_action: Union[np.ndarray, jax.Array]
    time_since_last_touchdown: Union[np.ndarray, jax.Array]


class LocomotionReward(TargetVelocityGoalReward):

    """
    Reward function extending the TargetVelocityGoalReward with typical additional penalties
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
        self._free_joint_qpos_ind = np.array(mj_jntname2qposid(self._info_props["root_free_joint_xml_name"], model))
        self._free_joint_qvel_ind = np.array(mj_jntname2qvelid(self._info_props["root_free_joint_xml_name"], model))
        self._free_joint_qpos_mask = np.zeros(model.nq, dtype=bool)
        self._free_joint_qpos_mask[self._free_joint_qpos_ind] = True
        self._free_joint_qvel_mask = np.zeros(model.nv, dtype=bool)
        self._free_joint_qvel_mask[self._free_joint_qvel_ind] = True
        self._foot_names = self._info_props["foot_geom_names"]

        self._floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in self._foot_names]

        # reward coefficients
        self._z_vel_coeff = kwargs.get("z_vel_coeff", 2.0)
        self._roll_pitch_vel_coeff = kwargs.get("roll_pitch_vel_coeff", 5e-2)
        self._roll_pitch_pos_coeff = kwargs.get("roll_pitch_pos_coeff", 2e-1)
        self._nominal_joint_pos_coeff = kwargs.get("nominal_joint_pos_coeff", 0.0)
        self._nominal_joint_pos_names = kwargs.get("nominal_joint_pos_names", None)
        self._joint_position_limit_coeff = kwargs.get("joint_position_limit_coeff", 10.0)
        self._joint_vel_coeff = kwargs.get("joint_vel_coeff", 0.0)
        self._joint_acc_coeff = kwargs.get("joint_acc_coeff", 2e-7)
        self._joint_torque_coeff = kwargs.get("joint_torque_coeff", 2e-5)
        self._action_rate_coeff = kwargs.get("action_rate_coeff", 1e-2)
        self._air_time_max = kwargs.get("air_time_max", 0.0)
        self._air_time_coeff = kwargs.get("air_time_coeff", 0.0)
        self._symmetry_air_coeff = kwargs.get("symmetry_air_coeff", 0.0)
        self._energy_coeff = kwargs.get("energy_coeff", 0.0)

        # get limits and nominal joint positions
        self._limited_joints = np.array(model.jnt_limited, dtype=bool)
        self._limited_joints_qpos_id = model.jnt_qposadr[np.where(self._limited_joints)]
        self._joint_ranges = model.jnt_range[self._limited_joints]
        self._nominal_joint_qpos = env._model.qpos0
        if self._nominal_joint_pos_names is None:
            # take all limited joints
            self._nominal_joint_qpos_id = self._limited_joints_qpos_id
        else:
            self._nominal_joint_qpos_id = np.concatenate([mj_jntname2qposid(name, model)
                                                          for name in self._nominal_joint_pos_names])

    def init_state(self, env: Any,
                   key: Any,
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType):
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
        return LocomotionRewardState(last_qvel=data.qvel, last_action=backend.zeros(env.info.action_space.shape[0]),
                                     time_since_last_touchdown=backend.zeros(len(self._foot_ids)))

    def reset(self,
              env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType):
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

        if backend == np:
            R = np_R
        else:
            R = jnp_R

        # get current reward state
        reward_state = carry.reward_state

        # get global pose quantities
        global_pose_root = data.qpos[self._free_joint_qpos_ind]
        global_pos_root = global_pose_root[:3]
        global_quat_root = global_pose_root[3:]
        global_rot = R.from_quat(quat_scalarfirst2scalarlast(global_quat_root))

        # get global velocity quantities
        global_vel_root = data.qvel[self._free_joint_qvel_ind]

        # get local velocity quantities
        local_vel_root_lin = global_rot.inv().apply(global_vel_root[:3])
        local_vel_root_ang = global_rot.inv().apply(global_vel_root[3:])

        # velocity reward
        if self._z_vel_coeff > 0.0:
            z_vel_reward = self._z_vel_coeff * -(backend.square(local_vel_root_lin[2]))
        else:
            z_vel_reward = 0.0
        if self._roll_pitch_vel_coeff > 0.0:
            roll_pitch_vel_reward = self._roll_pitch_vel_coeff * -backend.square(local_vel_root_ang[:2]).sum()
        else:
            roll_pitch_vel_reward = 0.0

        # position reward
        if self._roll_pitch_pos_coeff > 0.0:
            euler = global_rot.as_euler("xyz")
            roll_pitch_reward = self._roll_pitch_pos_coeff * -backend.square(euler[:2]).sum()
        else:
            roll_pitch_reward = 0.0

        # nominal joint pos reward
        if self._nominal_joint_pos_coeff > 0.0:
            joint_qpos_reward = (self._nominal_joint_pos_coeff *
                                 -backend.square(data.qpos[self._nominal_joint_qpos_id] -
                                                 self._nominal_joint_qpos[self._nominal_joint_qpos_id]).sum())
        else:
            joint_qpos_reward = 0.0

        # joint position limit reward
        if self._joint_position_limit_coeff > 0.0:
            joint_positions = backend.array(data.qpos[self._limited_joints_qpos_id])
            lower_limit_penalty = -backend.minimum(joint_positions - self._joint_ranges[:, 0], 0.0).sum()
            upper_limit_penalty = backend.maximum(joint_positions - self._joint_ranges[:, 1], 0.0).sum()
            joint_position_limit_reward = self._joint_position_limit_coeff * -(lower_limit_penalty + upper_limit_penalty)
        else:
            joint_position_limit_reward = 0.0

        # joint velocity reward
        joint_vel = data.qvel[~self._free_joint_qvel_mask]
        if self._joint_vel_coeff > 0.0:
            joint_vel_reward = self._joint_vel_coeff * -backend.square(joint_vel).sum()
        else:
            joint_vel_reward = 0.0

        # joint acceleration reward
        if self._joint_acc_coeff > 0.0:
            last_joint_vel = reward_state.last_qvel[~self._free_joint_qvel_mask]
            acceleration_norm = backend.sum(backend.square(joint_vel - last_joint_vel) / env.dt)
            acceleration_reward = self._joint_acc_coeff * -acceleration_norm
        else:
            acceleration_reward = 0.0

        # joint torque reward
        if self._joint_torque_coeff > 0.0:
            torque_norm = backend.sum(backend.square(data.qfrc_actuator[~self._free_joint_qvel_mask]))
            torque_reward = self._joint_torque_coeff * -torque_norm
        else:
            torque_reward = 0.0

        # action rate reward
        if self._action_rate_coeff > 0.0:
            action_rate_norm = backend.sum(backend.square(action - reward_state.last_action))
            action_rate_reward = self._action_rate_coeff * -action_rate_norm
        else:
            action_rate_reward = 0.0

        # air time reward
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

        # symmetry reward
        if self._symmetry_air_coeff > 0.0:
            symmetry_air_violations = 0.0
            if backend == np:
                if (not foots_on_ground[0] and not foots_on_ground[1]):
                    symmetry_air_violations += 1
                if not foots_on_ground[2] and not foots_on_ground[3]:
                    symmetry_air_violations += 1
            else:
                symmetry_air_violations = jax.lax.cond(jnp.logical_and(jnp.logical_not(foots_on_ground[0]),
                                                                       jnp.logical_not(foots_on_ground[1])),
                                                       lambda: symmetry_air_violations + 1,
                                                       lambda: symmetry_air_violations)

                symmetry_air_violations = jax.lax.cond(jnp.logical_and(jnp.logical_not(foots_on_ground[2]),
                                                                       jnp.logical_not(foots_on_ground[3])),
                                                       lambda: symmetry_air_violations + 1,
                                                       lambda: symmetry_air_violations)

            symmetry_air_reward = self._symmetry_air_coeff * -symmetry_air_violations
        else:
            symmetry_air_reward = 0.0

        # energy reward
        if self._energy_coeff > 0.0:
            energy = backend.sum(backend.abs(joint_vel) * backend.abs(data.qfrc_actuator[~self._free_joint_qvel_mask]))
            energy_reward = self._energy_coeff * -energy
        else:
            energy_reward = 0.0

        # total reward
        tracking_reward, _ = super().__call__(state, action, next_state, absorbing, info,
                                              env, model, data, carry, backend)
        penality_rewards = (z_vel_reward + roll_pitch_vel_reward + roll_pitch_reward + joint_qpos_reward
                            + joint_position_limit_reward + joint_vel_reward + acceleration_reward
                            + torque_reward + action_rate_reward + air_time_reward
                            + symmetry_air_reward + energy_reward)
        total_reward = tracking_reward + penality_rewards
        total_reward = backend.maximum(total_reward, 0.0)

        # update reward state
        reward_state = reward_state.replace(last_qvel=data.qvel, last_action=action, time_since_last_touchdown=tslt)
        carry = carry.replace(reward_state=reward_state)

        return total_reward, carry


@struct.dataclass
class FootPlacementRewardState:
    """State for the FootPlacementReward function."""
    last_action: Union[np.ndarray, jax.Array]
    
class FootPlacementReward(Reward):
    """
    Rewards accurate swing-foot tracking and stance-foot stability (position + orientation),
    while maintaining an upright torso height and smooth control.
    """

    def __init__(
            self, env,
            left_foot_site_name: str,
            right_foot_site_name: str,
            swing_pos_w=4,
            swing_orn_w=2,
            torso_height_w=5,
            action_rate_w=1e-2,
            sharp_pos=100.0,
            sharp_orn=1200.0,
            sharp_height=50,
            **kwargs
        ):
        super().__init__(env, **kwargs)
        self.goal_name = "GoalRandomFootPlacement"

        # Weights
        self._swing_pos_w = swing_pos_w
        self._swing_orn_w = swing_orn_w
        self._torso_height_w = torso_height_w
        self._action_rate_w = action_rate_w

        # Sharpness
        self._sharp_pos = sharp_pos
        self._sharp_orn = sharp_orn
        self._sharp_height = sharp_height

        # Get target torso height from environment
        min_h, max_h = env.root_height_healthy_range
        self._target_height = (min_h + max_h) / 2.0

        # Foot sites
        self._foot_site_id_left = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, left_foot_site_name)
        self._foot_site_id_right = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, right_foot_site_name)
        self._torso_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, env.root_body_name)

        assert self._foot_site_id_left != -1
        assert self._foot_site_id_right != -1
        assert self._torso_body_id != -1

    def init_state(self, env, key, model, data, backend):
        return FootPlacementRewardState(last_action=backend.zeros(env.info.action_space.shape[0]))

    def __call__(self, state, action, next_state, absorbing, info, env, model, data, carry, backend):
        R = np_R if backend == np else jnp_R
        reward_state = carry.reward_state
        goal_state = getattr(carry.observation_states, self.goal_name)

        # Goal info
        swing_target_pos = goal_state.swing_target_pos
        swing_target_orn = goal_state.swing_target_orn
        swing_foot_idx = goal_state.swing_foot_idx

        # Foot IDs
        swing_id = jax.lax.select((swing_foot_idx == 1), self._foot_site_id_right, self._foot_site_id_left)

        # Current poses
        swing_pos = data.site_xpos[swing_id]
        swing_orn = R.from_matrix(data.site_xmat[swing_id].reshape(3, 3)).as_quat(scalar_first=True)

        # Swing tracking
        pos_err_sq = backend.sum(backend.square(swing_pos - swing_target_pos))
        swing_pos_reward = self._swing_pos_w * backend.exp(-self._sharp_pos * pos_err_sq)

        dot_q = backend.sum(swing_target_orn * swing_orn)
        # numeric safety
        dot_q = backend.clip(dot_q, -1.0, 1.0)
        orn_err = 1.0 - backend.square(dot_q)         # in [0, 1]
        swing_orn_reward = self._swing_orn_w * backend.exp(-self._sharp_orn * orn_err)

        # Torso height stability 
        torso_z = data.xpos[self._torso_body_id][2]
        height_err_sq = backend.square(torso_z - self._target_height)
        torso_height_reward = self._torso_height_w * backend.exp(-self._sharp_height * height_err_sq)

        # Smoothness penalty
        action_rate_penalty = self._action_rate_w * backend.sum(backend.square(action - reward_state.last_action))

        # Total reward
        total_reward = (
            swing_pos_reward
            + swing_orn_reward
            + torso_height_reward
            - action_rate_penalty
        )

        # Update state
        reward_state = reward_state.replace(last_action=action)
        carry = carry.replace(reward_state=reward_state)
        return total_reward, carry


