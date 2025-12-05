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
class HumanoidLocomotionRewardState:
    """
    State of HumanoidLocomotionReward.
    """
    gait_process: float
    last_qvel: Union[np.ndarray, jax.Array]
    last_action: Union[np.ndarray, jax.Array]
    time_since_last_touchdown: Union[np.ndarray, jax.Array]
    reward_components: Dict[str, Union[np.ndarray, jax.Array]]


class HumanoidLocomotionReward(Reward):
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

        # Velocity tracking weights and coefficients
        self._tracking_w_exp_linvel_x = kwargs.get("tracking_w_exp_linvel_x", 0.0)
        self._tracking_w_sum_linvel_x = kwargs.get("tracking_w_sum_linvel_x", 0.0)
        self._tracking_w_exp_linvel_y = kwargs.get("tracking_w_exp_linvel_y", 0.0)
        self._tracking_w_sum_linvel_y = kwargs.get("tracking_w_sum_linvel_y", 0.0)
        self._tracking_w_exp_angvel = kwargs.get("tracking_w_exp_angvel", 0.0)
        self._tracking_w_sum_angvel = kwargs.get("tracking_w_sum_angvel", 0.0)

        # Nominal posture tracking weights and coefficients
        self._nominal_joint_pos_exp = kwargs.get("tracking_nominal_joint_pos_exp", 0.0)
        self._nominal_joint_pos_coeff = kwargs.get("tracking_nominal_joint_pos_coeff", 0.0)
        self._nominal_joint_pos_names = kwargs.get("tracking_nominal_joint_pos_names", None)

        self._joint_deviation_l1_coeff = kwargs.get("joint_deviation_l1_coeff", 0.0)   
        self._base_height_coeff = kwargs.get("base_height_coeff", 0.0)
        # self._base_height_target = kwargs.get("base_height_target", 0.0)
        self.orientation_coeff = kwargs.get("orientation_coeff", 0.0)

        # Torque and energy coefficients
        self._joint_torque_coeff = kwargs.get("joint_torque_coeff", 0.0)
        self._energy_coeff = kwargs.get("energy_coeff", 0.0)

        # Velocity and acceleration penalties
        self._z_vel_coeff = kwargs.get("z_vel_coeff", 0.0)
        self._roll_pitch_vel_coeff = kwargs.get("roll_pitch_vel_coeff", 0.0)
        self._joint_vel_coeff = kwargs.get("joint_vel_coeff", 0.0)
        self._joint_acc_coeff = kwargs.get("joint_acc_coeff", 0.0)
        self._root_acc_coeff = kwargs.get("root_acc_coeff", 0.0)
        self._action_rate_coeff = kwargs.get("action_rate_coeff", 0.0)

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
            "tracking/tracking_reward_linvel_x": 0.,
            "tracking/tracking_reward_linvel_y": 0.,
            "tracking/tracking_reward_angvel": 0.,
            "tracking/joint_qpos_reward": 0.,
            "tracking/feet_swing_reward": 0.,
            "penalties/joint_deviation_l1_penalty": 0.,
            "penalties/base_height_reward": 0.,
            "penalties/orientation_reward": 0.,
            "penalties/torque_reward": 0.,
            "penalties/energy_reward": 0.,
            "penalties/z_vel_reward": 0.,
            "penalties/roll_pitch_vel_reward": 0.,
            "penalties/joint_vel_reward": 0.,
            "penalties/acceleration_reward": 0.,
            "penalties/root_acceleration_reward": 0.,
            "penalties/action_rate_reward": 0.,
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

        return HumanoidLocomotionRewardState(
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
        goal_state = getattr(carry.observation_states, "GoalRandomRootVelocityAndPhase")

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
        goal_vel = backend.array([goal_state.goal_vel_x, goal_state.goal_vel_y, goal_state.goal_vel_yaw])
        
        tracking_reward_linvel_x = backend.exp(
            -backend.square(local_vel_root_lin[0] - goal_vel[0]) * self._tracking_w_exp_linvel_x
        )
        tracking_reward_linvel_y = backend.exp(
            -backend.square(local_vel_root_lin[1] - goal_vel[1]) * self._tracking_w_exp_linvel_y
        )
        tracking_reward_angvel = backend.exp(
            -backend.square(local_vel_root_ang[2] - goal_vel[2]) * self._tracking_w_exp_angvel
        )

        # Base height reward
        base_height_target = goal_state.goal_height
        base_height = global_pos_root[2] - 0  # Assuming flat ground at z=0
        base_height_reward = backend.square(base_height - base_height_target)

        # Orientation reward
        projected_gravity = global_rot.inv().apply(backend.array([0, 0, -1]))
        orientation_reward = backend.sum(backend.square(projected_gravity[:2]))  # Penalize deviation from vertical

        # Joint torque reward
        torque_reward = backend.sum(backend.square(data.qfrc_actuator[~self._free_joint_qvel_mask]))

        # Torque tiredness reward
        torques = data.qfrc_actuator[~self._free_joint_qvel_mask]

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

        # Feet swing reward
        gait_frequency = goal_state.gait_frequency
        gait_process = backend.fmod(reward_state.gait_process + env.dt * gait_frequency, 1.0)
        
        left_swing = (
            (backend.abs(gait_process - 0.25) < 0.5 * self._feet_swing_period) & 
            (gait_frequency > 1.0e-8)
        )
        right_swing = (
            (backend.abs(gait_process - 0.75) < 0.5 * self._feet_swing_period) & 
            (gait_frequency > 1.0e-8)
        )
        
        feet_swing_reward = (
            (left_swing & ~feet_on_ground[0]).astype(backend.float32) +
            (right_swing & ~feet_on_ground[1]).astype(backend.float32)
        )

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
        tracking_reward_linvel_x *= (self._tracking_w_sum_linvel_x * env.dt)
        tracking_reward_linvel_y *= (self._tracking_w_sum_linvel_y * env.dt)
        tracking_reward_angvel *= (self._tracking_w_sum_angvel * env.dt)
        joint_qpos_reward *= (self._nominal_joint_pos_coeff * env.dt)
        joint_deviation_l1_penalty *= (self._joint_deviation_l1_coeff * env.dt)
        base_height_reward *= (self._base_height_coeff * env.dt)
        orientation_reward *= (self.orientation_coeff * env.dt)
        torque_reward *= (self._joint_torque_coeff * env.dt)
        energy_reward *= (self._energy_coeff * env.dt)
        z_vel_reward *= (self._z_vel_coeff * env.dt)
        roll_pitch_vel_reward *= (self._roll_pitch_vel_coeff * env.dt)
        joint_vel_reward *= (self._joint_vel_coeff * env.dt)
        acceleration_reward *= (self._joint_acc_coeff * env.dt)
        root_acceleration_reward *= (self._root_acc_coeff * env.dt)
        action_rate_reward *= (self._action_rate_coeff * env.dt)
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
            tracking_reward_linvel_x + tracking_reward_linvel_y + tracking_reward_angvel +
            joint_qpos_reward + feet_swing_reward
        )
        
        penalty_rewards = (
            base_height_reward + orientation_reward + torque_reward + 
            energy_reward + z_vel_reward + roll_pitch_vel_reward + joint_vel_reward +
            acceleration_reward + root_acceleration_reward + action_rate_reward + 
            joint_position_limit_reward + feet_slip_reward + 
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
            "tracking/tracking_reward_linvel_x": tracking_reward_linvel_x,
            "tracking/tracking_reward_linvel_y": tracking_reward_linvel_y,
            "tracking/tracking_reward_angvel": tracking_reward_angvel,
            "tracking/joint_qpos_reward": joint_qpos_reward,
            "tracking/feet_swing_reward": feet_swing_reward,
            "penalties/base_height_reward": base_height_reward,
            "penalties/joint_deviation_l1_penalty": joint_deviation_l1_penalty,
            "penalties/orientation_reward": orientation_reward,
            "penalties/torque_reward": torque_reward,
            "penalties/energy_reward": energy_reward,
            "penalties/z_vel_reward": z_vel_reward,
            "penalties/roll_pitch_vel_reward": roll_pitch_vel_reward,
            "penalties/joint_vel_reward": joint_vel_reward,
            "penalties/acceleration_reward": acceleration_reward,
            "penalties/root_acceleration_reward": root_acceleration_reward,
            "penalties/action_rate_reward": action_rate_reward,
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