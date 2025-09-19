from types import ModuleType
from typing import Any, Dict, Tuple, Union

import mujoco
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model
from flax import struct
import numpy as np
import jax.numpy as jnp
from jax._src.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R

from loco_mujoco.core.reward.base import Reward
from loco_mujoco.core.observations.base import ObservationType
from loco_mujoco.core.utils import mj_jntname2qposid, mj_jntname2qvelid, mj_jntid2qposid, mj_jntid2qvelid, mj_check_collisions
import jax
# from loco_mujoco.core.utils import mj_jntname2qposid, mj_jntname2qvelid, mj_jntid2qposid, mj_jntid2qvelid
from loco_mujoco.core.utils.math import calculate_relative_site_quatities, quaternion_angular_distance
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast
from loco_mujoco.core.reward.utils import out_of_bounds_action_cost


def check_traj_provided(method):
    """
    Decorator to check if trajectory handler is None. Raises ValueError if not provided.
    """
    def wrapper(self, *args, **kwargs):
        env = kwargs.get('env', None) if 'env' in kwargs else args[5]  # Assumes 'env' is the 6th positional argument
        if getattr(env, "th") is None:
            raise ValueError("TrajectoryHandler not provided, but required for trajectory-based rewards.")
        return method(self, *args, **kwargs)
    return wrapper


class TrajectoryBasedReward(Reward):

    """
    Base class for trajectory-based reward functions. These reward functions require a
    trajectory handler to compute the reward.

    """

    @property
    def requires_trajectory(self) -> bool:
        return True


class TargetVelocityTrajReward(TrajectoryBasedReward):
    """
    Reward function that computes the reward based on the deviation from the trajectory velocity. The trajectory
    velocity is provided as an observation in the environment. The reward is computed as the negative exponential
    of the squared difference between the current velocity and the goal velocity. The reward is computed for the
    x, y, and yaw velocities of the root.

    """

    def __init__(self, env: Any,
                 w_exp=10.0,
                 **kwargs):
        """
        Initialize the reward function.

        Args:
            env (Any): Environment instance.
            w_exp (float, optional): Exponential weight for the reward. Defaults to 10.0.
            **kwargs (Any): Additional keyword arguments.
        """

        super().__init__(env, **kwargs)
        self._free_jnt_name = self._info_props["root_free_joint_xml_name"]
        self._free_joint_qpos_idx = np.array(mj_jntname2qposid(self._free_jnt_name, env._model))
        self._free_joint_qvel_idx = np.array(mj_jntname2qvelid(self._free_jnt_name, env._model))
        self._w_exp = w_exp

    @check_traj_provided
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
        Computes a tracking reward based on the deviation from the trajectory velocity.
        Tracking is done on the x, y, and yaw velocities of the root.

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

        Raises:
            ValueError: If trajectory handler is not provided.

        """
        if backend == np:
            R = np_R
        else:
            R = jnp_R

        def calc_local_vel(_d):
            _lin_vel_global = backend.squeeze(_d.qvel[self._free_joint_qvel_idx])[:3]
            _ang_vel_global = backend.squeeze(_d.qvel[self._free_joint_qvel_idx])[3:]
            _root_quat = R.from_quat(quat_scalarfirst2scalarlast(backend.squeeze(_d.qpos[self._free_joint_qpos_idx])[3:7]))
            _lin_vel_local = _root_quat.as_matrix().T @ _lin_vel_global
            # construct vel, x, y and yaw
            return backend.concatenate([_lin_vel_local[:2], backend.atleast_1d(_ang_vel_global[2])])

        # get root velocity from data
        vel_local = calc_local_vel(data)

        # calculate the same for the trajectory
        traj_data = env.th.traj.data.get(carry.traj_state.traj_no, carry.traj_state.subtraj_step_no, backend)
        traj_vel_local = calc_local_vel(traj_data)

        # calculate tracking reward
        tracking_reward = backend.exp(-self._w_exp*backend.mean(backend.square(vel_local - traj_vel_local)))

        # set nan values to 0
        tracking_reward = backend.nan_to_num(tracking_reward, nan=0.0)

        return tracking_reward, carry


@struct.dataclass
class MimicRewardState:
    """
    State of MimicReward.
    """
    last_qvel: Union[np.ndarray, jnp.ndarray]
    last_action: Union[np.ndarray, jnp.ndarray]


class MimicReward(TrajectoryBasedReward):
    """
    DeepMimic reward function that computes the reward based on the deviation from the trajectory. The reward is
    computed as the negative exponential of the squared difference between the current state and the trajectory state.
    The reward is computed for the joint positions, joint velocities, relative site positions,
    relative site orientations, and relative site velocities. These sites are specified in the environment properties
    and are placed at key points on the body to mimic the motion of the body.

    """

    def __init__(self, env: Any,
                 sites_for_mimic=None,
                 joints_for_mimic=None,
                 **kwargs):
        """
        Initialize the DeepMimic reward function.

        Args:
            env (Any): Environment instance.
            sites_for_mimic (List[str], optional): List of site names to mimic. Defaults to None, taking all.
            joints_for_mimic (List[str], optional): List of joint names to mimic. Defaults to None, taking all.
            **kwargs (Any): Additional keyword arguments.

        """

        super().__init__(env, **kwargs)

        # reward coefficients
        self._qpos_w_exp = kwargs.get("qpos_w_exp", 10.0)
        self._qvel_w_exp = kwargs.get("qvel_w_exp", 2.0)
        self._rpos_w_exp = kwargs.get("rpos_w_exp", 100.0)
        self._rquat_w_exp = kwargs.get("rquat_w_exp", 10.0)
        self._rvel_w_exp = kwargs.get("rvel_w_exp", 0.1)
        self._qpos_w_sum = kwargs.get("qpos_w_sum", 0.0)
        self._qvel_w_sum = kwargs.get("qvel_w_sum", 0.0)
        self._rpos_w_sum = kwargs.get("rpos_w_sum", 0.5)
        self._rquat_w_sum = kwargs.get("rquat_w_sum", 0.3)
        self._rvel_w_sum = kwargs.get("rvel_w_sum", 0.0)
        self._action_out_of_bounds_coeff = kwargs.get("action_out_of_bounds_coeff", 0.01)
        self._joint_acc_coeff = kwargs.get("joint_acc_coeff", 0.0)
        self._joint_torque_coeff = kwargs.get("joint_torque_coeff", 0.0)
        self._action_rate_coeff = kwargs.get("action_rate_coeff", 0.0)

        # get main body name of the environment
        self.main_body_name = self._info_props["upper_body_xml_name"]
        model = env._model
        self.main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.main_body_name)
        rel_site_names = self._info_props["sites_for_mimic"] if sites_for_mimic is None else sites_for_mimic
        self._rel_site_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                                       for name in rel_site_names])
        self._rel_body_ids = np.array([model.site_bodyid[site_id] for site_id in self._rel_site_ids])

        # determine qpos and qvel indices
        quat_in_qpos = []
        qpos_ind = []
        qvel_ind = []
        for i in range(model.njnt):
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joints_for_mimic is None or jnt_name in joints_for_mimic:
                qposid = mj_jntid2qposid(i, model)
                qvelid = mj_jntid2qvelid(i, model)
                qpos_ind.append(qposid)
                qvel_ind.append(qvelid)
                if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                    quat_in_qpos.append(qposid[3:])
        self._qpos_ind = np.concatenate(qpos_ind)
        self._qvel_ind = np.concatenate(qvel_ind)
        quat_in_qpos = np.concatenate(quat_in_qpos)
        self._quat_in_qpos = np.array([True if q in quat_in_qpos else False for q in self._qpos_ind])

        # calc mask for the root free joint velocities
        self._free_joint_qvel_ind = np.array(mj_jntname2qvelid(self._info_props["root_free_joint_xml_name"], model))
        self._free_joint_qvel_mask = np.zeros(model.nv, dtype=bool)
        self._free_joint_qvel_mask[self._free_joint_qvel_ind] = True

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
            MimicRewardState: The initialized reward state.

        """
        return MimicRewardState(last_qvel=data.qvel, last_action=backend.zeros(env.info.action_space.shape[0]))

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

    @check_traj_provided
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
        Computes a deep mimic tracking reward based on the deviation from the trajectory. The reward is computed as the
        negative exponential of the squared difference between the current state and the trajectory state. The reward
        is computed for the joint positions, joint velocities, relative site positions, relative site orientations, and
        relative site velocities.

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

        Raises:
            ValueError: If trajectory handler is not provided.

        """
        # get current reward state
        reward_state = carry.reward_state

        # get trajectory data
        traj_data = env.th.traj.data

        # get all quantities from trajectory
        traj_data_single = traj_data.get(carry.traj_state.traj_no, carry.traj_state.subtraj_step_no, backend)
        qpos_traj, qvel_traj = traj_data_single.qpos[self._qpos_ind], traj_data_single.qvel[self._qvel_ind]
        qpos_quat_traj = qpos_traj[self._quat_in_qpos].reshape(-1, 4)
        if len(self._rel_site_ids) > 1:
            site_rpos_traj, site_rangles_traj, site_rvel_traj =\
                calculate_relative_site_quatities(traj_data_single, self._rel_site_ids,
                                                self._rel_body_ids, model.body_rootid, backend)

        # get all quantities from the current data
        qpos, qvel = data.qpos[self._qpos_ind], data.qvel[self._qvel_ind]
        qpos_quat = qpos[self._quat_in_qpos].reshape(-1, 4)
        if len(self._rel_site_ids) > 1:
            site_rpos, site_rangles, site_rvel = (
                calculate_relative_site_quatities(data, self._rel_site_ids, self._rel_body_ids,
                                                model.body_rootid, backend))

        # calculate distances
        qpos_dist = backend.mean(backend.square(qpos[~self._quat_in_qpos] - qpos_traj[~self._quat_in_qpos]))
        qpos_dist += backend.mean(quaternion_angular_distance(qpos_quat, qpos_quat_traj, backend))
        qvel_dist = backend.mean(backend.square(qvel - qvel_traj))
        if len(self._rel_site_ids) > 1:
            rpos_dist = backend.mean(backend.square(site_rpos - site_rpos_traj))
            rangles_dist = backend.mean(backend.square(site_rangles - site_rangles_traj))
            rvel_rot_dist = backend.mean(backend.square(site_rvel[:,:3] - site_rvel_traj[:,:3]))
            rvel_lin_dist = backend.mean(backend.square(site_rvel[:,3:] - site_rvel_traj[:,3:]))

        # calculate rewards
        qpos_reward = backend.exp(-self._qpos_w_exp*qpos_dist)
        qvel_reward = backend.exp(-self._qvel_w_exp*qvel_dist)
        if len(self._rel_site_ids) > 1:
            rpos_reward = backend.exp(-self._rpos_w_exp*rpos_dist)
            rangles_reward = backend.exp(-self._rquat_w_exp*rangles_dist)
            rvel_rot_reward = backend.exp(-self._rvel_w_exp*rvel_rot_dist)
            rvel_lin_reward = backend.exp(-self._rvel_w_exp*rvel_lin_dist)

        # calculate costs
        # out of bounds action cost
        if self._action_out_of_bounds_coeff > 0.0:
            out_of_bound_reward = -out_of_bounds_action_cost(action, lower_bound=env.mdp_info.action_space.low,
                                                             upper_bound=env.mdp_info.action_space.high, backend=backend)
        else:
            out_of_bound_reward = 0.0

        # joint acceleration reward
        if self._joint_acc_coeff > 0.0:
            last_joint_vel = reward_state.last_qvel[~self._free_joint_qvel_mask]
            joint_vel = data.qvel[~self._free_joint_qvel_mask]
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

        # total penality rewards
        total_penalities = (self._action_out_of_bounds_coeff * out_of_bound_reward
                            + self._joint_acc_coeff * acceleration_reward
                            + self._joint_torque_coeff * torque_reward
                            + self._action_rate_coeff * action_rate_reward)
        total_penalities = backend.maximum(total_penalities, -1.0)

        # calculate total reward
        total_reward = (self._qpos_w_sum * qpos_reward + self._qvel_w_sum * qvel_reward)
        if len(self._rel_site_ids) > 1:
            total_reward = (total_reward
                        + self._rpos_w_sum * rpos_reward + self._rquat_w_sum * rangles_reward
                        + self._rvel_w_sum * rvel_rot_reward + self._rvel_w_sum * rvel_lin_reward)

        total_reward = total_reward + total_penalities

        # clip to positive values
        total_reward = backend.maximum(total_reward, 0.0)

        # set nan values to 0
        total_reward = backend.nan_to_num(total_reward, nan=0.0)

        # update reward state
        reward_state = reward_state.replace(last_qvel=data.qvel, last_action=action)
        carry = carry.replace(reward_state=reward_state)

        return total_reward, carry


@struct.dataclass
class CrispBoosterLocomotionRewardState:
    """
    State of LocomotionReward.
    """
    gait_process: float
    last_qvel: Union[np.ndarray, jax.Array]
    last_action: Union[np.ndarray, jax.Array]
    time_since_last_touchdown: Union[np.ndarray, jax.Array]
    reward_components: Dict[str, Union[np.ndarray, jax.Array]]


class CrispBoosterLocomotionReward(Reward):

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

        self._free_joint_qpos_ind = np.array(mj_jntname2qposid(self._free_jnt_name, model))
        self._free_joint_qvel_ind = np.array(mj_jntname2qvelid(self._free_jnt_name, model))
        self._free_joint_qpos_mask = np.zeros(model.nq, dtype=bool)
        self._free_joint_qpos_mask[self._free_joint_qpos_ind] = True
        self._free_joint_qvel_mask = np.zeros(model.nv, dtype=bool)
        self._free_joint_qvel_mask[self._free_joint_qvel_ind] = True


        self._floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        foot_names = self._info_props["foot_geom_names"]
        # get left foot and right foot names
        self._left_foot_names = [name for name in foot_names if "left" in name]
        self._right_foot_names = [name for name in foot_names if "right" in name]
        self._left_foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in self._left_foot_names]
        self._right_foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in self._right_foot_names]
        self._left_foot_body_ids = [model.geom_bodyid[foot_id] for foot_id in self._left_foot_ids]
        self._right_foot_body_ids = [model.geom_bodyid[foot_id] for foot_id in self._right_foot_ids]
        
        # the following block is adapted from
        # https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/h1/joystick_gait_tracking.py
        foot_sensor_adrs = []
        for foot_sensor in ['left_foot_global_linvel', 'right_foot_global_linvel']:
            sensor_id = model.sensor(foot_sensor).id
            sensor_adr = model.sensor_adr[sensor_id]
            sensor_dim = model.sensor_dim[sensor_id]
            foot_sensor_adrs.append(list(range(sensor_adr, sensor_adr + sensor_dim)))
        self._left_foot_sensor_adr = np.array(foot_sensor_adrs[0])
        self._right_foot_sensor_adr = np.array(foot_sensor_adrs[1])

        self._left_foot_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        self._right_foot_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

        # reward coefficients

        self._survival = kwargs.get("survival", 0.0)

        # velocity tracking weights and coefficients
        self._tracking_w_exp_linvel_x = kwargs.get("tracking_w_exp_linvel_x", 0.0)
        self._tracking_w_sum_linvel_x = kwargs.get("tracking_w_sum_linvel_x", 0.0)
        self._tracking_w_exp_linvel_y = kwargs.get("tracking_w_exp_linvel_y", 0.0)
        self._tracking_w_sum_linvel_y = kwargs.get("tracking_w_sum_linvel_y", 0.0)
        self._tracking_w_exp_angvel = kwargs.get("tracking_w_exp_angvel", 0.0)
        self._tracking_w_sum_angvel = kwargs.get("tracking_w_sum_angvel", 0.0)

        # nominal posture tracking weights and coefficients
        self._nominal_joint_pos_exp = kwargs.get("tracking_nominal_joint_pos_exp", 0.0)
        self._nominal_joint_pos_coeff = kwargs.get("tracking_nominal_joint_pos_coeff", 0.0)
        self._nominal_joint_pos_names = kwargs.get("tracking_nominal_joint_pos_names", None)

        self._joint_deviation_l1_coeff = kwargs.get("joint_deviation_l1_coeff", 0.0)   

        self._base_height_coeff = kwargs.get("base_height_coeff", 0.0)
        self._base_height_target = kwargs.get("base_height_target", 0.0)

        self.orientation_coeff = kwargs.get("orientation_coeff", 0.0)

        self._joint_torque_coeff = kwargs.get("joint_torque_coeff", 0.0)
        self._torque_tiredness_coeff = kwargs.get("torque_tiredness_coeff", 0.0)
        self._energy_coeff = kwargs.get("energy_coeff", 0.0)

        self._z_vel_coeff = kwargs.get("z_vel_coeff", 0.0)
        self._roll_pitch_vel_coeff = kwargs.get("roll_pitch_vel_coeff", 0.0)
        self._joint_vel_coeff = kwargs.get("joint_vel_coeff", 0.0)
        self._joint_acc_coeff = kwargs.get("joint_acc_coeff", 0.0)
        self._root_acc_coeff = kwargs.get("root_acc_coeff", 0.0)
        self._action_rate_coeff = kwargs.get("action_rate_coeff", 0.0)
        self._low_gains_coeff = kwargs.get("low_gains_coeff", 0.0)
        self._joint_position_limit_scale = kwargs.get("joint_position_limit_scale", 1.0)
        self._joint_position_limit_coeff = kwargs.get("joint_position_limit_coeff", 0.0)
        self._feet_slip_coeff = kwargs.get("feet_slip_coeff", 0.0)
        self._feet_yaw_diff_coeff = kwargs.get("feet_yaw_diff_coeff", 0.0)
        self._feet_yaw_mean_coeff = kwargs.get("feet_yaw_mean_coeff", 0.0)
        self._feet_roll_coeff = kwargs.get("feet_roll_coeff", 0.0)
        self._feet_distance_coeff = kwargs.get("feet_distance_coeff", 0.0)
        self._feet_distance_target = kwargs.get("feet_distance_target", 0.0)
        self._feet_swing_coeff = kwargs.get("feet_swing_coeff", 0.0)
        self._feet_swing_period = kwargs.get("feet_swing_period", 0.2)

        self._air_time_max = kwargs.get("air_time_max", 0.0)
        self._air_time_coeff = kwargs.get("air_time_coeff", 0.0)
        self._no_fly_coeff = kwargs.get("no_fly_coeff", 0.0)
        self._symmetry_air_coeff = kwargs.get("symmetry_air_coeff", 0.0)
        self._impact_threshold = kwargs.get("impact_threshold", 0.0)
        self._impact_coeff = kwargs.get("impact_coeff", 0.0)

        # get limits and nominal joint positions
        self._limited_joints = np.array(model.jnt_limited, dtype=bool)
        self._limited_joints_qpos_id = model.jnt_qposadr[np.where(self._limited_joints)]
        self._joint_ranges = model.jnt_range[self._limited_joints]
        # self._nominal_joint_qpos = env._model.qpos0
        self._nominal_joint_qpos = env._init_state_handler.qpos_init
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
        return CrispBoosterLocomotionRewardState(
                                    gait_process=0.0,
                                    last_qvel=data.qvel, last_action=backend.zeros(env.info.action_space.shape[0]),
                                    time_since_last_touchdown=backend.zeros(2, dtype=backend.float32),
                                    reward_components={
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
                                     )

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

        goal_state = getattr(carry.observation_states, "GoalChangingRandomRootVelocity")

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
        global_vel_root_ang = global_vel_root[3:]

        # survival reward
        survival_reward = 1.0

        # goal tracking reward
        goal_vel = backend.array([goal_state.goal_vel_x, goal_state.goal_vel_y, goal_state.goal_vel_yaw])
        tracking_reward_linvel_x = backend.exp(-backend.square(local_vel_root_lin[0] - goal_vel[0]) * self._tracking_w_exp_linvel_x)
        tracking_reward_linvel_y = backend.exp(-backend.square(local_vel_root_lin[1] - goal_vel[1]) * self._tracking_w_exp_linvel_y)
        tracking_reward_angvel = backend.exp(-backend.square(local_vel_root_ang[2] - goal_vel[2]) * self._tracking_w_exp_angvel)

        base_height_target = goal_state.goal_height
        # base height reward
        base_height = global_pos_root[2] - 0 # assuming flat ground at z=0
        base_height_reward = backend.square(base_height - base_height_target)
        # also do a tracking reward version
        # base_height_reward = backend.exp(-10. * backend.square(base_height - base_height_target))

        # orientation reward
        projected_gravity = global_rot.inv().apply(backend.array([0, 0, -1]))
        orientation_reward = backend.sum(backend.square(projected_gravity[:2]))  # penalize deviation from vertical

        # joint torque reward
        torque_reward = backend.sum(backend.square(data.qfrc_actuator[~self._free_joint_qvel_mask]))

        # torque tiredness reward
        torques = data.qfrc_actuator[~self._free_joint_qvel_mask]
        torque_tiredness_reward = 0.

        # energy reward
        energy_reward = backend.sum(backend.abs(data.qvel[~self._free_joint_qvel_mask]) * backend.abs(data.qfrc_actuator[~self._free_joint_qvel_mask]))

        # z linear velocity reward
        z_vel_reward = backend.square(local_vel_root_lin[2])

        # roll pitch velocity reward
        roll_pitch_vel_reward = backend.square(local_vel_root_ang[:2]).sum()

        # joint velocity reward
        joint_vel = data.qvel[~self._free_joint_qvel_mask]
        joint_vel_reward = backend.square(joint_vel).sum()

        # joint acceleration reward
        last_joint_vel = reward_state.last_qvel[~self._free_joint_qvel_mask]
        acceleration_reward = (backend.square((joint_vel - last_joint_vel) / env.dt)).sum()

        # root acceleration reward (linear and angular)
        root_acceleration_reward = backend.square((global_vel_root - reward_state.last_qvel[self._free_joint_qvel_ind]) / env.dt).sum()

        # action rate reward
        action_rate_reward = (backend.square(action - reward_state.last_action)).sum()

        # low gains reward. (incentivize the gains to be close to -1)
        low_gains_reward = 0.0
        if len(action) == 2 * len(self._limited_joints_qpos_id):
            # get the gains from the action space
            gains = action[len(self._limited_joints_qpos_id):]
            # calculate the low gains reward
            low_gains_reward = backend.sum(backend.square(gains + 1.0))  # penalize gains that are not close to -1

        # joint position limit reward
        joint_positions = backend.array(data.qpos[self._limited_joints_qpos_id])
        # lower_limit_penalty = -backend.minimum(joint_positions - self._joint_ranges[:, 0], 0.0).sum()
        # upper_limit_penalty = backend.maximum(joint_positions - self._joint_ranges[:, 1], 0.0).sum()
        # joint_position_limit_reward = self._joint_position_limit_coeff * -(lower_limit_penalty + upper_limit_penalty)
        # indicator penalty: for each dim that exceeds the limit, one
        lower = self._joint_ranges[:, 0] + 0.5 * (1-self._joint_position_limit_scale) * (self._joint_ranges[:, 1] - self._joint_ranges[:, 0])
        upper = self._joint_ranges[:, 1] - 0.5 * (1-self._joint_position_limit_scale) * (self._joint_ranges[:, 1] - self._joint_ranges[:, 0])
        joint_position_limit_reward = ((joint_positions < lower) + (joint_positions > upper)).sum() * 1.0
        

        def get_feet_contact_states():
            """
            Check if the foot is in contact with the floor.
            """
            left_contacts = [mj_check_collisions(f_id, self._floor_id, data, backend) for f_id in self._left_foot_ids]
            right_contacts = [mj_check_collisions(f_id, self._floor_id, data, backend) for f_id in self._right_foot_ids]
            
            if backend == np:
                left_foot_on_ground = any(left_contacts)
                right_foot_on_ground = any(right_contacts)
                foots_on_ground = np.array([left_foot_on_ground, right_foot_on_ground])
            else:
                # JAX-compatible version using jnp.logical_or.reduce
                left_foot_on_ground = jnp.logical_or.reduce(jnp.array(left_contacts)) if left_contacts else jnp.array(False)
                right_foot_on_ground = jnp.logical_or.reduce(jnp.array(right_contacts)) if right_contacts else jnp.array(False)
                foots_on_ground = jnp.array([left_foot_on_ground, right_foot_on_ground])
            return foots_on_ground
        

        # feet slip reward
        left_foot_body_id = self._left_foot_body_ids[0]
        right_foot_body_id = self._right_foot_body_ids[0]
        # get cartesian velocities of the feet (in world frame, NOT body frame! otherwise we will penalize swaying in place)
        left_foot_vel = data.sensordata[self._left_foot_sensor_adr]
        right_foot_vel = data.sensordata[self._right_foot_sensor_adr]
        # get contact state of the feet
        feet_on_ground = get_feet_contact_states()
        feet_slip_reward = (backend.square(left_foot_vel[:3] * feet_on_ground[0]) + backend.square(right_foot_vel[:3] * feet_on_ground[1])).sum()

        # feet yaw diff reward
        # convert from xmat to quat
        left_foot_yaw = R.from_matrix(data.site_xmat[self._left_foot_site_id]).as_euler('xyz')[2]
        left_foot_yaw = (left_foot_yaw + backend.pi) % (2 * backend.pi) - backend.pi
        right_foot_yaw = R.from_matrix(data.site_xmat[self._right_foot_site_id]).as_euler('xyz')[2]
        right_foot_yaw = (right_foot_yaw + backend.pi) % (2 * backend.pi) - backend.pi
        feet_yaw_diff_reward = backend.square(  (left_foot_yaw - right_foot_yaw + backend.pi) % (2 * backend.pi) - backend.pi  )

        # feet yaw mean reward
        feet_yaw_mean = (left_foot_yaw  * .5 + right_foot_yaw * .5) \
                        + backend.pi * (backend.abs(left_foot_yaw - right_foot_yaw) > backend.pi)
        base_yaw = global_rot.as_euler('xyz')[2]
        feet_yaw_mean_reward = backend.square(  (base_yaw - feet_yaw_mean + backend.pi) % (2 * backend.pi) - backend.pi  )

        # feet roll reward
        left_foot_roll = R.from_matrix(data.site_xmat[self._left_foot_site_id]).as_euler('xyz')[0]
        left_foot_roll = (left_foot_roll + backend.pi) % (2 * backend.pi) - backend.pi
        right_foot_roll = R.from_matrix(data.site_xmat[self._right_foot_site_id]).as_euler('xyz')[0]
        right_foot_roll = (right_foot_roll + backend.pi) % (2 * backend.pi) - backend.pi
        feet_roll_reward = backend.square(left_foot_roll) + backend.square(right_foot_roll)
        # feet_roll_reward = 0.0

        # feet distance reward
        left_foot_pos = data.site_xpos[self._left_foot_site_id]
        right_foot_pos = data.site_xpos[self._right_foot_site_id]
        feet_distance = (
            backend.cos(base_yaw) * (left_foot_pos[1] - right_foot_pos[1]) -
            backend.sin(base_yaw) * (left_foot_pos[0] - right_foot_pos[0])
        )
        feet_distance_reward = backend.clip(self._feet_distance_target - feet_distance, 0.0, 0.1)

        # feet swing reward
        gait_frequency = goal_state.gait_frequency
        gait_process = backend.fmod(reward_state.gait_process + env.dt * gait_frequency, 1.0)
        # still_command = backend.abs(goal_vel).sum() < 1e-2
        left_swing = (backend.abs(gait_process - 0.25) < 0.5 * self._feet_swing_period) & (gait_frequency > 1.0e-8)
        right_swing = (backend.abs(gait_process - 0.75) < 0.5 * self._feet_swing_period) & (gait_frequency > 1.0e-8)
        feet_swing_reward = (left_swing & ~feet_on_ground[0]).astype(backend.float32) + \
                             (right_swing & ~feet_on_ground[1]).astype(backend.float32)
        
        # nominal joint pos rewards
        joint_qpos_reward = backend.exp(-1 * self._nominal_joint_pos_exp *
            backend.square(data.qpos[self._nominal_joint_qpos_id] - self._nominal_joint_qpos[self._nominal_joint_qpos_id]).sum()
        )

        joint_deviation_l1_penalty = backend.sum(backend.abs(data.qpos[self._nominal_joint_qpos_id] - self._nominal_joint_qpos[self._nominal_joint_qpos_id]))

        # air time reward
        air_time_reward = 0.0
        tslt = reward_state.time_since_last_touchdown.copy()
        for i, _ in enumerate(["left", "right"]):
            foot_on_ground = feet_on_ground[i]
            if backend == np:
                if foot_on_ground:
                    if tslt[i] > 1e-6: # > 0, to avoid numerical issues
                        air_time_reward += (tslt[i] - self._air_time_max)
                    tslt[i] = 0.0
                else:
                    tslt[i] += env.dt
            else:

                tslt_i, air_time_reward = jax.lax.cond(foot_on_ground,
                                    lambda: (0.0, air_time_reward + (tslt[i] - self._air_time_max) * (tslt[i] > 1e-6)),
                                    lambda: (tslt[i] + env.dt, air_time_reward))
                tslt = tslt.at[i].set(tslt_i)
        
        air_time_reward = air_time_reward

        # no fly reward, only works if air_time_reward is nonzero (okay to make air_time_reward very small)
        # check tslt, if both feet are off the ground, the agent is flying
        # one-liner to check if both feet are off the ground
        flying = backend.logical_and(tslt[0] > 0.0, tslt[1] > 0.0)
        # if flying, add a penalty
        no_fly_reward = flying * 1.0

        # impact reward (penalize high impact forces at the feet)
        # get the contact forces at the feet
        left_foot_contact_forces = data.cfrc_ext[self._left_foot_body_ids, :3]
        right_foot_contact_forces = data.cfrc_ext[self._right_foot_body_ids, :3]
        # calculate the norm of the contact forces
        left_foot_contact_force_norm = backend.linalg.norm(left_foot_contact_forces, axis=1)
        right_foot_contact_force_norm = backend.linalg.norm(right_foot_contact_forces, axis=1)
        # check if the contact forces are above the impact threshold. binary indicator
        left_foot_impact = left_foot_contact_force_norm > self._impact_threshold
        right_foot_impact = right_foot_contact_force_norm > self._impact_threshold
        # if either foot has an impact, we penalize the agent
        # impact_reward = backend.where(backend.logical_or(left_foot_impact, right_foot_impact),
        #                               -1.0, 0.0)
        impact_reward = left_foot_impact * 1.0 + right_foot_impact * 1.0
        impact_reward = backend.mean(impact_reward)
        # # calculate the impact reward
        # impact_reward = -backend.mean(backend.concatenate([left_foot_contact_force_norm, right_foot_contact_force_norm]))

        # # symmetry reward
        # if self._symmetry_air_coeff > 0.0:
        #     symmetry_air_violations = 0.0
        #     if backend == np:
        #         if (not foots_on_ground[0] and not foots_on_ground[1]):
        #             symmetry_air_violations += 1
        #         if not foots_on_ground[2] and not foots_on_ground[3]:
        #             symmetry_air_violations += 1
        #     else:
        #         symmetry_air_violations = jax.lax.cond(jnp.logical_and(jnp.logical_not(foots_on_ground[0]),
        #                                                                jnp.logical_not(foots_on_ground[1])),
        #                                                lambda: symmetry_air_violations + 1,
        #                                                lambda: symmetry_air_violations)

        #         symmetry_air_violations = jax.lax.cond(jnp.logical_and(jnp.logical_not(foots_on_ground[2]),
        #                                                                jnp.logical_not(foots_on_ground[3])),
        #                                                lambda: symmetry_air_violations + 1,
        #                                                lambda: symmetry_air_violations)

        #     symmetry_air_reward = self._symmetry_air_coeff * -symmetry_air_violations
        # else:
        #     symmetry_air_reward = 0.0

        symmetry_air_reward = 0.0


        # total reward
        # tracking_reward, _ = super().__call__(state, action, next_state, absorbing, info,
        #                                       env, model, data, carry, backend)
        
        # now scale everything by the coefficients
        survival_reward *= (self._survival * env.dt)
        tracking_reward_linvel_x *= (self._tracking_w_sum_linvel_x * env.dt)
        tracking_reward_linvel_y *= (self._tracking_w_sum_linvel_y * env.dt)
        tracking_reward_angvel *= (self._tracking_w_sum_angvel * env.dt)
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

        tracking_reward = (
            tracking_reward_linvel_x + tracking_reward_linvel_y + tracking_reward_angvel
            + joint_qpos_reward + feet_swing_reward
        )
        penalty_rewards = (
            base_height_reward + orientation_reward + torque_reward + torque_tiredness_reward
            + energy_reward + z_vel_reward + roll_pitch_vel_reward + joint_vel_reward
            + acceleration_reward + root_acceleration_reward + action_rate_reward 
            + joint_position_limit_reward + low_gains_reward
            + feet_slip_reward + feet_yaw_diff_reward + feet_yaw_mean_reward + feet_roll_reward
            + feet_distance_reward 
            + air_time_reward + no_fly_reward + impact_reward + joint_deviation_l1_penalty
        )
        total_reward = (
            survival_reward + tracking_reward + penalty_rewards
        )
        # nan
        total_reward = backend.nan_to_num(total_reward, nan=0.0)
        # total_reward = backend.maximum(total_reward, 0.0) # keep it non-negative

        # update reward state
        reward_state = reward_state.replace(
            gait_process=gait_process,
            last_qvel=data.qvel, last_action=action, time_since_last_touchdown=tslt)
        # add all the ingredients of the reward to the reward state (under the key "reward_components")
        reward_state = reward_state.replace(reward_components={
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
        })
        carry = carry.replace(reward_state=reward_state)
        return total_reward, carry