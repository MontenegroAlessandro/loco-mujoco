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
from loco_mujoco.core.terrain import AdaPillarsTerrain

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
    # new state components
    last_swing_pos_reward: float
    last_swing_orn_reward: float
    stance_pos_reward: float
    stance_orn_reward: float
    last_swing_z_reward: float
    stance_z_reward: float
    swing_foot_idx: int

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

        # left and right knee names ids
        # TODO: make it more general by not hardcoding the names
        self._left_knee_site_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_knee_mimic")
        self._right_knee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_knee_mimic")
        
        self._left_foot_body_ids = [model.geom_bodyid[foot_id] for foot_id in self._left_foot_ids]
        self._right_foot_body_ids = [model.geom_bodyid[foot_id] for foot_id in self._right_foot_ids]
        
        # Initialize foot sensor addresses
        # Adapted from: https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/h1/joystick_gait_tracking.py
        foot_sensor_adrs = []
        # for foot_sensor in ['left_foot_global_linvel', 'right_foot_global_linvel', 'left_foot_global_angvel', 'right_foot_global_angvel']:
        for foot_sensor in ['left_foot_global_linvel', 'right_foot_global_linvel']:
            sensor_id = model.sensor(foot_sensor).id
            sensor_adr = model.sensor_adr[sensor_id]
            sensor_dim = model.sensor_dim[sensor_id]
            foot_sensor_adrs.append(list(range(sensor_adr, sensor_adr + sensor_dim)))
        
        self._left_foot_sensor_adr = np.array(foot_sensor_adrs[0])
        self._right_foot_sensor_adr = np.array(foot_sensor_adrs[1])
        # self._left_foot_sensor_adr_ang = np.array(foot_sensor_adrs[2])
        # self._right_foot_sensor_adr_ang = np.array(foot_sensor_adrs[3])

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
        self._tracking_stancetracking_orn_sharp = kwargs.get("stance_sharp_orn", 0.0)

        self._tracking_swing_pos_threshold = kwargs.get("swing_pos_threshold", 0.0)
        
        # Feet height terms
        # terms ensuring the height at the mid of the swinging phase is resepcted
        self._gait_height_coeff = kwargs.get("gait_height_coeff", 0.0)
        self._gait_height_sharp = kwargs.get("gait_height_sharp", 0.0)
        # terms to track the 
        self._tracking_swing_z_coeff = kwargs.get("tracking_swing_z_coeff", 0.0)
        self._tracking_swing_z_sharp = kwargs.get("tracking_swing_z_sharp", 0.0)

        # knee-related terms
        self._knee_lift_coeff = kwargs.get("knee_lift_coeff", 0.0)
        self._knee_lift_sharp = kwargs.get("knee_lift_sharp", 10.0)
        self._knee_clearance_margin = kwargs.get("knee_clearance_margin", 0.20)

        # Nominal posture  weights and coefficients
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
        self.epsilon_standing = 0.03 # FIXME
        self._gen_gp_coeff = kwargs.get("gen_gp_coeff", 0.0)
        self._gen_gp_coeff_sharp = kwargs.get("gen_gp_coeff_sharp", 0.0)
        self._gen_gp_track = kwargs.get("gen_gp_track", 0.02)

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
        
        # Store all actuated joints for full body tracking (when hold_still is True)
        self._all_joints_qpos_id = self._limited_joints_qpos_id
        
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
            "tracking/gp_generation": 0.,
            "tracking/tracking_swing_z": 0.,
            "tracking/tracking_stance_z": 0.,
            "tracking/joint_qpos_reward": 0.,
            "tracking/feet_swing_reward": 0.,
            "tracking/gait_height_reward": 0.,
            "tracking/knee_height": 0.,
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

        # Pillars ids
        self.terrain_is_adaptive = isinstance(env._terrain, AdaPillarsTerrain)
        self.pil_ids = []
        if self.terrain_is_adaptive:
            for i in range(env._terrain.num_pillars):
                pil_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"pillar_{i}")
                self.pil_ids.append(pil_id)
            
            # get new ids since they have been modified when adding the pillars!
            foot_names = self._info_props["foot_geom_names"]
            self._left_foot_names = [name for name in foot_names if "left" in name]
            self._right_foot_names = [name for name in foot_names if "right" in name]
            self._left_foot_ids = [
                mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name) 
                for name in self._left_foot_names
            ]
            self._right_foot_ids = [
                mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name) 
                for name in self._right_foot_names
            ]
            self._floor_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            self._left_foot_site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
            self._right_foot_site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
            foot_sensor_adrs = []
            # for foot_sensor in ['left_foot_global_linvel', 'right_foot_global_linvel', 'left_foot_global_angvel', 'right_foot_global_angvel']:
            for foot_sensor in ['left_foot_global_linvel', 'right_foot_global_linvel']:
                sensor_id = env.model.sensor(foot_sensor).id
                sensor_adr = env.model.sensor_adr[sensor_id]
                sensor_dim = env.model.sensor_dim[sensor_id]
                foot_sensor_adrs.append(list(range(sensor_adr, sensor_adr + sensor_dim)))
            
            self._left_foot_sensor_adr = np.array(foot_sensor_adrs[0])
            self._right_foot_sensor_adr = np.array(foot_sensor_adrs[1])
            # self._left_foot_sensor_adr_ang = np.array(foot_sensor_adrs[2])
            # self._right_foot_sensor_adr_ang = np.array(foot_sensor_adrs[3])

        return CrispBoosterLocomotionRewardFootPlacementState(
            gait_process=0.0,
            last_qvel=data.qvel, 
            last_action=backend.zeros(env.info.action_space.shape[0]),
            time_since_last_touchdown=backend.zeros(2, dtype=backend.float32),
            reward_components=reward_components,
            last_swing_pos_reward=self._tracking_swing_pos_w, # give by default the max reward
            last_swing_orn_reward=self._tracking_swing_orn_w, # give by default the max reward
            stance_pos_reward=self._tracking_stance_pos_w,
            stance_orn_reward=self._tracking_stance_orn_w,
            last_swing_z_reward=self._tracking_swing_z_coeff,
            stance_z_reward=self._tracking_swing_z_coeff,
            swing_foot_idx=0 
        )

    def reset(
        self, env: Any, model: Union[MjModel, Model], data: Union[MjData, Data], 
        carry: Any, backend: ModuleType
    ):
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
        # initialize the reward state
        reward_state = self.init_state(env, None, model, data, backend)
        
        # synchronize the swing foot index
        goal_state = getattr(carry.observation_states, self._goal_name)
        reward_state = reward_state.replace(swing_foot_idx=goal_state.swing_foot_idx,gait_process=goal_state.gait_process)
        
        # carry update
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
        
        left_foot_pos = data.site_xpos[self._left_foot_site_id]
        right_foot_pos = data.site_xpos[self._right_foot_site_id]
        
        # Common terms
        next_gait_process = goal_state.gait_process
        gait_process = reward_state.gait_process
        hold_still = goal_state.still_phase & (goal_state.num_gaits >= 2)
        swing_foot_idx = goal_state.swing_foot_idx
        goal_resampled = (reward_state.swing_foot_idx ^ swing_foot_idx)
        is_gp_adaptive = getattr(goal_state, "is_gp_adaptive", False)
        # update reward when needed        
        reward_state = jax.lax.cond(
            goal_resampled,
            lambda: reward_state.replace(
                stance_pos_reward=reward_state.last_swing_pos_reward, 
                stance_orn_reward=reward_state.last_swing_orn_reward,
                stance_z_reward=reward_state.last_swing_z_reward,
                swing_foot_idx=swing_foot_idx
            ),
            lambda: reward_state
        )
        # get stance and swing site ids for the knees
        swing_knee_site_id, stance_knee_site_id = jax.lax.cond(
            swing_foot_idx == 0,  # left is swinging
            lambda: (self._left_knee_site_id, self._right_knee_site_id),
            lambda: (self._right_knee_site_id, self._left_knee_site_id),
        )
        
        # ==============================================REWARD COMPONENTS==============================================
        # Survival reward
        survival_reward = 1.0

        # Goal tracking rewards
        if self._goal_name in ["GoalDoubleFootPlacement"]:
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
            
            # take just the xy components for the swing positions (just x and y)
            swing_target_pos = swing_target_pos[:2]
            swing_curr_pos = swing_curr_pos[:2]
            # take care of the z
            swing_target_z = swing_target_pos[2]
            swing_curr_z = swing_curr_pos[2]
            
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
            # compute the z error
            # constrain the gp to be in [0,0.5]
            scaled_gp = gait_process - backend.where(gait_process>=0.5, 0.5, 0.0)
            # we add a slack to z just when the gait phase is before its half, otherwise no slack
            swing_target_z_slack = swing_target_z + backend.where(scaled_gp <= 0.25, goal_state.gait_height, 0.0)
            swing_z_error_sq = backend.sum(backend.square(swing_curr_z - swing_target_z_slack))
            swing_z_error_sq_plain = backend.sum(backend.square(swing_curr_z - swing_target_z))
            
            def _wrap_to_pi(angle):
                return (angle + backend.pi) % (2 * backend.pi) - backend.pi
            swing_orn_error = backend.square(_wrap_to_pi(swing_target_orn - swing_curr_orn))
            stance_orn_error = backend.square(_wrap_to_pi(stance_target_orn - stance_curr_orn))
            
            # if we are in the right phase (gait_process >= 0.5) we remove 0.5
            # the quantity we consider is always in 2 * [0, 0.5]
            # gait_sharpness = 2 * (gait_process - backend.where(gait_process > 0.5, 0.5, 0))
            # still_coeff = jax.lax.select(hold_still, 0.0, 1.0)
            """NOTE: if hold still condition is met, then we say the agent not to move"""

            # NOTE: adaptive sharpness is just for the swing targets
            # swing_pos_reward = self._tracking_swing_pos_w * backend.exp(-self._tracking_swing_pos_sharp * swing_pos_error_sq * gait_sharpness) * still_coeff
            # stance_pos_reward = self._tracking_stance_pos_w * backend.exp(-self._tracking_stance_pos_sharp * stance_pos_error_sq) * still_coeff 
            # swing_pos_reward = self._tracking_swing_pos_w * backend.exp(-self._tracking_swing_pos_sharp * swing_pos_error_sq * gait_sharpness * still_coeff)
            # stance_pos_reward = self._tracking_stance_pos_w * backend.exp(-self._tracking_stance_pos_sharp * stance_pos_error_sq * still_coeff)
            swing_pos_reward = self._tracking_swing_pos_w * backend.exp(-self._tracking_swing_pos_sharp * swing_pos_error_sq)
            # stance_pos_reward = self._tracking_stance_pos_w * backend.exp(-self._tracking_stance_pos_sharp * stance_pos_error_sq)
            stance_pos_reward = reward_state.stance_pos_reward

            # swing_orn_reward = self._tracking_swing_orn_w * backend.exp(-self._tracking_swing_orn_sharp * swing_orn_error * gait_sharpness) * still_coeff 
            # stance_orn_reward = self._tracking_stance_orn_w * backend.exp(-self._tracking_stance_orn_sharp * stance_orn_error) * still_coeff 
            swing_orn_reward = self._tracking_swing_orn_w * backend.exp(-self._tracking_swing_orn_sharp * swing_orn_error)
            # stance_orn_reward = self._tracking_stance_orn_w * backend.exp(-self._tracking_stance_orn_sharp * stance_orn_error)
            stance_orn_reward = reward_state.stance_orn_reward
            
            # swing_z_reward = self._tracking_swing_z_coeff * backend.exp(-self._tracking_swing_z_sharp * swing_z_error_sq * gait_sharpness) * still_coeff
            # swing_z_reward = self._tracking_swing_z_coeff * backend.exp(-self._tracking_swing_z_sharp * swing_z_error_sq * gait_sharpness * still_coeff)
            swing_z_reward = self._tracking_swing_z_coeff * backend.exp(-self._tracking_swing_z_sharp * swing_z_error_sq)
            swing_z_reward_plain = self._tracking_swing_z_coeff * backend.exp(-self._tracking_swing_z_sharp * swing_z_error_sq_plain)
            stance_z_reward = reward_state.stance_z_reward

            # knee height reward
            swing_knee_z = data.site_xpos[swing_knee_site_id][2]
            knee_target_z = swing_target_z + self._knee_clearance_margin
            knee_z_error = backend.maximum(knee_target_z - swing_knee_z, 0.0) # just one sided
            knee_lift_reward = self._knee_lift_coeff * backend.exp(-self._knee_lift_sharp * backend.square(knee_z_error))
            # TODO: adaptive gait?
            
            # update in the state the last reward
            reward_state = reward_state.replace(
                last_swing_pos_reward=swing_pos_reward, 
                last_swing_orn_reward=swing_orn_reward, 
                last_swing_z_reward=swing_z_reward_plain
            )
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
            swing_z_reward = 0
            stance_z_reward = 0
            knee_lift_reward = 0

        # Base height reward
        floor_offset = env._terrain.get_height_at_xy(carry.terrain_state, global_pos_root[:2], backend)
        desired_z = getattr(carry.terrain_state, "desired_z", 0.0)
        if backend == np:
            if isinstance(env._terrain, AdaPillarsTerrain):
                floor_offset = desired_z
        else:
            floor_offset = jax.lax.cond(
                isinstance(env._terrain, AdaPillarsTerrain),
                lambda: desired_z,
                lambda: floor_offset
            )
        base_height_target = goal_state.goal_height + floor_offset
        base_height = global_pos_root[2] # - floor_offset
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
        action_rate_coeff = self._action_rate_coeff
        if self._goal_name in ["GoalDoubleFootPlacement"]:
            action_rate_coeff = jax.lax.cond(hold_still, lambda: 2 * self._action_rate_coeff, lambda: self._action_rate_coeff)
        # mask the last action (if the last one is the generation of the gp)
        if backend == jnp:
            action_to_consider, last_action_to_consider = jax.lax.cond(
                is_gp_adaptive,
                lambda: (action.at[-1].set(0.0), reward_state.last_action.at[-1].set(0.0)),
                lambda: (action, reward_state.last_action)
            )
        else:
            if is_gp_adaptive:
                action_to_consider = action.copy()
                action_to_consider[-1] = 0.0
                last_action_to_consider = reward_state.last_action.copy()
                last_action_to_consider[-1] = 0.0
            else:
                action_to_consider = action
                last_action_to_consider = reward_state.last_action
        # action_rate_reward = (backend.square(action - reward_state.last_action)).sum()
        action_rate_reward = (backend.square(action_to_consider - last_action_to_consider)).sum()

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
        
        def old_get_feet_contact_states():
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
    
        def get_feet_contact_states():
            """Check if the foot is in contact with the floor or pillars."""
            def check_single_contact(f_id):
                floor_contact = mj_check_collisions(f_id, self._floor_id, data, backend) 
                is_pillar = False

                if self.terrain_is_adaptive and len(self.pil_ids) > 0:
                    pil_contacts = [
                        mj_check_collisions(f_id, p_id, data, backend)
                        for p_id in self.pil_ids
                    ]
                    if backend == np:
                        is_pillar = any(pil_contacts)
                    else:
                        is_pillar = jnp.logical_or.reduce(jnp.array(pil_contacts)) if pil_contacts else jnp.array(False)
                return floor_contact | is_pillar

            left_contacts = [check_single_contact(f_id) for f_id in self._left_foot_ids]
            right_contacts = [check_single_contact(f_id) for f_id in self._right_foot_ids]
            
            if backend == np:
                left_foot_on_ground = any(left_contacts)
                right_foot_on_ground = any(right_contacts)
                foots_on_ground = np.array([left_foot_on_ground, right_foot_on_ground])
            else:
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
        # left_foot_vel_ang = data.sensordata[self._left_foot_sensor_adr_ang]
        # right_foot_vel_ang = data.sensordata[self._right_foot_sensor_adr_ang]
        feet_on_ground = get_feet_contact_states()
        
        feet_slip_reward = (
            backend.square(left_foot_vel[:3] * feet_on_ground[0]) + 
            backend.square(right_foot_vel[:3] * feet_on_ground[1]) 
            # backend.square(left_foot_vel_ang[:3] * feet_on_ground[0]) + 
            # backend.square(right_foot_vel_ang[:3] * feet_on_ground[1])
        ).sum()

        # Feet yaw difference reward
        left_foot_yaw = R.from_matrix(data.site_xmat[self._left_foot_site_id]).as_euler('xyz')[2]
        left_foot_yaw = (left_foot_yaw + backend.pi) % (2 * backend.pi) - backend.pi
        
        right_foot_yaw = R.from_matrix(data.site_xmat[self._right_foot_site_id]).as_euler('xyz')[2]
        right_foot_yaw = (right_foot_yaw + backend.pi) % (2 * backend.pi) - backend.pi
        
        feet_yaw_diff_coeff = self._feet_yaw_diff_coeff
        feet_yaw_mean_coeff = self._feet_yaw_mean_coeff
        if self._goal_name in ["GoalDoubleFootPlacement"]:
            feet_yaw_diff_coeff, feet_yaw_mean_coeff = jax.lax.cond(
                ~hold_still, lambda: (0.0, 0.0), lambda: (self._feet_yaw_diff_coeff, self._feet_yaw_mean_coeff)
            )
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
        # left_foot_pos = data.site_xpos[self._left_foot_site_id]
        # right_foot_pos = data.site_xpos[self._right_foot_site_id]
        
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
            next_gait_process = goal_state.gait_process
        else:
            next_gait_process = backend.fmod(reward_state.gait_process + env.dt * gait_frequency, 1.0)
            
        # swinging conditions
        left_swing = (
            (backend.abs(gait_process - 0.25) < 0.5 * self._feet_swing_period) & 
            (gait_frequency > 1.0e-8)
        )
        right_swing = (
            (backend.abs(gait_process - 0.75) < 0.5 * self._feet_swing_period) & 
            (gait_frequency > 1.0e-8)
        )
        
        if self._goal_name in ["GoalDoubleFootPlacement"]:
            # compute the exponential sharpness of the gait height tracking
            # get feet positions
            stance_foot_pos = jax.lax.select(
                swing_foot_idx == 0, # gait_process < 0.5, # left swinging
                right_foot_pos,
                left_foot_pos
            )
            # get offset wrt world
            swing_target_pos = jax.lax.select(
                swing_foot_idx == 0, # gait_process < 0.5, # left swinging
                goal_state.left_foot_target_pos,
                goal_state.right_foot_target_pos
            )

            feet_swing_reward = (
                (left_swing & ~feet_on_ground[0] & ~hold_still).astype(backend.float32) +
                (right_swing & ~feet_on_ground[1] & ~hold_still).astype(backend.float32) +
                backend.astype(hold_still & (feet_on_ground[0] & feet_on_ground[1]), backend.float32)
            )
            
            # desired_gait_height = env._terrain.get_height_at_xy(carry.terrain_state, stance_foot_pos[:2], backend) + goal_state.gait_height
            clipped_gp = gait_process - backend.where(gait_process >= 0.5, 0.5, 0.0) # gp in [0, 0.5]
            desired_gait_height_ada_sharp = jax.lax.select(clipped_gp < 0.25, 4 * clipped_gp, - 4 * clipped_gp + 2)
            desired_gait_height = swing_target_pos[2] + goal_state.gait_height
            desired_gait_height_error_sq = jax.lax.select(
                swing_curr_pos[2] < desired_gait_height,
                backend.square(desired_gait_height - swing_curr_pos[2]),
                0.0
            )
            gait_height_reward = self._gait_height_coeff * backend.exp(
                -self._gait_height_sharp * desired_gait_height_error_sq * desired_gait_height_ada_sharp * backend.astype(~hold_still, backend.float32)
            ) 
        else:
            feet_swing_reward = (
                (left_swing & ~feet_on_ground[0]).astype(backend.float32) +
                (right_swing & ~feet_on_ground[1]).astype(backend.float32)
            ) 
            gait_height_reward = 0.0 # for compatibility
        
        # =================================================ADAPTIVE GP=================================================
        if self._goal_name in ["GoalDoubleFootPlacement"]:
            # the gait process is already updated by the goal state
            is_left_swing = (swing_foot_idx == 0)
            is_left_about_to_stance = (gait_process < 0.5) & (gait_process > (0.25 + self._feet_swing_period * 0.5))
            is_right_about_to_stance = (gait_process >= 0.5) & (gait_process > (0.75 + self._feet_swing_period * 0.5))
            # gen_gp_reward = (is_left_swing & is_left_about_to_stance & feet_on_ground[0] & (next_gait_process >= 0.5)).astype(backend.float32) + \
            #                 (~is_left_swing & is_right_about_to_stance & feet_on_ground[1] & (next_gait_process < 0.5)).astype(backend.float32) + \
            #                 (is_left_swing & ~is_left_about_to_stance & (next_gait_process < 0.5)).astype(backend.float32) + \
            #                 (~is_left_swing & ~is_right_about_to_stance & (next_gait_process >= 0.5)).astype(backend.float32)
            # gen_gp_reward = (is_left_swing & feet_on_ground[0] & (next_gait_process >= 0.5)).astype(backend.float32) + \
            #                 (~is_left_swing & feet_on_ground[1] & (next_gait_process < 0.5)).astype(backend.float32)
            gp_off = getattr(carry.control_func_state, "gait_phase_offset", 0.0)
            gp_error = jax.lax.cond(
                is_gp_adaptive,
                lambda: backend.square(self._gen_gp_track - gp_off), # backend.square(self._gen_gp_track - action[-1]), # backend.abs(self._gen_gp_track - gp_off)
                lambda: 0.0
            )
            gen_gp_reward = self._gen_gp_coeff * backend.exp(-self._gen_gp_coeff_sharp * gp_error)
        else:
            gen_gp_reward = 0.0

        # Nominal joint position rewards
        # joint_qpos_coeff = jax.lax.cond(hold_still, lambda: 2 * self._nominal_joint_pos_coeff, lambda: self._nominal_joint_pos_coeff)
        joint_qpos_coeff = self._nominal_joint_pos_coeff
        if backend == jnp:
            # Compute both rewards (all joints and nominal joints)
            joint_qpos_reward_all = backend.exp(
                -1 * self._nominal_joint_pos_exp *
                backend.square(
                    data.qpos[self._all_joints_qpos_id] - 
                    self._nominal_joint_qpos[self._all_joints_qpos_id]
                ).sum()
            )
            joint_qpos_reward_nominal = backend.exp(
                -1 * self._nominal_joint_pos_exp *
                backend.square(
                    data.qpos[self._nominal_joint_qpos_id] - 
                    self._nominal_joint_qpos[self._nominal_joint_qpos_id]
                ).sum()
            )
            # Select the appropriate reward based on hold_still
            joint_qpos_reward = jax.lax.cond(
                hold_still,
                lambda: joint_qpos_reward_all,
                lambda: joint_qpos_reward_nominal
            )
            
            # Compute both L1 penalties
            joint_deviation_l1_all = backend.sum(backend.abs(
                data.qpos[self._all_joints_qpos_id] - 
                self._nominal_joint_qpos[self._all_joints_qpos_id]
            ))
            joint_deviation_l1_nominal = backend.sum(backend.abs(
                data.qpos[self._nominal_joint_qpos_id] - 
                self._nominal_joint_qpos[self._nominal_joint_qpos_id]
            ))
            # Select the appropriate penalty based on hold_still
            joint_deviation_l1_penalty = jax.lax.cond(
                hold_still,
                lambda: joint_deviation_l1_all,
                lambda: joint_deviation_l1_nominal
            )
        else:
            # NumPy backend
            joints_to_track = self._all_joints_qpos_id if hold_still else self._nominal_joint_qpos_id
            joint_qpos_reward = backend.exp(
                -1 * self._nominal_joint_pos_exp *
                backend.square(
                    data.qpos[joints_to_track] - 
                    self._nominal_joint_qpos[joints_to_track]
                ).sum()
            )
            joint_deviation_l1_penalty = backend.sum(backend.abs(
                data.qpos[joints_to_track] - 
                self._nominal_joint_qpos[joints_to_track]
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
        swing_z_reward *= env.dt
        stance_z_reward *= env.dt
        gait_height_reward *= env.dt
        gen_gp_reward *= env.dt
        knee_lift_reward *= env.dt
        joint_qpos_reward *= (joint_qpos_coeff * env.dt)
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
        action_rate_reward *= (action_rate_coeff * env.dt)
        low_gains_reward *= (self._low_gains_coeff * env.dt)
        joint_position_limit_reward *= (self._joint_position_limit_coeff * env.dt)
        feet_slip_reward *= (self._feet_slip_coeff * env.dt)
        feet_yaw_diff_reward *= (feet_yaw_diff_coeff * env.dt)
        feet_yaw_mean_reward *= (feet_yaw_mean_coeff * env.dt)
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
            joint_qpos_reward + feet_swing_reward +
            swing_z_reward + stance_z_reward +
            gait_height_reward + gen_gp_reward + knee_lift_reward
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
            gait_process=next_gait_process,
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
            "tracking/gp_generation": gen_gp_reward,
            "tracking/tracking_swing_z": swing_z_reward,
            "tracking/tracking_stance_z": stance_z_reward,
            "tracking/gait_height_reward": gait_height_reward,
            "tracking/joint_qpos_reward": joint_qpos_reward,
            "tracking/feet_swing_reward": feet_swing_reward,
            "tracking/knee_height": knee_lift_reward,
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
            "penalties/impact_reward": impact_reward
        }
        
        reward_state = reward_state.replace(reward_components=updated_reward_components)
        carry = carry.replace(reward_state=reward_state)
        
        return total_reward, carry