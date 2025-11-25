from typing import Dict, List, Tuple, Any, Union
from types import ModuleType
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from jax.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R
from flax import struct
from mujoco import MjSpec, MjModel, MjData
from mujoco.mjx import Model, Data

from loco_mujoco.core.observations.visualizer import FootPlacementVisualizer, DoubleFootPlacementVisualizer

from loco_mujoco.core.utils.math import (
    calculate_relative_site_quatities,
    quat_scalarfirst2scalarlast,
    quat_scalarlast2scalarfirst
)
from loco_mujoco.core.utils.mujoco import (
    mj_jntid2qposid, mj_jntid2qvelid,
    mj_jntname2qposid, mj_jntname2qvelid
)

from loco_mujoco.core.observations.goals import Goal, GoalChangingRandomRootVelocity

@struct.dataclass
class GoalRandomFootPlacementState:
    """State for the goal of a random foot placement position."""
    swing_target_pos: jax.Array         # 3D (x,y,z) desired WORLD position of the swing foot
    swing_target_orn: jax.Array         # 4D (w,x,y,z) desired WORLD world orientation quaternion of the swing foot
    swing_foot_idx: int                 # 0 for left, 1 for right
    goal_height: float                  # the desired height to maintain (for booster is 0.68)
    gait_frequency: float               # the desired gait frequency (1.0 is normal, 2.0 is very fast)     

class GoalRandomFootPlacement(Goal, FootPlacementVisualizer):
    """
    Goal for tracking a random target (x,y,z) position, (w,x,y,z) orientation and swing foot.
    Target is relative to the stance foot.
    """
    def __init__(
            self,
            info_props: Dict,
            left_foot_site_name: str,
            right_foot_site_name: str,
            xy_distance_range: List[float] = [0.2, 0.4],
            z_height_range: List[float] = [0.05, 0.15],
            angle_range_deg: List[float] = [-180.0, 180.0],
            yaw_range_deg: List[float] = [-15.0, 15.0],
            goal_height: float = 0.68,
            gait_frequency: float = 1.0,
            **kwargs
        ):
        
        self.foot_site_names = [left_foot_site_name, right_foot_site_name]
        self.xy_distance_range = xy_distance_range
        self.z_height_range = z_height_range
        self.angle_range_rad = [np.deg2rad(angle_range_deg[0]), np.deg2rad(angle_range_deg[1])]
        self.yaw_range_rad = [np.deg2rad(yaw_range_deg[0]), np.deg2rad(yaw_range_deg[1])]
        self.goal_height = goal_height
        self.gait_frequency = gait_frequency
        
        self._foot_site_ids = [-1, -1]
        self._root_joint_name = info_props["root_free_joint_xml_name"]
        self._root_qpos_ids = []

        FootPlacementVisualizer.__init__(self)
        n_visual_geoms = self._n_visual_geoms if kwargs.get("visualize_goal") else 0

        super().__init__(info_props, n_visual_geoms=n_visual_geoms, **kwargs)

    def _init_from_mj(self, env, model, data, current_obs_size):
        """Initialize IDs from the MuJoCo model."""
        self.obs_ind = np.arange(current_obs_size, current_obs_size + self.dim)
        self._foot_site_id_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.foot_site_names[0])
        
        self._foot_site_id_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.foot_site_names[1])

        self._root_qpos_ids = jnp.array(mj_jntname2qposid(self._root_joint_name, model))

        self.min = [-np.inf] * self.dim
        self.max = [np.inf] * self.dim

        assert self._foot_site_id_left != -1, f"Site '{self.foot_site_names[0]}' not found."
        assert self._foot_site_id_right != -1, f"Site '{self.foot_site_names[1]}' not found."
        self._initialized_from_mj = True

    def init_state(self, env, key, model, data, backend) -> GoalRandomFootPlacementState:
        """Initializes the state with a zero target."""
        return GoalRandomFootPlacementState(
            swing_target_pos=backend.zeros(3), 
            swing_target_orn=backend.array([1.0, 0.0, 0.0, 0.0]), 
            swing_foot_idx=0,
            goal_height=0.68,
            gait_frequency=1.0
        )

    def reset_state(self, env, model, data, carry, backend):
        """Sample a new random foot placement goal for a random foot in any direction."""
        R = jnp_R if backend == jnp else np_R
        key = carry.key
        
        # [1.1] Select swing foot (0=left, 1=right)
        key, subkey = jax.random.split(key)
        # sample the swing foot idx
        swing_foot_idx = jax.random.randint(subkey, shape=(), minval=0, maxval=2) 
        # compute the stance foot idx
        stance_foot_idx = 1 - swing_foot_idx 
        # retrieve the stance foot id to access data
        stance_is_right = (stance_foot_idx == 1)
        stance_foot_site_id = jax.lax.select(
            stance_is_right,
            self._foot_site_id_right,
            self._foot_site_id_left
        )

        # [1.2] Current state of stance foot and root
        # stance foot posiiton in the WORLD
        stance_foot_pos = data.site_xpos[stance_foot_site_id]
        # orientation of the body in the WORLD (needed to compute the right angles), to be converted into (x,y,z,w)
        root_quat_mj = jnp.array(data.qpos)[self._root_qpos_ids[3:7]]
        root_quat_scipy = quat_scalarfirst2scalarlast(root_quat_mj)

        # [2] Generate Position Target
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        # how far to step
        distance = jax.random.uniform(subkey1, minval=self.xy_distance_range[0], maxval=self.xy_distance_range[1])
        # step direction
        angle = jax.random.uniform(subkey2, minval=self.angle_range_rad[0], maxval=self.angle_range_rad[1])
        lateral_sign = jnp.where(stance_is_right, 1, -1)
        # target height 
        target_z_offset = jax.random.uniform(subkey3, minval=self.z_height_range[0], maxval=self.z_height_range[1])
        # step vector to be added to the stance foot coordinates
        step_vec_local = backend.array(
            [distance * backend.cos(angle), distance * backend.sin(angle) * lateral_sign, 0.0]
        )
        # compute the WORLD coordinates of the displacement
        root_rot = R.from_quat(root_quat_scipy)
        step_vec_world = root_rot.apply(step_vec_local)
        # compute the target position for the foot in WORLD coordinates, applying the displacement to WORLD stance foot
        # target_pos = stance_foot_pos + step_vec_world
        target_pos = stance_foot_pos + step_vec_local
        target_pos = target_pos.at[2].set(stance_foot_pos[2] + target_z_offset)

        # [3] Generate Orientation Target
        key, subkey4 = jax.random.split(key)
        # sample the yaw relative to the current stance foot yaw
        rand_yaw = jax.random.uniform(subkey4, minval=self.yaw_range_rad[0], maxval=self.yaw_range_rad[1]) * lateral_sign
        # target orientation in WORLD coordinates via a displacement w.r.t. the current yaw of the stance foot
        # target_orn_rot = R.from_euler('z', rand_yaw) * R.from_quat(root_quat_scipy)
        target_orn_rot = R.from_euler('z', rand_yaw)
        target_orn = target_orn_rot.as_quat(scalar_first=True)

        # [4] Update the carry object
        goal_state = GoalRandomFootPlacementState(
            swing_target_pos=target_pos, swing_target_orn=target_orn, swing_foot_idx=swing_foot_idx,
            goal_height=self.goal_height, gait_frequency=self.gait_frequency
        )
        observation_states = carry.observation_states.replace(**{self.name: goal_state})
        return data, carry.replace(key=key, observation_states=observation_states)

    def complex_get_obs_and_update_state(self, env, model, data, carry, backend):
        """
        Swing offset:  [Δx, Δy, Δz, Δquat(4)]
        One hot dwing foot id
        """
        R = jnp_R if backend == jnp else np_R
        state = getattr(carry.observation_states, self.name)

        # Root rotation (world -> root local) 
        root_quat_mj = jnp.array(data.qpos)[self._root_qpos_ids[3:7]]
        root_rot = R.from_quat(quat_scalarfirst2scalarlast(root_quat_mj))
        R_wr = root_rot.as_matrix().T # world->root


        # Retrieve world information for the swing foot
        def left_info(_):
            left_pos_w  = data.site_xpos[self._foot_site_id_left]
            left_mat_w  = data.site_xmat[self._foot_site_id_left].reshape(3, 3)
            left_quat_w = R.from_matrix(left_mat_w).as_quat(scalar_first=True)  # (w,x,y,z)
            return (left_pos_w, left_quat_w)

        def right_info(_):
            right_pos_w  = data.site_xpos[self._foot_site_id_right]
            right_mat_w  = data.site_xmat[self._foot_site_id_right].reshape(3, 3)
            right_quat_w = R.from_matrix(right_mat_w).as_quat(scalar_first=True)
            return (right_pos_w, right_quat_w)

        if backend == jnp:
            swing_pos_w, swing_orn_w = jax.lax.cond(
                (state.swing_foot_idx == 0),
                left_info,
                right_info,
                operand=None
            )
        else:
            swing_pos_w, swing_orn_w = left_info(None) if state.swing_foot_idx == 0 else right_info(None)

        # Helper function: relative quaternion q_rel = q_t ⊗ q_c^{-1}  (scalar-first I/O)
        def quat_rel_sf(q_t_sf, q_c_sf):
            Rt = R.from_quat(quat_scalarfirst2scalarlast(q_t_sf))
            Rc = R.from_quat(quat_scalarfirst2scalarlast(q_c_sf))
            q_rel_sf = (Rt * Rc.inv()).as_quat(scalar_first=True)
            # Hemisphere correction (keep w >= 0 for continuity)
            if backend == jnp:
                sign = jnp.where(q_rel_sf[0] < 0, -1.0, 1.0)
                q_rel_sf = q_rel_sf * sign
            else:
                if q_rel_sf[0] < 0:
                    q_rel_sf = -q_rel_sf
            return q_rel_sf

        # Offsets in root-local frame
        d_swing_local = R_wr @ (state.swing_target_pos - swing_pos_w)
        q_swing_rel = quat_rel_sf(state.swing_target_orn, swing_orn_w)

        # Compute the one hot of the foot to move
        swing_one_hot = jax.nn.one_hot(state.swing_foot_idx, 2)

        # Concatenate the observation
        observation = backend.concatenate([
            d_swing_local,
            q_swing_rel,
            swing_one_hot
        ])

        if self.visualize_goal:
            carry = self.set_visuals(observation, env, model, data, carry, self.visual_geoms_idx, backend)
        return observation, carry
    
    def get_obs_and_update_state(self, env, model, data, carry, backend):
        R = jnp_R if backend == jnp else np_R
        state = getattr(carry.observation_states, self.name)

        # Get the rotation matrix to convert into the root frame
        global_pose_root = data.qpos[self._root_qpos_ids]
        global_pos = global_pose_root[:3] # root global position
        global_quat = global_pose_root[3:7] # root global orientation
        global_rot = R.from_quat(quat_scalarfirst2scalarlast(global_quat)) # root rotation matrix

        # retireve info about the swing foot
        def left_info(_):
            left_pos_w  = data.site_xpos[self._foot_site_id_left]
            left_mat_w  = data.site_xmat[self._foot_site_id_left].reshape(3, 3)
            left_quat_w = R.from_matrix(left_mat_w).as_quat(scalar_first=True)  # (w,x,y,z)
            return (left_pos_w, left_quat_w)

        def right_info(_):
            right_pos_w  = data.site_xpos[self._foot_site_id_right]
            right_mat_w  = data.site_xmat[self._foot_site_id_right].reshape(3, 3)
            right_quat_w = R.from_matrix(right_mat_w).as_quat(scalar_first=True)
            return (right_pos_w, right_quat_w)

        if backend == jnp:
            swing_pos_w, swing_orn_w = jax.lax.cond(
                (state.swing_foot_idx == 0),
                left_info,
                right_info,
                operand=None
            )
        else:
            swing_pos_w, swing_orn_w = left_info(None) if state.swing_foot_idx == 0 else right_info(None)

        # Compute the target position offset in base frame
        # global_offset_pos = state.swing_target_pos - global_pos # offset in global frame
        global_offset_pos = state.swing_target_pos - swing_pos_w # offset in the world
        local_target_pos_offset = global_rot.inv().apply(global_offset_pos) # offset in local frame

        # Compute the orientation offset in base frame
        swing_mat = R.from_quat(quat_scalarfirst2scalarlast(swing_orn_w)) # rotation matrix of the swing foot
        R_target_orn_world = R.from_quat(quat_scalarfirst2scalarlast(state.swing_target_orn))
        # local_target_offset_orn = (R_target_orn_world * global_rot.inv()).as_quat(scalar_first=True)
        # local_target_offset_orn = (global_rot.inv() * R_target_orn_world).as_quat(scalar_first=True)
        local_target_offset_orn = (swing_mat.inv() * R_target_orn_world).as_quat(scalar_first=True)
        # Hemisphere correction (keep w >= 0 for continuity)
        if backend == jnp:
            sign = jnp.where(local_target_offset_orn[0] < 0, -1.0, 1.0)
            local_target_offset_orn = local_target_offset_orn * sign
        else:
            if local_target_offset_orn[0] < 0:
                local_target_offset_orn = -local_target_offset_orn

        # Compute the one hot of the foot to move
        swing_one_hot = jax.nn.one_hot(state.swing_foot_idx, 2)

        # Concatenate the observation
        observation = backend.concatenate([
            local_target_pos_offset,
            local_target_offset_orn,
            swing_one_hot
        ])

        if self.visualize_goal:
            carry = self.set_visuals(observation, env, model, data, carry, self.visual_geoms_idx, backend)
        return observation, carry


    @property
    def dim(self) -> int:
        return 9

    @property
    def has_visual(self) -> bool:
        """Visualization could be added later (e.g., a sphere at the target)."""
        return True
    
@struct.dataclass
class GoalRandomChangingFootPlacementState:
    """State for the goal of a random foot placement position."""
    swing_target_pos: jax.Array         # 3D (x,y,z) desired WORLD position of the swing foot
    swing_target_orn: jax.Array         # 4D (w,x,y,z) desired WORLD world orientation quaternion of the swing foot
    swing_foot_idx: int                 # 0 for left, 1 for right
    goal_height: float                  # the desired height to maintain (for booster is 0.68)
    gait_frequency: float               # the desired gait frequency (1.0 is normal, 2.0 is very fast)     
    # Gait information
    gait_process: float                 # \in [0,1] s.t. left \in [0,0.5) and right \in [0.5,1]

class GoalRandomChangingFootPlacement(Goal, FootPlacementVisualizer):
    """
    Goal for tracking a random target (x,y,z) position, (w,x,y,z) orientation and swing foot.
    Target is relative to the stance foot.
    """
    def __init__(
            self,
            info_props: Dict,
            left_foot_site_name: str,
            right_foot_site_name: str,
            xy_distance_range: List[float] = [0.2, 0.4],
            z_height_range: List[float] = [0.05, 0.15],
            angle_range_deg: List[float] = [-180.0, 180.0],
            yaw_range_deg: List[float] = [-15.0, 15.0],
            goal_height: float = 0.68,
            feet_swing_period: float = 0.2,
            gait_frequency_range: List[float] = [1.0, 2.0],
            **kwargs
        ):
        
        self.foot_site_names = [left_foot_site_name, right_foot_site_name]
        self.xy_distance_range = xy_distance_range
        self.z_height_range = z_height_range
        self.angle_range_rad = [np.deg2rad(angle_range_deg[0]), np.deg2rad(angle_range_deg[1])]
        self.yaw_range_rad = [np.deg2rad(yaw_range_deg[0]), np.deg2rad(yaw_range_deg[1])]
        self.goal_height = goal_height
        self.gait_frequency_range = gait_frequency_range
        self.feet_swing_period = feet_swing_period
        
        self._foot_site_ids = [-1, -1]
        self._root_joint_name = info_props["root_free_joint_xml_name"]
        self._root_qpos_ids = []

        FootPlacementVisualizer.__init__(self)
        n_visual_geoms = self._n_visual_geoms if kwargs.get("visualize_goal") else 0

        super().__init__(info_props, n_visual_geoms=n_visual_geoms, **kwargs)

    def _init_from_mj(self, env, model, data, current_obs_size):
        """Initialize IDs from the MuJoCo model."""
        self.obs_ind = np.arange(current_obs_size, current_obs_size + self.dim)
        self._foot_site_id_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.foot_site_names[0])
        
        self._foot_site_id_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.foot_site_names[1])

        self._root_qpos_ids = jnp.array(mj_jntname2qposid(self._root_joint_name, model))

        self.min = [-np.inf] * self.dim
        self.max = [np.inf] * self.dim

        assert self._foot_site_id_left != -1, f"Site '{self.foot_site_names[0]}' not found."
        assert self._foot_site_id_right != -1, f"Site '{self.foot_site_names[1]}' not found."
        self._initialized_from_mj = True

    def init_state(self, env, key, model, data, backend) -> GoalRandomChangingFootPlacementState:
        """Initializes the state with a zero target."""
        return GoalRandomChangingFootPlacementState(
            swing_target_pos=backend.zeros(3), 
            swing_target_orn=backend.array([1.0, 0.0, 0.0, 0.0]), 
            swing_foot_idx=0,
            goal_height=0.68,
            gait_frequency=1.0,
            gait_process=0.0
        )
    
    def reset_state(self, env, model, data, carry, backend):
        """Reset the goal state by sampling a new random foot placement goal."""
        R = jnp_R if backend == jnp else np_R
        key = carry.key

        # Sample the random starting phase of the gait
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        gp0 = jax.random.randint(subkey1, shape=(), minval=0, maxval=2) / 2

        # sample the gait frequency
        gait_frequency = jax.random.uniform(
            subkey2, 
            minval=self.gait_frequency_range[0], 
            maxval=self.gait_frequency_range[1]
        )

        # Sample the initial goal
        goal_state = self.sample_goal(
            env=env,
            data=data,
            carry=carry.replace(key=subkey3),
            backend=backend,
            initial_gait=gp0,
            gait_frequency=gait_frequency
        )

        observation_states = carry.observation_states.replace(**{self.name: goal_state})
        return data, carry.replace(key=key, observation_states=observation_states)

    def sample_goal(self, env, data, carry, backend, initial_gait, gait_frequency):
        """Sample a new random foot placement goal for a random foot in any direction."""
        R = jnp_R if backend == jnp else np_R
        key = carry.key

        # Select the swing foot based on the gait process
        gp = initial_gait
        left_swing = (gp < 0.5)
        swing_foot_idx = backend.astype(~left_swing, backend.int32)
        
        # Retrieve the stance foot id to access data
        stance_foot_idx = 1 - swing_foot_idx 
        stance_is_right = (stance_foot_idx == 1)
        stance_foot_site_id = jax.lax.select(
            stance_is_right,
            self._foot_site_id_right,
            self._foot_site_id_left
        )
        
        # stance foot posiiton in the WORLD
        stance_foot_pos = data.site_xpos[stance_foot_site_id]
        # orientation of the body in the WORLD (needed to compute the right angles), to be converted into (x,y,z,w)
        root_quat_mj = jnp.array(data.qpos)[self._root_qpos_ids[3:7]]
        root_quat_scipy = quat_scalarfirst2scalarlast(root_quat_mj)

        # Generate Position Target
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        # how far to step
        distance = jax.random.uniform(subkey1, minval=self.xy_distance_range[0], maxval=self.xy_distance_range[1])
        # step direction
        angle = jax.random.uniform(subkey2, minval=self.angle_range_rad[0], maxval=self.angle_range_rad[1])
        lateral_sign = jnp.where(stance_is_right, 1, -1)
        # target height 
        target_z_offset = jax.random.uniform(subkey3, minval=self.z_height_range[0], maxval=self.z_height_range[1])
        # step vector to be added to the stance foot coordinates
        step_vec_local = backend.array(
            [distance * backend.cos(angle), distance * backend.sin(angle) * lateral_sign, 0.0]
        )
        # compute the WORLD coordinates of the displacement
        # NOTE: one would naturally make the rotation to the world frame, but actually the orientation is the one of
        # NOTE: the torso (that moves quite a lot), so it makes more sense to keep the step in local frame. 
        # NOTE: The stuff to do would be to consider the waist instead of the torso.
        # root_rot = R.from_quat(root_quat_scipy)
        # step_vec_world = root_rot.apply(step_vec_local)
        # target_pos = stance_foot_pos + step_vec_world
        target_pos = stance_foot_pos + step_vec_local
        target_pos = target_pos.at[2].set(stance_foot_pos[2] + target_z_offset)

        # Generate Orientation Target
        key, subkey4 = jax.random.split(key)
        # sample the yaw relative to the current stance foot yaw
        rand_yaw = jax.random.uniform(subkey4, minval=self.yaw_range_rad[0], maxval=self.yaw_range_rad[1]) * lateral_sign
        # NOTE: if one wants to consider the relative yaw w.r.t. the stance foot, one should compose the rotations
        # target_orn_rot = R.from_euler('z', rand_yaw) * R.from_quat(root_quat_scipy) # ... add stance rotation !
        target_orn_rot = R.from_euler('z', rand_yaw)
        target_orn = target_orn_rot.as_quat(scalar_first=True)

        # Update the carry object
        goal_state = GoalRandomChangingFootPlacementState(
            swing_target_pos=target_pos, 
            swing_target_orn=target_orn, 
            swing_foot_idx=swing_foot_idx,
            goal_height=self.goal_height, 
            gait_frequency=gait_frequency,
            gait_process=gp
        )
        return goal_state

    def get_obs_and_update_state(self, env, model, data, carry, backend):
        R = jnp_R if backend == jnp else np_R
        state = getattr(carry.observation_states, self.name)

        # Check whether to update the goal
        # (each time the phase is over)
        gp = state.gait_process
        left_swing = (gp < 0.5)
        swing_foot_idx = backend.astype(~left_swing, backend.int32)
    
        # check if it is needed to resample the goal
        resample_goal = (swing_foot_idx != state.swing_foot_idx)
        if backend == np:
            if resample_goal:
                new_goal = self.sample_goal(env=env, data=data, carry=carry, backend=backend, initial_gait=gp, gait_frequency=state.gait_frequency)
                state = new_goal
        else:
            state = jax.lax.cond(
                resample_goal,
                lambda s: self.sample_goal(env=env, data=data, carry=carry, backend=backend, initial_gait=gp, gait_frequency=state.gait_frequency),
                lambda s: s,
                operand=state
            )

        # Get the rotation matrix to convert into the root frame
        global_pose_root = data.qpos[self._root_qpos_ids]
        global_pos = global_pose_root[:3] # root global position
        global_quat = global_pose_root[3:7] # root global orientation
        global_rot = R.from_quat(quat_scalarfirst2scalarlast(global_quat)) # root rotation matrix

        # retireve info about the swing foot
        def left_info(_):
            left_pos_w  = data.site_xpos[self._foot_site_id_left]
            left_mat_w  = data.site_xmat[self._foot_site_id_left].reshape(3, 3)
            left_quat_w = R.from_matrix(left_mat_w).as_quat(scalar_first=True)  # (w,x,y,z)
            return (left_pos_w, left_quat_w)

        def right_info(_):
            right_pos_w  = data.site_xpos[self._foot_site_id_right]
            right_mat_w  = data.site_xmat[self._foot_site_id_right].reshape(3, 3)
            right_quat_w = R.from_matrix(right_mat_w).as_quat(scalar_first=True)
            return (right_pos_w, right_quat_w)

        if backend == jnp:
            swing_pos_w, swing_orn_w = jax.lax.cond(
                (state.swing_foot_idx == 0),
                left_info,
                right_info,
                operand=None
            )
        else:
            swing_pos_w, swing_orn_w = left_info(None) if state.swing_foot_idx == 0 else right_info(None)

        # Compute the target position offset in base frame
        global_offset_pos = state.swing_target_pos - swing_pos_w # offset in the world
        local_target_pos_offset = global_rot.apply(global_offset_pos, inverse=True) # offset in local frame
        # NOTE: using the flag inverse=True is more efficient since internally it is just taking the transpose.
        # NOTE: global_rot is needed to rotate from the torso frame into the world frame. 
        # NOTE: global_rot.inv() is the oppsite!

        # Compute the orientation offset in base frame
        swing_mat = R.from_quat(quat_scalarfirst2scalarlast(swing_orn_w)) # rotation matrix of the swing foot
        R_target_orn_world = R.from_quat(quat_scalarfirst2scalarlast(state.swing_target_orn))
        local_target_offset_orn = (swing_mat.inv() * R_target_orn_world).as_quat(scalar_first=True)
        # Hemisphere correction (keep w >= 0 for continuity)
        if backend == jnp:
            sign = jnp.where(local_target_offset_orn[0] < 0, -1.0, 1.0)
            local_target_offset_orn = local_target_offset_orn * sign
        else:
            if local_target_offset_orn[0] < 0:
                local_target_offset_orn = -local_target_offset_orn

        # Compute the one hot of the foot to move
        swing_one_hot = jax.nn.one_hot(state.swing_foot_idx, 2)

        # Concatenate the observation
        observation = backend.concatenate([
            local_target_pos_offset,
            local_target_offset_orn,
            swing_one_hot
        ])

        # make the gait process progress
        gp = backend.fmod(gp + env.dt * state.gait_frequency, 1.0)
        state = state.replace(gait_process=gp)
        observation_states = carry.observation_states.replace(**{self.name: state})
        carry = carry.replace(observation_states=observation_states)

        if self.visualize_goal:
            carry = self.set_visuals(observation, env, model, data, carry, self.visual_geoms_idx, backend)
        return observation, carry


    @property
    def dim(self) -> int:
        return 9

    @property
    def has_visual(self) -> bool:
        """Visualization could be added later (e.g., a sphere at the target)."""
        return True
    
@struct.dataclass
class GoalDoubleFootPlacementState:
    """State for the goal of a random foot placement position."""
    left_foot_target_pos: jax.Array     # 3D (x,y,z) desired WORLD position of the left foot
    left_foot_target_orn: jax.Array     # 4D (w,x,y,z) desired WORLD world orientation quaternion of the left foot
    right_foot_target_pos: jax.Array    # 3D (x,y,z) desired WORLD position of the right foot
    right_foot_target_orn: jax.Array    # 4D (w,x,y,z) desired WORLD world orientation quaternion of the right foot
    swing_foot_idx: int                 # 0 for left, 1 for right
    goal_height: float                  # the desired height to maintain (for booster is 0.68)
    gait_frequency: float               # the desired gait frequency (1.0 is normal, 2.0 is very fast)     
    gait_process: float                 # \in [0,1] s.t. left \in [0,0.5) and right \in [0.5,1]
    gait_height: float                  # desired height of the steps
    angle_range_rad: List[float]
    distance_range: List[float]
    movement_direction: float           # angle in rad defining the movement direction
    feet_direction: float               # angle in rad defining the feet direction

class GoalDoubleFootPlacement(Goal, DoubleFootPlacementVisualizer):
    """
    Goal for tracking a random target (x,y,z) position, (w,x,y,z) orientation and swing foot.
    Target is relative to the stance foot.
    """
    def __init__(
            self,
            info_props: Dict,
            left_foot_site_name: str = "left_foot",
            right_foot_site_name: str = "right_foot",
            # canonical FP target generation
            xy_distance_range: List[float] = [0.2, 0.4],
            angle_range_deg: List[float] = [-180.0, 180.0],
            yaw_range_deg: List[float] = [-15.0, 15.0],
            goal_height: float = 0.68,
            feet_distance: float = 0.2,
            # gait information
            gait_frequency_range: List[float] = [1.0, 2.0],
            gait_height: float = 0.1,
            # movement direction
            direction_range_deg: List[float] = [0.0, 0.0],
            change_direction_range_deg: List[float] = [0.0, 0.0],
            # feet direction
            feet_direction_range_deg: List[float] = [0.0, 0.0],
            track_movement_only: bool = False,
            # still proportion
            still_proportion: float = 0.05,
            **kwargs
        ):
        
        self.foot_site_names = [left_foot_site_name, right_foot_site_name]
        self.xy_distance_range = xy_distance_range
        self.angle_range_rad = [jnp.deg2rad(angle_range_deg[0]), jnp.deg2rad(angle_range_deg[1])]
        self.yaw_range_rad = [jnp.deg2rad(yaw_range_deg[0]), jnp.deg2rad(yaw_range_deg[1])]
        self.goal_height = goal_height
        self.gait_height = gait_height
        self.gait_frequency_range = gait_frequency_range
        self.feet_distance = feet_distance
        self.direction_range_rad = [jnp.deg2rad(direction_range_deg[0]), jnp.deg2rad(direction_range_deg[1])]
        self.change_direction_range_rad = [jnp.deg2rad(change_direction_range_deg[0]), jnp.deg2rad(change_direction_range_deg[1])]
        self.feet_direction_range_rad = [jnp.deg2rad(feet_direction_range_deg[0]), jnp.deg2rad(feet_direction_range_deg[1])] 
        self.track_movement_only = track_movement_only
        self.still_proportion = still_proportion
        
        self._foot_site_ids = [-1, -1]
        self._root_joint_name = info_props["root_free_joint_xml_name"]
        self._root_qpos_ids = []

        # Walking Schemes
        self._scheme_direction = jnp.array([0.0, np.pi, np.pi/2.0, -np.pi/2.0, 0.0])  # [forward, back, left, right, stand]
        self._scheme_forward = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0])
        self._scheme_lateral = jnp.array([0.0, 0.0, 1.0, 1.0, 0.0])
        self._scheme_height = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self._scheme_yaw = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
        # local safe range for foot placement computation
        self.local_angle_range_rad = [jnp.deg2rad(20.0), jnp.deg2rad(160.0)]

        DoubleFootPlacementVisualizer.__init__(self)
        n_visual_geoms = self._n_visual_geoms if kwargs.get("visualize_goal") else 0

        super().__init__(info_props, n_visual_geoms=n_visual_geoms, **kwargs)

    def _init_from_mj(self, env, model, data, current_obs_size):
        """Initialize IDs from the MuJoCo model."""
        self.obs_ind = np.arange(current_obs_size, current_obs_size + self.dim)
        self._foot_site_id_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.foot_site_names[0])
        
        self._foot_site_id_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.foot_site_names[1])

        self._root_qpos_ids = jnp.array(mj_jntname2qposid(self._root_joint_name, model))

        self.min = [-np.inf] * self.dim
        self.max = [np.inf] * self.dim

        assert self._foot_site_id_left != -1, f"Site '{self.foot_site_names[0]}' not found."
        assert self._foot_site_id_right != -1, f"Site '{self.foot_site_names[1]}' not found."
        self._initialized_from_mj = True

    def init_state(self, env, key, model, data, backend) -> GoalDoubleFootPlacementState:
        """Initializes the state with a zero target."""
        return GoalDoubleFootPlacementState(
            # goals to track
            left_foot_target_pos=backend.zeros(3), 
            left_foot_target_orn=backend.array([1.0, 0.0, 0.0, 0.0]), 
            right_foot_target_pos=backend.zeros(3), 
            right_foot_target_orn=backend.array([1.0, 0.0, 0.0, 0.0]), 
            swing_foot_idx=0,
            goal_height=0.68,
            # gait info
            gait_frequency=1.0,
            gait_process=0.0,
            gait_height=0.1,
            # ranges
            angle_range_rad=self.angle_range_rad,
            distance_range=self.xy_distance_range,
            # movement direction
            movement_direction=0.0,
            # feet direction
            feet_direction=0.0
        )
    
    def reset_state(self, env, model, data, carry, backend):
        """Reset the goal state by sampling a new random foot placement goal."""
        R = jnp_R if backend == jnp else np_R
        key = carry.key
        key, subkey1, subkey2, subkey3, subkey4, subkey5, subkey6 = jax.random.split(key, 7)
        
        # decide whether to keep the robot to stay still
        if backend == jnp:
            hold_still = (jax.random.uniform(subkey6) < self.still_proportion)
        else:
            hold_still = (np.rand() < self.still_proportion)

        # sample the movement direction
        movement_direction = jax.random.uniform(
            subkey3,
            shape=(),
            minval=self.direction_range_rad[0],
            maxval=self.direction_range_rad[1],
        )
        movement_direction = self.wrap_to_pi(movement_direction, backend)
        
        # sample feet direction
        feet_direction = jax.random.uniform(
            subkey5,
            shape=(),
            minval=self.feet_direction_range_rad[0],
            maxval=self.feet_direction_range_rad[1]
        )
        feet_direction = self.wrap_to_pi(feet_direction, backend=backend)
        feet_direction = jax.lax.select(self.track_movement_only, movement_direction, feet_direction)

        # Sample the random starting phase of the gait
        # NOTE: if the robot is required to move on its left side, then the left foot moves first, else the right one
        rel_direction = self.wrap_to_pi(movement_direction - feet_direction, backend=backend)
        left_direction = (rel_direction > 0) & (rel_direction < backend.pi)
        right_direction = (rel_direction < 0) & (rel_direction > - backend.pi)
        boundaries = (rel_direction == 0) | (rel_direction == backend.pi) | (rel_direction == -backend.pi)
        
        # sample randomly the first gate 
        rand_gp0 = jax.random.randint(subkey1, shape=(), minval=0, maxval=2) / 2
        
        gp0 = jnp.where(
            left_direction, 0.0,
            jnp.where(
                right_direction, 0.5,
                jnp.where(boundaries, rand_gp0, 0.0)
            )
        )
        
        # cond_left = (movement_direction > 0) & (movement_direction < backend.pi)
        # cond_right = (movement_direction > -backend.pi) & (movement_direction < 0)
        # cond_boundaries = (movement_direction == 0) | (movement_direction == backend.pi) | (movement_direction == -backend.pi)
        # gp0 = jnp.where(
        #     cond_left, 0.0,
        #     jnp.where(
        #         cond_right, 0.5,
        #         jnp.where(cond_boundaries, rand_gp0, 0.0)
        #     )
        # )

        # sample the gait frequency
        gait_frequency = jax.random.uniform(
            subkey2, 
            minval=self.gait_frequency_range[0], 
            maxval=self.gait_frequency_range[1]
        )
        gait_frequency = jax.lax.select(hold_still, 0.0, gait_frequency) # if the robot has to stay still, then 0

        # adjust the distance range base on the selected gait
        distance_range = backend.array(
            [
                self.xy_distance_range[0],
                backend.minimum(self.xy_distance_range[1], (self.xy_distance_range[1] * self.gait_frequency_range[0]) / gait_frequency)
            ]
        )
        angle_range_rad = backend.array(self.angle_range_rad)
        
        # Sample the initial goal
        goal_state = self.sample_goal(
            env=env, 
            data=data,
            carry=carry.replace(key=subkey4),
            backend=backend,
            initial_gait=gp0,
            gait_frequency=gait_frequency,
            distance_range=distance_range,
            angle_range_rad=angle_range_rad,
            movement_direction=movement_direction,
            feet_direction=feet_direction,
            reset=True
        )

        observation_states = carry.observation_states.replace(**{self.name: goal_state})
        return data, carry.replace(key=key, observation_states=observation_states)

    @staticmethod
    def wrap_to_pi(angle, backend):
        """Wrap any angle (in rad) to be in [-pi, pi]"""
        return (angle + backend.pi) % (2 * backend.pi) - backend.pi

    def sample_goal(self, env, data, carry, backend, initial_gait, gait_frequency, distance_range, angle_range_rad, reset = False, movement_direction = 0.0, feet_direction = 0.0):
        """Sample a new random foot placement goal for a random foot in any direction."""
        R = jnp_R if backend == jnp else np_R
        key = carry.key

        # current state
        state = getattr(carry.observation_states, self.name)
        
        # verify whether the task is to stay still
        hold_still = (gait_frequency == 0.0)

        # Select the swing foot based on the gait process
        gp = initial_gait
        left_swing = (gp < 0.5)
        swing_foot_idx = backend.astype(~left_swing, backend.int32)
        
        # Retrieve the stance foot id to access data
        stance_foot_idx = 1 - swing_foot_idx 
        stance_is_right = (stance_foot_idx == 1)
        stance_foot_site_id = jax.lax.select(
            stance_is_right,
            self._foot_site_id_right,
            self._foot_site_id_left
        )
        swing_foot_site_id = jax.lax.select(
            ~stance_is_right,
            self._foot_site_id_right,
            self._foot_site_id_left
        )
        
        # define the movement direction
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        rand_direction_change = jax.random.uniform(
            subkey3, minval=self.change_direction_range_rad[0], maxval=self.change_direction_range_rad[1]
        )
        rand_direction_change = jax.lax.select(reset, 0.0, rand_direction_change) # if reset do not change the direction
        mov_dir_rot = R.from_euler('z', rand_direction_change) * R.from_euler('z', movement_direction)
        movement_direction = mov_dir_rot.as_euler('xyz')[2]
        # movement_direction = self.wrap_to_pi(mov_dir_rot.as_euler('xyz')[2], backend) # ensure the angle is in [-pi. pi]
        
        # re-assign the feet direction if it is the case
        feet_direction = jax.lax.select(
            self.track_movement_only,
            movement_direction,
            feet_direction
        )

        # Generate Position Target
        # stance foot posiiton in the WORLD
        stance_foot_pos = data.site_xpos[stance_foot_site_id]
        # foot orientation in the world
        stance_foot_orn_mat = data.site_xmat[stance_foot_site_id].reshape(3, 3)
        stance_foot_orn = R.from_matrix(stance_foot_orn_mat).as_quat(scalar_first=True)
        current_stance_yaw = R.from_matrix(stance_foot_orn_mat).as_euler('xyz')[2]
        # stance_foot_yaw_rot = R.from_euler('z', R.as_euler('xyz', R.from_matrix(stance_foot_orn_mat))[2]) # take the rotation considering the yaw only part
        swing_foot_pos = data.site_xpos[swing_foot_site_id]
        swing_foot_orn_mat = data.site_xmat[swing_foot_site_id].reshape(3, 3)
        swing_foot_orn = R.from_matrix(swing_foot_orn_mat).as_quat(scalar_first=True)

        # how far to step
        distance = jax.random.uniform(subkey1, minval=distance_range[0], maxval=distance_range[1])
        
        # ============================================FOOT PLACEMENT TARGET============================================
        # get the ideal world coordinates
        def _generate_position_target_no_tracking():
            angle_rand = jax.random.uniform(subkey2, minval=angle_range_rad[0], maxval=angle_range_rad[1])
            angle = (R.from_euler('z', angle_rand) * mov_dir_rot).as_euler('xyz')[2]
            ideal_world_target = backend.array(
                [distance * backend.cos(angle), distance * backend.sin(angle), 0.0]
            )
            ideal_world_target = ideal_world_target + stance_foot_pos
            
            # clip the coordinates to be in the safe areas
            # NOTE: we define a direction centered in self.feet_distance / 2.0 and with the same orientation of the 
            # NOTE: stance foot orientation, while the ideal targets are generated in the movement direction
            
            # take the rotation of the current stance yaw
            current_stance_yaw_rot = R.from_euler('z', current_stance_yaw)
            
            # convert the ideal target into the local frame of the stance foot
            local_target = current_stance_yaw_rot.apply(ideal_world_target - stance_foot_pos, inverse=True)

            # define the maximum lateral step
            max_lateral = 0.5 * self.feet_distance  # max allowed distance from foot
            
            # clip on the boundaries if it is needed
            if backend == jnp:
                local_target = jax.lax.select(
                    stance_is_right,
                    local_target.at[1].set(
                        backend.maximum(local_target[1], max_lateral)
                    ),
                    local_target.at[1].set(
                        backend.minimum(local_target[1], -max_lateral)
                    )
                )
            else:
                if stance_is_right:
                    # swing foot is left → can only be y_local >= +min_dist
                    local_target = local_target.at[1].set(
                        backend.maximum(local_target[1], max_lateral)
                    )
                else:
                    # swing foot is right → can only be y_local <= -min_dist
                    local_target = local_target.at[1].set(
                        backend.minimum(local_target[1], -max_lateral)
                    )
            
            # move the clipped foot placement from local coordinates back to world coordinates
            target_pos_pre_z = current_stance_yaw_rot.apply(local_target, inverse=False) + stance_foot_pos
            return target_pos_pre_z
        
        def _generate_position_target_tracking():
            # define safe areas
            min_local_angle = self.local_angle_range_rad[0]
            max_local_angle = self.local_angle_range_rad[1]
            
            # define the sign of the resulting angle based on the foot to swing 
            lateral_sign = jnp.where(stance_is_right, 1, -1) 
            
            # sample the random offset to add to the movement direction
            angle_rand = lateral_sign * jax.random.uniform(subkey2, minval=angle_range_rad[0], maxval=angle_range_rad[1]) 
            
            # compute the unclipped world angle
            angle = (R.from_euler('z', angle_rand) * mov_dir_rot).as_euler('xyz')[2]
            
            # convert the unclipped world angle into the stance foot frame
            local_step_angle = self.wrap_to_pi(angle - current_stance_yaw, backend)
            
            # deifne precise clip bounds
            local_clip_min = backend.where(stance_is_right, min_local_angle, -max_local_angle)
            local_clip_max = backend.where(stance_is_right, max_local_angle, -min_local_angle)
            
            # clip the angle
            clipped_local_angle = backend.clip(
                local_step_angle,
                local_clip_min,
                local_clip_max
            )
            
            # convert into the world
            final_world_angle = current_stance_yaw + clipped_local_angle
            
            # step vector to be added to the stance foot coordinates
            step_vec_local = backend.array(
                [distance * backend.cos(final_world_angle), distance * backend.sin(final_world_angle), 0.0]
            )
            # compute the WORLD coordinates of the displacement
            # NOTE: one would naturally make the rotation to the world frame, but actually the orientation is the one of
            # NOTE: the torso (that moves quite a lot), so it makes more sense to keep the step in local frame. 
            # NOTE: The stuff to do would be to consider the waist instead of the torso.
            target_pos_pre_z = stance_foot_pos + step_vec_local
            return target_pos_pre_z
        
        target_pos_pre_z = jax.lax.cond(
            self.track_movement_only,
            _generate_position_target_tracking,
            _generate_position_target_no_tracking
        )
        
        # adjust the height of the target based on the terrain properties
        target_pos_xy = target_pos_pre_z[:2]
        target_z_from_terrain = env._terrain.get_height_at_xy(carry.terrain_state, target_pos_xy, backend)
        target_pos = target_pos_pre_z.at[2].set(target_z_from_terrain)

        # ===========================================FOOT ORIENTATION TARGET===========================================
        feet_dir_rot = R.from_euler('z', feet_direction)
        key, subkey4 = jax.random.split(key)
        
        # sample the yaw relative to the current stance foot yaw
        rand_yaw = jax.random.uniform(subkey4, minval=self.yaw_range_rad[0], maxval=self.yaw_range_rad[1])
        
        # angle_yaw = (R.from_euler('z', rand_yaw) * mov_dir_rot).as_euler('xyz')[2]
        angle_yaw = (R.from_euler('z', rand_yaw) * feet_dir_rot).as_euler('xyz')[2]
        
        # keep the angle in the safe range
        # NOTE: here we have to ensure that each foot is not rotated by a crazy angle
        yaw_displacement = self.wrap_to_pi(angle_yaw - current_stance_yaw, backend)
        clipped_abs_displacement = backend.clip(backend.abs(yaw_displacement), 0, backend.pi / 2)
        clipped_yaw_displacement = backend.sign(yaw_displacement) * clipped_abs_displacement
        
        # compute final yaw
        # final_yaw = current_stance_yaw + clipped_yaw_displacement
        final_yaw = self.wrap_to_pi(current_stance_yaw + clipped_yaw_displacement, backend)
        target_orn_rot = R.from_euler('z', final_yaw)
        target_orn = target_orn_rot.as_quat(scalar_first=True)

        # replace the information we already know we can substitute
        state = state.replace(
            swing_foot_idx=swing_foot_idx,
            goal_height=self.goal_height,
            gait_frequency=gait_frequency,
            gait_height=self.gait_height,
            gait_process=gp,
            distance_range=distance_range,
            angle_range_rad=angle_range_rad,
            movement_direction=movement_direction,
            feet_direction=feet_direction
        )

        # if need to hold still, modify the target such that it is the initial position
        if backend == np:
            target_pos = swing_foot_pos if hold_still else target_pos
            target_orn = swing_foot_orn if hold_still else target_orn
        else:
            target_pos = jax.lax.select(
                hold_still,
                swing_foot_pos,
                target_pos
            )
            target_orn = jax.lax.select(
                hold_still,
                swing_foot_orn,
                target_orn
            )

        # Replace the info for the left or right foot (the stance foot has its current position and orientations as targets)
        if backend == np:
            if not reset: 
                if swing_foot_idx == 0:
                    state = state.replace(
                        left_foot_target_pos=target_pos,
                        left_foot_target_orn=target_orn,
                    )
                else:
                    state = state.replace(
                        right_foot_target_pos=target_pos,
                        right_foot_target_orn=target_orn,
                    )
            else:
                if swing_foot_idx == 0:
                    state = state.replace(
                        left_foot_target_pos=target_pos,
                        left_foot_target_orn=target_orn,
                        right_foot_target_pos=stance_foot_pos,
                        right_foot_target_orn=stance_foot_orn
                    )
                else:
                    state = state.replace(
                        right_foot_target_pos=target_pos,
                        right_foot_target_orn=target_orn,
                        left_foot_target_pos=stance_foot_pos,
                        left_foot_target_orn=stance_foot_orn
                    )
        else:
            def normal_step_update(s):
                return jax.lax.cond(
                    (swing_foot_idx == 0),
                    lambda s_inner: s_inner.replace(
                        left_foot_target_pos=target_pos,
                        left_foot_target_orn=target_orn
                    ),
                    lambda s_inner: s_inner.replace(
                        right_foot_target_pos=target_pos,
                        right_foot_target_orn=target_orn
                    ),
                    operand=s
                )
            def reset_step_update(s):
                return jax.lax.cond(
                    (swing_foot_idx == 0), 
                    lambda s_inner: s_inner.replace(
                        left_foot_target_pos=target_pos,
                        left_foot_target_orn=target_orn,
                        right_foot_target_pos=stance_foot_pos, 
                        right_foot_target_orn=stance_foot_orn
                    ),
                    lambda s_inner: s_inner.replace( 
                        right_foot_target_pos=target_pos,
                        right_foot_target_orn=target_orn,
                        left_foot_target_pos=stance_foot_pos, 
                        left_foot_target_orn=stance_foot_orn
                    ),
                    operand=s
                )
            state = jax.lax.cond(
                reset,
                reset_step_update,  
                normal_step_update, 
                operand=state
            )

        return state

    def get_obs_and_update_state(self, env, model, data, carry, backend):
        R = jnp_R if backend == jnp else np_R
        state = getattr(carry.observation_states, self.name)

        # Check whether to update the goal
        # (each time the phase is over)
        gp = state.gait_process
        left_swing = (gp < 0.5)
        swing_foot_idx = backend.astype(~left_swing, backend.int32)
    
        # check if it is needed to resample the goal
        resample_goal = (swing_foot_idx != state.swing_foot_idx)

        # resample goal if needed
        if backend == np:
            if resample_goal:
                new_goal = self.sample_goal(
                    env=env, data=data, carry=carry, backend=backend, initial_gait=gp, gait_frequency=state.gait_frequency,
                    distance_range=state.distance_range, angle_range_rad=state.angle_range_rad, 
                    movement_direction=state.movement_direction, feet_direction=state.feet_direction
                )
                state = new_goal
        else:
            state = jax.lax.cond(
                resample_goal,
                lambda s: self.sample_goal(
                    env=env, data=data, carry=carry, backend=backend, initial_gait=gp, gait_frequency=state.gait_frequency,
                    distance_range=state.distance_range, angle_range_rad=state.angle_range_rad, 
                    movement_direction=state.movement_direction, feet_direction=state.feet_direction
                ),
                lambda s: s,
                operand=state
            )

        # Get the rotation matrix to convert into the root frame
        # global_pose_root = data.qpos[self._root_qpos_ids]
        # global_pos = global_pose_root[:3] # root global position
        # global_quat = global_pose_root[3:7] # root global orientation
        # global_rot = R.from_quat(quat_scalarfirst2scalarlast(global_quat)) # root rotation matrix

        # retireve info about both feet
        left_pos_w  = data.site_xpos[self._foot_site_id_left]
        left_mat_w  = data.site_xmat[self._foot_site_id_left].reshape(3, 3)
        left_quat_w = R.from_matrix(left_mat_w).as_quat(scalar_first=True)  # (w,x,y,z)

        right_pos_w  = data.site_xpos[self._foot_site_id_right]
        right_mat_w  = data.site_xmat[self._foot_site_id_right].reshape(3, 3)
        right_quat_w = R.from_matrix(right_mat_w).as_quat(scalar_first=True)

        # Compute the target position offset in base frame
        # left_pos_offset_w = state.left_foot_target_pos - left_pos_w
        # right_pos_offset_w = state.right_foot_target_pos - right_pos_w
        # change coordinates into local frame
        # left_pos_offset_local = global_rot.apply(left_pos_offset_w, inverse=True)
        # right_pos_offset_local = global_rot.apply(right_pos_offset_w, inverse=True)

        # Compute the orientation offset in base frame
        left_foot_matrix = R.from_quat(quat_scalarfirst2scalarlast(left_quat_w))
        right_foot_matrix = R.from_quat(quat_scalarfirst2scalarlast(right_quat_w))
        # get the rotation of the targets
        left_R_target_orn_world =  R.from_quat(quat_scalarfirst2scalarlast(state.left_foot_target_orn))
        right_R_target_orn_world =  R.from_quat(quat_scalarfirst2scalarlast(state.right_foot_target_orn))
        # rotate the offsets into local frame
        left_local_target_offset_orn = (left_foot_matrix.inv() * left_R_target_orn_world).as_quat(scalar_first=True)
        right_local_target_offset_orn = (right_foot_matrix.inv() * right_R_target_orn_world).as_quat(scalar_first=True)
        # Hemisphere correction (keep w >= 0 for continuity)
        if backend == jnp:
            # left
            sign = jnp.where(left_local_target_offset_orn[0] < 0, -1.0, 1.0)
            left_local_target_offset_orn = left_local_target_offset_orn * sign
            # right
            sign = jnp.where(right_local_target_offset_orn[0] < 0, -1.0, 1.0)
            right_local_target_offset_orn = right_local_target_offset_orn * sign
        else:
            # left
            if left_local_target_offset_orn[0] < 0:
                left_local_target_offset_orn = -left_local_target_offset_orn
            # right
            if right_local_target_offset_orn[0] < 0:
                right_local_target_offset_orn = -right_local_target_offset_orn
        
        # get the stance foot positions and orientations
        stance_pos, stance_orn, swing_pos_target, swing_orn_target, stance_pos_target, stance_orn_target = jax.lax.cond(
            (swing_foot_idx == 0),
            lambda _: (right_pos_w, R.from_matrix(right_mat_w), state.left_foot_target_pos, state.left_foot_target_orn, state.right_foot_target_pos, state.right_foot_target_orn),
            lambda _: (left_pos_w, R.from_matrix(left_mat_w), state.right_foot_target_pos, state.right_foot_target_orn, state.left_foot_target_pos, state.left_foot_target_orn),
            operand=0
        )
        # offset wrt the stance foot of the swing foot
        pos_offset = stance_orn.apply(swing_pos_target - stance_pos, inverse=True)
        orn_offset = (stance_orn.inv() * R.from_quat(quat_scalarfirst2scalarlast(swing_orn_target))).as_quat(scalar_first=True)
        sign = jnp.where(orn_offset[0] < 0, -1.0, 1.0)
        orn_offset = orn_offset * sign
        # offset wrt the stance foot of the stance foot
        hold_pos = stance_orn.apply(stance_pos_target - stance_pos, inverse=True) 
        hold_orn = (stance_orn.inv() * R.from_quat(quat_scalarfirst2scalarlast(stance_orn_target))).as_quat(scalar_first=True) 
        sign = jnp.where(hold_orn[0] < 0, -1.0, 1.0)
        hold_orn = hold_orn * sign
        left_pos_targ, left_orn_targ, right_pos_targ, right_orn_targ = jax.lax.cond(
            (swing_foot_idx == 0),
            lambda _: (pos_offset, orn_offset, hold_pos, hold_orn),
            lambda _: (hold_pos, hold_orn, pos_offset, orn_offset),
            operand=0
        )
        observation = backend.concatenate(
            [
                left_pos_targ,
                left_orn_targ,
                right_pos_targ,
                right_orn_targ,
                backend.array([backend.cos(2 * backend.pi * gp) * (state.gait_frequency > 1.0e-8),
                backend.sin(2 * backend.pi * gp) * (state.gait_frequency > 1.0e-8)])
            ]
        )

        # make the gait process progress
        gp = backend.fmod(gp + env.dt * state.gait_frequency, 1.0)
        state = state.replace(gait_process=gp)
        observation_states = carry.observation_states.replace(**{self.name: state})
        carry = carry.replace(observation_states=observation_states)

        if self.visualize_goal:
            carry = self.set_visuals(observation, env, model, data, carry, self.visual_geoms_idx, backend)
        return observation, carry

    @property
    def dim(self) -> int:
        return 16 

    @property
    def has_visual(self) -> bool:
        """Visualization could be added later (e.g., a sphere at the target)."""
        return True
    
@struct.dataclass
class GoalVelocityAndObstaclesState:
    # Inheriting from the ones of the GoalRandomRootVelocityAndFrequencyState 
    goal_vel_x: float
    goal_vel_y: float
    goal_vel_yaw: float
    goal_height: float
    gait_frequency: float
    # New terms for the obstacles
    obstacle_positions: jax.Array  # (num_obstacles, 2) for (x, y), in the WORLD frame
    obstacle_radii: jax.Array      # (num_obstacles,)

class GoalVelocityAndObstacles(GoalChangingRandomRootVelocity):
    """
    Extends GoalChangingRandomRootVelocity to include obstacles generation.
    - Spawns random cylindrical obstacles at the start of an episode (... they can be resampled or they can move)
    - Provides a height map observation representing the terrain in front of the robot
    - The final observation is [velocity_goal(6), height_map(N*M)]
    """
    def __init__(
        self,
        info_props: Dict,
        # Heightmap parameters
        heightmap_grid_size: Tuple[int, int] = (16, 5),  # (rows, cols)
        heightmap_resolution: float = 0.1,  # meters per grid cell
        heightmap_offset: Tuple[float, float] = (0.3, 0.0), # (x, y) offset from robot base
        # Obstacle parameters
        num_obstacles: int = 10,
        obstacle_radius_range: List[float] = [0.1, 0.3],
        spawn_area_dims: Tuple[float, float] = (10.0, 5.0), # (width, height) in front
        **kwargs # make sure to pass what is needed by the goal we are inheriting from
    ):   
        # Initialize internal vars
        self.heightmap_grid_size = heightmap_grid_size
        self.heightmap_resolution = heightmap_resolution
        self.heightmap_offset = jnp.array(heightmap_offset)
        self.num_obstacles = num_obstacles
        self.obstacle_radius_range = obstacle_radius_range
        self.spawn_area_dims = spawn_area_dims

        # Create the local grid points for the heightmap sensor (once)
        rows, cols = self.heightmap_grid_size
        x_coords = jnp.linspace(0, (rows - 1) * self.heightmap_resolution, rows) + self.heightmap_offset[0]
        y_coords = jnp.linspace(-(cols - 1)/2 * self.heightmap_resolution, (cols - 1)/2 * self.heightmap_resolution, cols) + self.heightmap_offset[1]
        self._local_grid_x, self._local_grid_y = jnp.meshgrid(x_coords, y_coords, indexing='ij')
        self._local_grid_points = jnp.stack([self._local_grid_x.flatten(), self._local_grid_y.flatten()], axis=1)

        # Grid visualization stuff
        self._grid_sphere_size = np.array([0.015, 0.0, 0.0]) # Small spheres
        self._free_color = np.array([0.2, 0.8, 0.2, 0.6])    # Green, semi-transparent
        self._occupied_color = np.array([0.8, 0.2, 0.2, 0.6])# Red, semi-transparent
        n_grid_geoms = self.heightmap_grid_size[0] * self.heightmap_grid_size[1]

        # Initialize the parent class
        # kwargs["n_visual_geoms"] = kwargs.get("n_visual_geoms", 0) + self.num_obstacles
        super().__init__(info_props, **kwargs)    
        self.n_visual_geoms = self.n_visual_geoms + self.num_obstacles + n_grid_geoms

        # Logic for separating the observations
        # access to the super velocity dimension
        self.velocity_dim = super().dim
        # access to the height map dimension
        self.hm_dim = self.heightmap_grid_size[0] * self.heightmap_grid_size[1]
        # define sub-groups indices
        self.obs_subgroups = {
            "velocity": jnp.arange(0, self.velocity_dim),
            "height_map": jnp.arange(self.velocity_dim, self.velocity_dim + self.hm_dim)
        }
    
    def init_state(
        self,
        env: Any,
        key: jax.random.PRNGKey,
        model: Union[MjModel, Model],
        data: Union[MjData, Data],
        backend: ModuleType
    ) -> GoalVelocityAndObstaclesState:
        """Initializes the state with zero velocity and no obstacles."""
        return GoalVelocityAndObstaclesState(
            goal_vel_x=0.0, goal_vel_y=0.0, goal_vel_yaw=0.0,
            goal_height=0.68, gait_frequency=0.0,
            obstacle_positions=backend.zeros((self.num_obstacles, 2)),
            obstacle_radii=backend.zeros((self.num_obstacles,))
        )
    
    def reset_state(
        self,
        env: Any,
        model: Union[MjModel, Model],
        data: Union[MjData, Data],
        carry: Any,
        backend: ModuleType
    ) -> Tuple[Union[MjData, Any], Any]:
        """Resets velocity commands (from parent) and spawns new obstacles."""
        # Call the reset of the velocity goal
        data, carry = super().reset_state(env, model, data, carry, backend)
        key = carry.key
        
        # Sample new obstacle positions and radii in the world frame
        # Obstacles spawn in a rectangular area in front of the robot's starting position
        start_pos = data.qpos[:2] # Robot's initial (x, y)
        
        if backend == np:
            key, subkey1, subkey2 = (None, None, None) 
            rand_fn = np.random.uniform
        else:
            key, subkey1, subkey2 = jax.random.split(key, 3)
            rand_fn = jax.random.uniform

        # Sample positions
        min_spawn = start_pos + np.array([0, -self.spawn_area_dims[1] / 2])
        max_spawn = start_pos + np.array([self.spawn_area_dims[0], self.spawn_area_dims[1] / 2])
        obstacle_positions = rand_fn(subkey1, shape=(self.num_obstacles, 2), minval=min_spawn, maxval=max_spawn)
        
        # Sample radii
        obstacle_radii = rand_fn(subkey2, shape=(self.num_obstacles,), minval=self.obstacle_radius_range[0], maxval=self.obstacle_radius_range[1])

        # Update the state in the carry object
        goal_state_parent = getattr(carry.observation_states, self.name)
        new_goal_state = GoalVelocityAndObstaclesState(
            goal_vel_x=goal_state_parent.goal_vel_x,
            goal_vel_y=goal_state_parent.goal_vel_y,
            goal_vel_yaw=goal_state_parent.goal_vel_yaw,
            goal_height=goal_state_parent.goal_height,
            gait_frequency=goal_state_parent.gait_frequency,
            obstacle_positions=obstacle_positions,
            obstacle_radii=obstacle_radii
        )
        observation_states = carry.observation_states.replace(**{self.name: new_goal_state})
        
        return data, carry.replace(key=key, observation_states=observation_states)
    
    def get_obs_and_update_state(
        self,
        env: Any,
        model: Union[MjModel, Model],
        data: Union[MjData, Data],
        carry: Any,
        backend: ModuleType
    ) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """Generates the velocity goal + the heightmap observation."""
        # Get the standard velocity goal observation from the parent class
        velocity_obs, carry = super().get_obs_and_update_state(env, model, data, carry, backend)
        state = getattr(carry.observation_states, self.name)

        # Get robot's current pose from `data`
        root_pos = data.qpos[:2]
        root_quat_mj = data.qpos[3:7]
        R = jnp_R if backend == jnp else np_R
        # We only care about yaw for the 2D heightmap
        root_yaw = R.from_quat(quat_scalarfirst2scalarlast(root_quat_mj)).as_euler('xyz')[2]

        # Transform local grid points to world frame
        c, s = backend.cos(root_yaw), backend.sin(root_yaw)
        rot_mat = backend.array([[c, -s], [s, c]])
        world_grid_points = root_pos + self._local_grid_points @ rot_mat.T

        # Sense the obstacles: check if grid points are inside any obstacle
        # Use broadcasting to compute distances efficiently
        # distances shape: (num_grid_points, num_obstacles)
        distances = backend.linalg.norm(world_grid_points[:, None, :] - state.obstacle_positions, axis=2)
        
        # is_inside shape: (num_grid_points, num_obstacles)
        is_inside = distances < state.obstacle_radii

        # For each grid point, is it inside ANY obstacle?
        # occupied shape: (num_grid_points,)
        occupied = backend.any(is_inside, axis=1)

        # Create the height map: 1.0 for obstacle, 0.0 for free space
        heightmap = backend.where(occupied, 1.0, 0.0)

        # Concatenate velocity goal and flattened heightmap
        final_obs = backend.concatenate([velocity_obs, heightmap])

        # Update visuals for obstacles (parent class handles velocity arrow)
        if self.visualize_goal:
            carry = self.set_visuals(
                velocity_obs, env, model, data, carry, self._root_body_id,
                self._free_jnt_qpos_id, self.visual_geoms_idx, backend
            )

        return final_obs, carry
    
    def set_visuals(
            self, goal: Union[np.ndarray, jnp.ndarray], env: Any, model: Union[MjModel, Model], 
            data: Union[MjData, Data], carry: Any, root_body_id: int, free_jnt_qposid: Union[np.ndarray, jnp.ndarray], 
            visual_geoms_idx: List[int], backend: ModuleType
        ) -> Any:
        """Draws the velocity arrow (via parent) and the cylindrical obstacles."""
        state = getattr(carry.observation_states, self.name)
        R = jnp_R if backend == jnp else np_R

        parent_geoms_count = self._arrow_n_visual_geoms
        obstacle_start_idx = parent_geoms_count
        grid_start_idx = parent_geoms_count + self.num_obstacles

        parent_geoms_idx = visual_geoms_idx[:obstacle_start_idx]
        obstacle_geoms_idx = visual_geoms_idx[obstacle_start_idx:grid_start_idx]
        grid_geoms_idx = visual_geoms_idx[grid_start_idx:]

        # Call parent to draw the velocity arrow
        carry = super().set_visuals(goal, env, model, data, carry, root_body_id, free_jnt_qposid, parent_geoms_idx, backend)

        root_pos = data.qpos[:2]
        root_quat_mj = data.qpos[3:7]
        root_yaw = R.from_quat(quat_scalarfirst2scalarlast(root_quat_mj)).as_euler('xyz')[2]
        c, s = backend.cos(root_yaw), backend.sin(root_yaw)
        rot_mat = backend.array([[c, -s], [s, c]])
        world_grid_points = root_pos + self._local_grid_points @ rot_mat.T
        distances = backend.linalg.norm(world_grid_points[:, None, :] - state.obstacle_positions, axis=2)
        is_inside = distances < state.obstacle_radii
        occupied = backend.any(is_inside, axis=1)

        # Draw the obstacles
        if backend == jnp:
            geom_pos = carry.user_scene.geoms.pos
            geom_size = carry.user_scene.geoms.size
            geom_type = carry.user_scene.geoms.type
            geom_rgba = carry.user_scene.geoms.rgba

            # Draw the obstacles
            for i in range(self.num_obstacles):
                idx = obstacle_geoms_idx[i]
                pos3d = jnp.array([state.obstacle_positions[i, 0], state.obstacle_positions[i, 1], 0.0])
                size3d = jnp.array([state.obstacle_radii[i], state.obstacle_radii[i], 0.5])
                geom_pos = geom_pos.at[idx].set(pos3d)
                geom_size = geom_size.at[idx].set(size3d)
                geom_type = geom_type.at[idx].set(int(mujoco.mjtGeom.mjGEOM_CYLINDER))
                geom_rgba = geom_rgba.at[idx].set(jnp.array([0.8, 0.2, 0.2, 0.7]))

            # Draw the heightmap grid 
            for i in range(len(grid_geoms_idx)):
                idx = grid_geoms_idx[i]
                grid_pos_3d = jnp.array([world_grid_points[i, 0], world_grid_points[i, 1], 0.01]) # slightly above ground
                color = jnp.where(occupied[i], self._occupied_color, self._free_color)
                geom_pos = geom_pos.at[idx].set(grid_pos_3d)
                geom_size = geom_size.at[idx].set(self._grid_sphere_size)
                geom_type = geom_type.at[idx].set(int(mujoco.mjtGeom.mjGEOM_SPHERE))
                geom_rgba = geom_rgba.at[idx].set(color)
            
            new_geoms = carry.user_scene.geoms.replace(pos=geom_pos, size=geom_size, type=geom_type, rgba=geom_rgba)
        
        else: # NumPy backend
            # Draw the obstacles
            for i in range(self.num_obstacles):
                idx = obstacle_geoms_idx[i]
                pos3d = np.array([state.obstacle_positions[i, 0], state.obstacle_positions[i, 1], 0.0])
                size3d = np.array([state.obstacle_radii[i], state.obstacle_radii[i], 0.5])
                carry.user_scene.geoms.pos[idx] = pos3d
                carry.user_scene.geoms.size[idx] = size3d
                carry.user_scene.geoms.type[idx] = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
                carry.user_scene.geoms.rgba[idx] = np.array([0.8, 0.2, 0.2, 0.7])

            # Draw the heightmap grid
            for i in range(len(grid_geoms_idx)):
                idx = grid_geoms_idx[i]
                grid_pos_3d = np.array([world_grid_points[i, 0], world_grid_points[i, 1], 0.01])
                color = self._occupied_color if occupied[i] else self._free_color
                carry.user_scene.geoms.pos[idx] = grid_pos_3d
                carry.user_scene.geoms.size[idx] = self._grid_sphere_size
                carry.user_scene.geoms.type[idx] = int(mujoco.mjtGeom.mjGEOM_SPHERE)
                carry.user_scene.geoms.rgba[idx] = color

            new_geoms = carry.user_scene.geoms

        new_user_scene = carry.user_scene.replace(geoms=new_geoms)
        return carry.replace(user_scene=new_user_scene)

    @property
    def dim(self) -> int:
        """The dimension is the parent's dimension plus the size of the heightmap."""
        return super().dim + (self.heightmap_grid_size[0] * self.heightmap_grid_size[1])