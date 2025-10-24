from typing import Dict, List
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from jax.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R
from flax import struct

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

from loco_mujoco.core.observations.goals import Goal

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
            data=data,
            carry=carry.replace(key=subkey3),
            backend=backend,
            initial_gait=gp0,
            gait_frequency=gait_frequency
        )

        observation_states = carry.observation_states.replace(**{self.name: goal_state})
        return data, carry.replace(key=key, observation_states=observation_states)

    def sample_goal(self, data, carry, backend, initial_gait, gait_frequency):
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
                new_goal = self.sample_goal(data=data, carry=carry, backend=backend, initial_gait=gp, gait_frequency=state.gait_frequency)
                state = new_goal
        else:
            state = jax.lax.cond(
                resample_goal,
                lambda s: self.sample_goal(data=data, carry=carry, backend=backend, initial_gait=gp, gait_frequency=state.gait_frequency),
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

class GoalDoubleFootPlacement(Goal, DoubleFootPlacementVisualizer):
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

    def init_state(self, env, key, model, data, backend) -> GoalDoubleFootPlacementState:
        """Initializes the state with a zero target."""
        return GoalDoubleFootPlacementState(
            left_foot_target_pos=backend.zeros(3), 
            left_foot_target_orn=backend.array([1.0, 0.0, 0.0, 0.0]), 
            right_foot_target_pos=backend.zeros(3), 
            right_foot_target_orn=backend.array([1.0, 0.0, 0.0, 0.0]), 
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
            data=data,
            carry=carry.replace(key=subkey3),
            backend=backend,
            initial_gait=gp0,
            gait_frequency=gait_frequency
        )

        observation_states = carry.observation_states.replace(**{self.name: goal_state})
        return data, carry.replace(key=key, observation_states=observation_states)

    def sample_goal(self, data, carry, backend, initial_gait, gait_frequency):
        """Sample a new random foot placement goal for a random foot in any direction."""
        R = jnp_R if backend == jnp else np_R
        key = carry.key

        # current state
        state = getattr(carry.observation_states, self.name)

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
        stance_foot_orn = R.from_matrix(data.site_xmat[stance_foot_site_id].reshape(3, 3)).as_quat(scalar_first=True)

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
        target_z_offset = 0 # FIXME jax.random.uniform(subkey3, minval=self.z_height_range[0], maxval=self.z_height_range[1])
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

        # replace the information we already know we can substitute
        state = state.replace(
            swing_foot_idx=swing_foot_idx,
            goal_height=self.goal_height,
            gait_frequency=gait_frequency,
            gait_process=gp
        )

        # Replace the info for the left or right foot (the stance foot has its current position and orientations as targets)
        if backend == np:
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
            state = jax.lax.cond(
                (swing_foot_idx == 0),
                lambda s: s.replace(
                    left_foot_target_pos=target_pos,
                    left_foot_target_orn=target_orn,
                    right_foot_target_pos=stance_foot_pos,
                    right_foot_target_orn=stance_foot_orn
                ),
                lambda s: s.replace(
                    right_foot_target_pos=target_pos,
                    right_foot_target_orn=target_orn,
                    left_foot_target_pos=stance_foot_pos,
                    left_foot_target_orn=stance_foot_orn
                ),
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
        if backend == np:
            if resample_goal:
                new_goal = self.sample_goal(data=data, carry=carry, backend=backend, initial_gait=gp, gait_frequency=state.gait_frequency)
                state = new_goal
        else:
            state = jax.lax.cond(
                resample_goal,
                lambda s: self.sample_goal(data=data, carry=carry, backend=backend, initial_gait=gp, gait_frequency=state.gait_frequency),
                lambda s: s,
                operand=state
            )

        # Get the rotation matrix to convert into the root frame
        global_pose_root = data.qpos[self._root_qpos_ids]
        global_pos = global_pose_root[:3] # root global position
        global_quat = global_pose_root[3:7] # root global orientation
        global_rot = R.from_quat(quat_scalarfirst2scalarlast(global_quat)) # root rotation matrix

        # retireve info about both feet
        left_pos_w  = data.site_xpos[self._foot_site_id_left]
        left_mat_w  = data.site_xmat[self._foot_site_id_left].reshape(3, 3)
        left_quat_w = R.from_matrix(left_mat_w).as_quat(scalar_first=True)  # (w,x,y,z)

        right_pos_w  = data.site_xpos[self._foot_site_id_right]
        right_mat_w  = data.site_xmat[self._foot_site_id_right].reshape(3, 3)
        right_quat_w = R.from_matrix(right_mat_w).as_quat(scalar_first=True)

        # Compute the target position offset in base frame
        left_pos_offset_w = state.left_foot_target_pos - left_pos_w
        right_pos_offset_w = state.right_foot_target_pos - right_pos_w
        # change coordinates into local frame
        left_pos_offset_local = global_rot.apply(left_pos_offset_w, inverse=True)
        right_pos_offset_local = global_rot.apply(right_pos_offset_w, inverse=True)

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

        # Compute the one hot of the foot to move
        swing_one_hot = jax.nn.one_hot(state.swing_foot_idx, 2)

        # Concatenate the observation
        observation = backend.concatenate([
            left_pos_offset_local, # left info pos
            left_local_target_offset_orn, # left info orn
            right_pos_offset_local, # right info pos
            right_local_target_offset_orn, # right info orn
            backend.array([gp, state.gait_frequency]), # gait process info
            # swing_one_hot
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
        return 16 # let's see... if also the one hot then 16

    @property
    def has_visual(self) -> bool:
        """Visualization could be added later (e.g., a sphere at the target)."""
        return True