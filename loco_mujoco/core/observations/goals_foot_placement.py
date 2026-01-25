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

from loco_mujoco.core.observations.visualizer import DoubleFootPlacementVisualizer
from loco_mujoco.core.terrain.adaptive_pillars import AdaPillarsTerrain

from loco_mujoco.core.utils.math import (
    quat_scalarfirst2scalarlast,
)
from loco_mujoco.core.utils.mujoco import (
    mj_jntname2qposid
)

from loco_mujoco.core.observations.goals import Goal

@struct.dataclass
class GoalDoubleFootPlacementState:
    """State for the goal of a random foot placement position."""
    # target positions and orientations
    left_foot_target_pos: jax.Array     # 3D (x,y,z) desired WORLD position of the left foot
    left_foot_target_orn: jax.Array     # 4D (w,x,y,z) desired WORLD world orientation quaternion of the left foot
    right_foot_target_pos: jax.Array    # 3D (x,y,z) desired WORLD position of the right foot
    right_foot_target_orn: jax.Array    # 4D (w,x,y,z) desired WORLD world orientation quaternion of the right foot
    # swing foot index and goal height to mantain
    swing_foot_idx: int                 # 0 for left, 1 for right
    goal_height: float                  # the desired height to maintain (for booster is 0.68)
    # gait information
    gait_frequency: float               # the desired gait frequency (1.0 is normal, 2.0 is very fast)     
    gait_process: float                 # \in [0,1] s.t. left \in [0,0.5) and right \in [0.5,1]
    gait_height: float                  # desired height of the steps
    # ranges for foot placement target generation
    angle_range_rad: jax.Array
    distance_range: jax.Array
    movement_direction: float           # angle in rad defining the movement direction
    feet_direction: float               # angle in rad defining the feet direction
    z_distance_range: jax.Array
    steps: int
    # still process parameters
    still_phase: bool                   # boolean number indicating if the goal to provide is the one to be still
    # number of gait phase switches
    num_gaits: int                      # integer stating how many gait switches happened so far
    # vars for adaptive terrain
    foot_pillar_ids: jax.Array
    free_pillar_id: jax.Array
    pending_free_pillar_id: jax.Array
    # term for tracking whether the gp is decided by the policy
    is_gp_adaptive: bool
    # goal based absorbing management
    absorbing: bool

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
            feet_distance: float = 0.5,
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
            still_feet_distance: float = 0.2,
            still_threshold: float = 0.05,
            # number of gait phases for goal switching
            max_num_gaits: int = 20,
            # define terrain type and height sampling parameters
            adaptive_terrain: bool = False,
            z_distance_range: List[float] = [0.0, 0.0],
            max_z_distance_up: float = 0.0,
            max_z_distance_low: float = 0.0,
            # curriculum parameters
            n_envs: float = 0,
            num_total_timesteps: float = 0,
            curriculum_starts_from: float = 0,
            curriculum_ends_at: float = 0, 
            curriculum: bool = False,
            update_z_every: float = 1.0,
            # start still flag
            start_still: bool = False,
            # absorbing when tracking fails
            tracking_fail_is_absorbing: bool = False,
            stance_absorbing_threshold: float = 0.5,
            # flag indicating whether the policy is able to learn the gp offset
            is_gp_adaptive: bool = False,
            **kwargs
        ):
        # store parameters
        self.foot_site_names = [left_foot_site_name, right_foot_site_name]
        self.xy_distance_range = xy_distance_range
        self.angle_range_rad = [jnp.deg2rad(angle_range_deg[0]), jnp.deg2rad(angle_range_deg[1])]
        self.yaw_range_rad = [jnp.deg2rad(yaw_range_deg[0]), jnp.deg2rad(yaw_range_deg[1])]
        self.goal_height = goal_height
        self.gait_height = gait_height
        self.gait_frequency_range = gait_frequency_range
        self.foot_safe_distance = feet_distance
        self.direction_range_rad = [jnp.deg2rad(direction_range_deg[0]), jnp.deg2rad(direction_range_deg[1])]
        self.change_direction_range_rad = [jnp.deg2rad(change_direction_range_deg[0]), jnp.deg2rad(change_direction_range_deg[1])]
        self.feet_direction_range_rad = [jnp.deg2rad(feet_direction_range_deg[0]), jnp.deg2rad(feet_direction_range_deg[1])] 
        self.track_movement_only = track_movement_only
        self.still_proportion = still_proportion
        self.still_feet_distance = still_feet_distance
        self.max_num_gaits = max_num_gaits
        self.still_threshold = still_threshold
        self.adaptive_terrain = adaptive_terrain
        self.z_distance_range = z_distance_range
        self.max_z_distance_up = max_z_distance_up
        self.max_z_distance_low = max_z_distance_low
        self.update_z_every = update_z_every
        self.start_still = start_still
        self.goal_based_absorbing = tracking_fail_is_absorbing
        self.stance_absorbing_threshold = stance_absorbing_threshold
        self.is_gp_adaptive = is_gp_adaptive
        
        # curriculum parmeters
        self.curriculum = curriculum
        curriculum_ends_at = min(curriculum_ends_at, num_total_timesteps)
        self.curriculum_end = int(curriculum_ends_at // n_envs)
        self.curriculum_start = int(curriculum_starts_from // n_envs)
        self.tot_curriculum_steps = int((curriculum_ends_at - curriculum_starts_from) //n_envs)
        self.incremental_z_up = (self.max_z_distance_up - self.z_distance_range[1]) / self.tot_curriculum_steps
        self.incremental_z_low = (self.max_z_distance_low - self.z_distance_range[0]) / self.tot_curriculum_steps
        # phase-based curriculum
        self.update_z_every = max(update_z_every, n_envs) // n_envs
        self.delta_pcurr_up = self.incremental_z_up * self.update_z_every
        self.delta_pcurr_low = self.incremental_z_low * self.update_z_every
        
        self._foot_site_ids = [-1, -1]
        self._root_joint_name = info_props["root_free_joint_xml_name"]
        self._root_qpos_ids = []

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
        
        # Pillar avoidance terms for adapitve terrains
        pillar_d = float(getattr(getattr(env, "_terrain", None), "diameter", 0.0))
        pillar_r = 0.5 * pillar_d
        foot_r = 0.0
        fd = getattr(getattr(env, "_terrain", None), "foot_dimension", None)
        if fd is not None and len(fd) >= 2:
            L, W = float(fd[0]), float(fd[1])
            foot_r = 0.5 * np.sqrt(L * L + W * W)
        self._overlap_margin = 0.05
        self._foot_margin = 0.05
        safe_overlap = pillar_d + self._overlap_margin
        safe_foot = pillar_r + foot_r + self._foot_margin
        self._pillar_min_center_dist = float(max(safe_overlap, safe_foot))

        assert self._foot_site_id_left != -1, f"Site '{self.foot_site_names[0]}' not found."
        assert self._foot_site_id_right != -1, f"Site '{self.foot_site_names[1]}' not found."
        self._initialized_from_mj = True

    def init_state(self, env, key, model, data, backend) -> GoalDoubleFootPlacementState:
        """Initializes the state with a zero target."""
        num_pillars = int(getattr(getattr(env, "_terrain", None), "num_pillars", 0))
        if num_pillars >= 3:
            foot_pillar_ids = backend.array([0, 1], dtype=backend.int32)
            free_pillar_id = backend.array(2, dtype=backend.int32)
        else:
            # fallback (old behaviour): pillar_id == swing_foot_idx
            foot_pillar_ids = backend.array([0, 1], dtype=backend.int32)
            free_pillar_id = backend.array(-1, dtype=backend.int32)

        pending_free_pillar_id = backend.array(-1, dtype=backend.int32)

        # udoate ids after adding of pillars
        self._foot_site_id_left = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, self.foot_site_names[0])
        self._foot_site_id_right = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, self.foot_site_names[1])
        self._root_qpos_ids = jnp.array(mj_jntname2qposid(self._root_joint_name, env.model))

        pillar_d = float(getattr(getattr(env, "_terrain", None), "diameter", 0.0))
        pillar_r = 0.5 * pillar_d
        foot_r = 0.0
        fd = getattr(getattr(env, "_terrain", None), "foot_dimension", None)
        if fd is not None and len(fd) >= 2:
            L, W = float(fd[0]), float(fd[1])
            foot_r = 0.5 * np.sqrt(L * L + W * W)
        self._overlap_margin = 0.05
        self._foot_margin = 0.05
        safe_overlap = pillar_d + self._overlap_margin
        safe_foot = pillar_r + foot_r + self._foot_margin
        self._pillar_min_center_dist = float(max(safe_overlap, safe_foot))

        assert self._foot_site_id_left != -1, f"Site '{self.foot_site_names[0]}' not found."
        assert self._foot_site_id_right != -1, f"Site '{self.foot_site_names[1]}' not found."
        self._initialized_from_mj = True
        
        return GoalDoubleFootPlacementState(
            # goals to track
            left_foot_target_pos=backend.zeros(3), 
            left_foot_target_orn=backend.array([1.0, 0.0, 0.0, 0.0]), 
            right_foot_target_pos=backend.zeros(3), 
            right_foot_target_orn=backend.array([1.0, 0.0, 0.0, 0.0]), 
            swing_foot_idx=0,
            goal_height=self.goal_height,
            # gait info
            gait_frequency=1.0,
            gait_process=0.0,
            gait_height=self.gait_height,
            # ranges
            angle_range_rad=backend.array(self.angle_range_rad),
            distance_range=backend.array(self.xy_distance_range),
            z_distance_range=backend.array(self.z_distance_range),
            steps=0,
            # movement direction
            movement_direction=0.0,
            # feet direction
            feet_direction=0.0,
            # still info
            still_phase=self.start_still,
            # number of gait phase switches
            num_gaits=0,
            # adaptive terrains
            foot_pillar_ids=foot_pillar_ids,
            free_pillar_id=free_pillar_id,
            pending_free_pillar_id=pending_free_pillar_id,
            absorbing=False,
            is_gp_adaptive=self.is_gp_adaptive
        )
        
    def reset_state(self, env, model, data, carry, backend):
        # get the key
        key = carry.key 
        key, sk1, sk2 = jax.random.split(key, 3)
        
        # sample initial gait parmeters
        movement_dir, feet_dir, gp0, gait_frequency, distance_range, angle_range_rad = self._sample_gait_parameters(sk1)
        
        # Sample the initial goal
        goal_state, carry = self.sample_goal(
            env=env, 
            data=data,
            carry=carry.replace(key=sk2),
            backend=backend,
            initial_gait=gp0,
            gait_frequency=gait_frequency,
            distance_range=backend.array(distance_range),
            angle_range_rad=backend.array(angle_range_rad),
            movement_direction=movement_dir,
            feet_direction=feet_dir,
            reset=True,
            z_distance_range=backend.array(self.z_distance_range)
        )
        
        # adaptive terrains stuff
        num_pillars = int(getattr(getattr(env, "_terrain", None), "num_pillars", 0))
        if num_pillars >= 3:
            foot_pillar_ids = backend.array([0, 1], dtype=backend.int32)
            free_pillar_id = backend.array(2, dtype=backend.int32)
        else:
            # fallback (old behaviour): pillar_id == swing_foot_idx
            foot_pillar_ids = backend.array([0, 1], dtype=backend.int32)
            free_pillar_id = backend.array(-1, dtype=backend.int32)

        pending_free_pillar_id = backend.array(-1, dtype=backend.int32)
        goal_state = goal_state.replace(
            foot_pillar_ids=foot_pillar_ids,
            free_pillar_id=free_pillar_id,
            pending_free_pillar_id=pending_free_pillar_id,
            absorbing=False,
            is_gp_adaptive=self.is_gp_adaptive,
            gait_frequency=gait_frequency
        )

        # update observation with the new goal state in the carry
        observation_states = carry.observation_states.replace(**{self.name: goal_state})
        
        return data, carry.replace(key=key, observation_states=observation_states)
    
    def _sample_movement_direction(self, key) -> Tuple[float, float, float]:
        """
        Sample a movement direction and a feet direction given the current state. Given the direction, select the 
        initial gait process too.
        NOTE: just JAX!
        """
        # get the keys to smaple random directions
        sk1, sk2, sk3 = jax.random.split(key, 3)
        
        # ==================================================MOVEMENT==================================================
        # sample movement direction
        movement_dir = jax.random.uniform(
            sk1, shape=(), minval=self.direction_range_rad[0], maxval=self.direction_range_rad[1]
        )
        movement_dir = self.wrap_to_pi(movement_dir, jnp)
        
        # ====================================================FEET====================================================
        # sample feet direction
        feet_dir = jax.random.uniform(
            sk2, shape=(), minval=self.feet_direction_range_rad[0], maxval=self.feet_direction_range_rad[1]
        )
        feet_dir = self.wrap_to_pi(feet_dir, jnp)
        # if we just need to tracke the movement, we trash the feet_dir
        feet_dir = jax.lax.select(self.track_movement_only, movement_dir, feet_dir) 
        
        # ==============================================GAIT PROCESS at 0==============================================
        # random sampling of 0 (LEFT) or 0.5 (RIGHT)
        rand_gp0 = jax.random.randint(sk3, shape=(), minval=0, maxval=2) / 2
        
        # NOT tracking the movement
        rel_direction = self.wrap_to_pi(movement_dir - feet_dir, jnp)
        left_direction = (rel_direction > 0) & (rel_direction < jnp.pi)
        right_direction = (rel_direction < 0) & (rel_direction > - jnp.pi)
        boundaries = (rel_direction == 0) | (rel_direction == jnp.pi) | (rel_direction == -jnp.pi)
        gp0_no_track = jnp.where(
            left_direction, 0.0,
            jnp.where(
                right_direction, 0.5,
                jnp.where(boundaries, rand_gp0, 0.0)
            )
        )
        
        # ONLY tracking the movement
        left_direction = (movement_dir > 0) & (movement_dir < jnp.pi)
        right_direction = (movement_dir > -jnp.pi) & (movement_dir < 0)
        boundaries = (movement_dir == 0) | (movement_dir == jnp.pi) | (movement_dir == -jnp.pi)
        gp0_track_only = jnp.where(
            left_direction, 0.0,
            jnp.where(
                right_direction, 0.5,
                jnp.where(boundaries, rand_gp0, 0.0)
            )
        )
        
        # decide the initial gp0
        gp0 = jax.lax.select(self.track_movement_only, gp0_track_only, gp0_no_track)
        
        return movement_dir, feet_dir, gp0
    
    def _sample_gait_frequency(self, key) -> Tuple[float, jax.Array, jax.Array]:
        """
        Sample the gait frequency and adjust the distance range for safety
        """
        # sampel gait frequency
        gait_frequency = jax.random.uniform(
            key, shape=(), minval=self.gait_frequency_range[0], maxval=self.gait_frequency_range[1]
        )
        # adjust the distance range base on the selected gait
        distance_range = jnp.array(
            [
                self.xy_distance_range[0],
                jnp.minimum(self.xy_distance_range[1], (self.xy_distance_range[1] * self.gait_frequency_range[0]) / gait_frequency)
            ]
        )
        # adjust the ange range
        angle_range_rad = jnp.array(self.angle_range_rad)
        
        return gait_frequency, distance_range, angle_range_rad
    
    def _sample_gait_parameters(self, key) -> Tuple[float, float, float, float, jax.Array, jax.Array]:
        """
        Call to both functions _sample_movement_direction and _sample_gait_frequency
        """ 
        sk1, sk2 = jax.random.split(key, 2)
        
        # sample: movement direction, feet direction, gait process
        movement_dir, feet_dir, gp0 = self._sample_movement_direction(sk1)
        
        # sample gait frequency and adjust the ranges
        gait_frequency, distance_range, angle_range_rad = self._sample_gait_frequency(sk2)
        
        return movement_dir, feet_dir, gp0, gait_frequency, distance_range, angle_range_rad
    
    def old_reset_state(self, env, model, data, carry, backend):
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
        
        gp0_no_track = jnp.where(
            left_direction, 0.0,
            jnp.where(
                right_direction, 0.5,
                jnp.where(boundaries, rand_gp0, 0.0)
            )
        )
        
        cond_left = (movement_direction > 0) & (movement_direction < backend.pi)
        cond_right = (movement_direction > -backend.pi) & (movement_direction < 0)
        cond_boundaries = (movement_direction == 0) | (movement_direction == backend.pi) | (movement_direction == -backend.pi)
        gp0_track_only = jnp.where(
            cond_left, 0.0,
            jnp.where(
                cond_right, 0.5,
                jnp.where(cond_boundaries, rand_gp0, 0.0)
            )
        )
        
        gp0 = jax.lax.select(self.track_movement_only, gp0_track_only, gp0_no_track)

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
        goal_state, carry = self.sample_goal(
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

    def _push_target_out_of_other_pillars(
        self,
        target_pos_pre_z,
        terrain_state,
        swing_pillar_id: int,
        stance_foot_pos, 
        swing_foot_pos,
        backend,
    ):
        """
        Deterministic projection that avoids all other pillars AND the current stance and swing feet.
        In particular, we are avoiding that the free pillar is spawned colliding with the feet.
        """

        # If we don't have pillar info, do nothing
        safe = float(getattr(self, "_pillar_min_center_dist", 0.0))
        if safe <= 0.0:
            return target_pos_pre_z

        centers_xy = terrain_state.positions[:, :2]   # (P,2)
        idxs = backend.arange(centers_xy.shape[0])
        big = backend.array(1e6, dtype=backend.float32)
        safe = backend.array(safe, dtype=backend.float32)

        def _one_push(xy):
            # Distances to all pillars
            dxy = xy[None, :] - centers_xy                 # (P,2)
            d = backend.linalg.norm(dxy, axis=1)           # (P,)

            # Ignore the pillar we are going to move/update anyway
            d = backend.where(idxs == swing_pillar_id, big, d)

            # Find closest pillar
            j = backend.argmin(d)
            closest_pillar_xy = centers_xy[j]
            closest_pillar_d = d[j]
            
            # Distance to feet
            # Stance Foot (the current position not the desired one that maybe has not been reached)
            vec_stance = xy - stance_foot_pos[:2]
            d_stance = backend.linalg.norm(vec_stance)
            # Swing Foot
            vec_swing = xy - swing_foot_pos[:2]
            d_swing = backend.linalg.norm(vec_swing)
            
            # Determine the actual closest obstacle 
            # Compare all three: stance foot, swing foot, and closest pillar
            is_stance_closer = (d_stance < closest_pillar_d) & (d_stance < d_swing)
            is_swing_closer = (d_swing < closest_pillar_d) & (d_swing < d_stance)
            
            # Select the closest obstacle
            closest_xy = backend.where(
                is_stance_closer, stance_foot_pos[:2],
                backend.where(is_swing_closer, swing_foot_pos[:2], closest_pillar_xy)
            )
            closest_d = backend.where(
                is_stance_closer, d_stance,
                backend.where(is_swing_closer, d_swing, closest_pillar_d)
            )

            # If too close push to boundary
            vec = xy - closest_xy
            norm = backend.linalg.norm(vec)
            default_dir = backend.array([1.0, 0.0], dtype=backend.float32)
            dir_xy = backend.where(norm > 1e-6, vec / norm, default_dir)

            pushed_xy = closest_xy + dir_xy * safe
            xy2 = backend.where(closest_d < safe, pushed_xy, xy)
            return xy2

        # Iterate a few times so we don’t get pushed into another pillar
        if backend == jnp:
            def body_fun(i, xy):
                return _one_push(xy)
            xy0 = target_pos_pre_z[:2]
            xy = jax.lax.fori_loop(0, 4, body_fun, xy0)  
            return target_pos_pre_z.at[:2].set(xy)
        else:
            out = target_pos_pre_z.copy()
            xy = out[:2]
            for _ in range(4):
                xy = np.array(_one_push(xy))
            out[:2] = xy
            return out

    @staticmethod
    def wrap_to_pi(angle, backend):
        """Wrap any angle (in rad) to be in [-pi, pi]"""
        return (angle + backend.pi) % (2 * backend.pi) - backend.pi

    def sample_goal(
        self, env, data, carry, backend, initial_gait, gait_frequency, 
        distance_range, angle_range_rad, z_distance_range, 
        reset = False, movement_direction = 0.0, feet_direction = 0.0
    ):
        """Sample a new random foot placement goal for a random foot in any direction."""
        # take rotation backend; key for jax randomness; goal state
        R = jnp_R if backend == jnp else np_R
        key = carry.key
        state = getattr(carry.observation_states, self.name)
        
        # verify whether we are at reset time
        hold_still = state.still_phase 

        # ===========================================SWING / STANCE FOOT IDX===========================================
        gp = initial_gait
        left_swing = (gp < 0.5)
        swing_foot_idx = backend.astype(~left_swing, backend.int32)
        
        stance_foot_idx = 1 - swing_foot_idx 
        stance_is_right = (stance_foot_idx == 1)
        
        stance_foot_site_id = jax.lax.select(
            stance_is_right, self._foot_site_id_right, self._foot_site_id_left
        )
        swing_foot_site_id = jax.lax.select(
            ~stance_is_right, self._foot_site_id_right, self._foot_site_id_left
        )
        
        # =====================================MODIFY FEET and MOVEMENT DIRECTIONS=====================================
        # define the movement direction
        key, subkey_dir, key_sampling = jax.random.split(key, 3)
        
        sign = backend.where(swing_foot_idx == 0, -1, 1)
        rand_direction_change = sign * jax.random.uniform(
            subkey_dir, minval=self.change_direction_range_rad[0], maxval=self.change_direction_range_rad[1]
        )
        rand_direction_change = jax.lax.select(hold_still, 0.0, rand_direction_change)
        
        mov_dir_rot = R.from_euler('z', rand_direction_change) * R.from_euler('z', movement_direction)
        movement_direction = mov_dir_rot.as_euler('xyz')[2]
        
        # re-assign the feet direction if it is the case
        feet_direction = jax.lax.select(self.track_movement_only, movement_direction, feet_direction)

        # ============================================FOOT PLACEMENT TARGET============================================
        # stance foot position in the WORLD
        stance_foot_pos = data.site_xpos[stance_foot_site_id]
        
        # foot orientation
        stance_foot_orn_mat = data.site_xmat[stance_foot_site_id].reshape(3, 3)
        stance_foot_orn = R.from_matrix(stance_foot_orn_mat).as_quat(scalar_first=True)
        current_stance_yaw = R.from_matrix(stance_foot_orn_mat).as_euler('xyz')[2]
        
        feet_direction = jax.lax.select(hold_still, current_stance_yaw, feet_direction)
        
        swing_foot_pos = data.site_xpos[swing_foot_site_id]
        swing_foot_orn_mat = data.site_xmat[swing_foot_site_id].reshape(3, 3)
        swing_foot_orn = R.from_matrix(swing_foot_orn_mat).as_quat(scalar_first=True)

        # =============================================PILLARS MANAGEMENT=============================================
        
        # management of the three pillars
        num_pillars = backend.astype(getattr(getattr(env, "_terrain", None), "num_pillars", 0), backend.int32)
        use_three_pillars = backend.astype(self.adaptive_terrain & (num_pillars >= 3) & ~hold_still, backend.int32)
        
        def _retrieve_pillar_id_for_goal(state_in):
            foot_pillar_ids = state_in.foot_pillar_ids
            free_pillar_id = state_in.free_pillar_id
            pending_free = state_in.pending_free_pillar_id

            neg_one = backend.array(-1, dtype=backend.int32)

            # Release pending pillar (from previous swing) so it becomes the free pillar now
            has_pending = pending_free >= 0
            free_pillar_id = jax.lax.select(has_pending, pending_free, free_pillar_id)
            pending_free = jax.lax.select(has_pending, neg_one, pending_free)

            # Allocate the free pillar to the NEW swing-foot target
            pillar_id_for_goal = free_pillar_id

            # Mark the OLD swing-foot pillar as pending-free (do NOT reuse immediately)
            old_swing_pillar = foot_pillar_ids[swing_foot_idx]
            foot_pillar_ids = foot_pillar_ids.at[swing_foot_idx].set(pillar_id_for_goal)

            pending_free = old_swing_pillar
            free_pillar_id = neg_one

            state_out = state_in.replace(
                foot_pillar_ids=foot_pillar_ids,
                free_pillar_id=free_pillar_id,
                pending_free_pillar_id=pending_free,
            )
            return pillar_id_for_goal, state_out
        
        def _fallback(state_in):
            return swing_foot_idx, state_in
        
        pillar_id_for_goal, state = jax.lax.cond(
            use_three_pillars,
            _retrieve_pillar_id_for_goal,
            _fallback,
            operand=state
        )

        # ============================================REJECTION SAMPLING============================================
        
        # Define the generator for a single candidate target
        def _get_candidate_target(rng_key):
            # Split keys for distance and angle
            k1, k2 = jax.random.split(rng_key)
            
            # 1. Sample Distance
            dist = jax.random.uniform(k1, minval=distance_range[0], maxval=distance_range[1])
            dist = jax.lax.select(hold_still, self.still_feet_distance, dist)
            
            # 2. Logic for Tracking vs No-Tracking
            def _no_tracking_candidate():
                ref_pos, ref_yaw = stance_foot_pos, current_stance_yaw
                angle_rnd = jax.random.uniform(k2, minval=angle_range_rad[0], maxval=angle_range_rad[1])
                angle = (R.from_euler('z', angle_rnd) * mov_dir_rot).as_euler('xyz')[2]
                
                # Ideal world target
                ideal = backend.array([dist * backend.cos(angle), dist * backend.sin(angle), 0.0])
                
                # Convert to local frame of stance foot
                ref_yaw_rot = R.from_euler('z', ref_yaw)
                local = ref_yaw_rot.apply(ideal, inverse=True)
                
                # Clip Lateral
                max_lat = self.foot_safe_distance
                # Apply clipping based on swing foot side logic
                y_clipped = jax.lax.select(
                    stance_is_right,
                    backend.maximum(local[1], max_lat),  # Left swing needs +y
                    backend.minimum(local[1], -max_lat)  # Right swing needs -y
                )
                local_clipped = local.at[1].set(y_clipped)
                
                return ref_yaw_rot.apply(local_clipped, inverse=False) + ref_pos

            def _tracking_candidate():
                min_ang, max_ang = self.local_angle_range_rad
                lat_sign = backend.where(stance_is_right, 1, -1)
                
                angle_rnd = lat_sign * jax.random.uniform(k2, minval=angle_range_rad[0], maxval=angle_range_rad[1])
                angle_world = (R.from_euler('z', angle_rnd) * mov_dir_rot).as_euler('xyz')[2]
                
                local_step_angle = self.wrap_to_pi(angle_world - current_stance_yaw, backend)
                
                clip_min = backend.where(stance_is_right, min_ang, -max_ang)
                clip_max = backend.where(stance_is_right, max_ang, -min_ang)
                clipped_angle = backend.clip(local_step_angle, clip_min, clip_max)
                
                final_world_angle = current_stance_yaw + clipped_angle
                step_vec = backend.array([dist * backend.cos(final_world_angle), dist * backend.sin(final_world_angle), 0.0])
                return stance_foot_pos + step_vec

            def _hold_still_candidate():
                sign = jnp.where((swing_foot_idx == 0), 1, -1)
                ideal_local = backend.array([0.0, sign * dist, 0.0])
                cur_rot = R.from_euler('z', current_stance_yaw)
                return cur_rot.apply(ideal_local, inverse=False) + stance_foot_pos

            # Select Strategy
            candidate = jax.lax.cond(
                hold_still,
                _hold_still_candidate,
                lambda: jax.lax.cond(
                    self.track_movement_only,
                    _tracking_candidate,
                    _no_tracking_candidate
                )
            )
            return candidate

        # Collision Check Function
        def _check_valid(pos_xy):
            # stance foot condition
            dist_stance = backend.linalg.norm(pos_xy[:2] - stance_foot_pos[:2])
            valid_stance = dist_stance > self._pillar_min_center_dist 

            # swing foot condition
            dist_swing = backend.linalg.norm(pos_xy[:2] - swing_foot_pos[:2])
            valid_swing = dist_swing > self._pillar_min_center_dist

            # pillars condition
            valid_pillars = backend.array(True)
            if self.adaptive_terrain:
                pil_coo = carry.terrain_state.positions
                pillars_d = backend.linalg.norm(pil_coo[:, :2] - pos_xy[None, :2], axis=1)
                is_too_close = pillars_d < self._pillar_min_center_dist
                if backend == jnp:
                    is_too_close = is_too_close.at[pillar_id_for_goal].set(False)
                else:
                    is_too_close[pillar_id_for_goal] = False
                valid_pillars = ~backend.any(is_too_close)

            return valid_stance & valid_swing & valid_pillars

        # Perform Sampling
        if backend == jnp:
            def cond_fun(val):
                _, count, found, _ = val
                return (~found) & (count < 1000)

            def body_fun(val):
                rng, count, _, current_best = val
                rng, step_key = jax.random.split(rng)
                
                candidate = _get_candidate_target(step_key)
                is_valid = _check_valid(candidate)

                return (rng, count + 1, is_valid, candidate)

            # Initial sample 
            key, init_key = jax.random.split(key_sampling)
            initial_target = _get_candidate_target(init_key)
            init_val = (key, 0, False, initial_target)
            
            # If hold_still is True, we don't need to loop 
            final_val = jax.lax.cond(
                hold_still,
                lambda x: (x[0], x[1], True, x[3]), 
                lambda x: jax.lax.while_loop(cond_fun, body_fun, x),
                init_val
            )
            _, _, found, target_pos_pre_z = final_val
            
            # If rejection sampling failed and we're using pillars, push target away from obstacles
            if self.adaptive_terrain:
                target_pos_pre_z = jax.lax.cond(
                    (~found) & use_three_pillars,
                    lambda pos: self._push_target_out_of_other_pillars(
                        pos, carry.terrain_state, pillar_id_for_goal, 
                        stance_foot_pos, swing_foot_pos, backend
                    ),
                    lambda pos: pos,
                    target_pos_pre_z
                )

        else:
            # NUMPY: Standard loop
            found = False
            count = 0
            target_pos_pre_z = _get_candidate_target(key) # init
            
            if not hold_still:
                while not found and count < 1000:
                    key, subkey = jax.random.split(key) 
                    cand = _get_candidate_target(subkey)
                    if _check_valid(cand):
                        target_pos_pre_z = cand
                        found = True
                    count += 1
                
                # If rejection sampling failed and we're using pillars, push target away
                if not found and self.adaptive_terrain and num_pillars >= 3:
                    target_pos_pre_z = self._push_target_out_of_other_pillars(
                        target_pos_pre_z, carry.terrain_state, pillar_id_for_goal,
                        stance_foot_pos, swing_foot_pos, backend
                    )
            else:
                target_pos_pre_z = _get_candidate_target(key)
        
        # =============================================FOOT HEIGHT TARGET=============================================
        key, zkey = jax.random.split(key)
        target_z = env._terrain.get_height_at_xy(carry.terrain_state, target_pos_pre_z[:2], backend)
        if self.adaptive_terrain:
            z_sampled = jax.random.uniform(zkey, minval=z_distance_range[0], maxval=z_distance_range[1])
            z_sampled = backend.where(hold_still, 0.0, z_sampled) 
            target_z = backend.maximum(z_sampled + stance_foot_pos[2], 0.0)

        target_pos = target_pos_pre_z.at[2].set(target_z)
        
        # when the terrain is adaptive, we need to build a pillar below the foot
        terrain_state = carry.terrain_state
        if self.adaptive_terrain:
            terrain_state = env._terrain.set_height_at_xy(carry.terrain_state, target_pos[:2], target_z, pillar_id_for_goal, backend)
        carry = carry.replace(terrain_state=terrain_state)

        # ===========================================FOOT ORIENTATION TARGET===========================================
        feet_dir_rot = R.from_euler('z', feet_direction)
        key, subkey4 = jax.random.split(key)
        
        # sample the yaw relative to the current stance foot yaw
        # feet_direction=current_stance_yaw # FIXME
        rand_yaw = jax.random.uniform(subkey4, minval=self.yaw_range_rad[0], maxval=self.yaw_range_rad[1])
        # angle_yaw = (R.from_euler('z', rand_yaw) * feet_dir_rot).as_euler('xyz')[2]
        angle_yaw = feet_direction + rand_yaw
        
        # keep the angle in the safe range
        yaw_displacement = self.wrap_to_pi(angle_yaw - current_stance_yaw, backend)
        clipped_abs_displacement = backend.clip(backend.abs(yaw_displacement), 0, backend.pi / 2)
        clipped_yaw_displacement = backend.sign(yaw_displacement) * clipped_abs_displacement
        
        # compute final yaw
        final_yaw = self.wrap_to_pi(current_stance_yaw + clipped_yaw_displacement, backend)
        # final_yaw = (R.from_euler('z', rand_yaw) * R.from_euler('z', current_stance_yaw)).as_euler('xyz')[2]
        target_orn_rot = R.from_euler('z', final_yaw)
        
        target_orn = jax.lax.select(
            hold_still,
            stance_foot_orn,
            target_orn_rot.as_quat(scalar_first=True)
            # R.from_euler('z', rand_yaw + current_stance_yaw).as_quat(scalar_first=True) 
        )

        # ===============================================ASSIGN TARGETS===============================================
        num_gaits = jax.lax.select(
            reset,
            1,
            backend.fmod(state.num_gaits + 1, self.max_num_gaits)    
        )
        
        state = state.replace(
            swing_foot_idx=swing_foot_idx,
            goal_height=self.goal_height,
            gait_frequency=gait_frequency,
            gait_height=self.gait_height,
            gait_process=gp,
            distance_range=distance_range,
            angle_range_rad=angle_range_rad,
            movement_direction=movement_direction,
            feet_direction=feet_direction,
            num_gaits=num_gaits,
        )

        if backend == np:
            target_pos = swing_foot_pos if reset else target_pos
            target_orn = swing_foot_orn if reset else target_orn
        else:
            target_pos = jax.lax.select(reset, swing_foot_pos, target_pos)
            target_orn = jax.lax.select(reset, swing_foot_orn, target_orn)

        if backend == np:
            if not reset: 
                if swing_foot_idx == 0:
                    state = state.replace(left_foot_target_pos=target_pos, left_foot_target_orn=target_orn)
                else:
                    state = state.replace(right_foot_target_pos=target_pos, right_foot_target_orn=target_orn)
            else:
                if swing_foot_idx == 0:
                    state = state.replace(
                        left_foot_target_pos=target_pos, left_foot_target_orn=target_orn,
                        right_foot_target_pos=stance_foot_pos, right_foot_target_orn=stance_foot_orn
                    )
                else:
                    state = state.replace(
                        right_foot_target_pos=target_pos, right_foot_target_orn=target_orn,
                        left_foot_target_pos=stance_foot_pos, left_foot_target_orn=stance_foot_orn
                    )
        else:
            def normal_step_update(s):
                return jax.lax.cond(
                    (swing_foot_idx == 0),
                    lambda s_inner: s_inner.replace(left_foot_target_pos=target_pos, left_foot_target_orn=target_orn),
                    lambda s_inner: s_inner.replace(right_foot_target_pos=target_pos, right_foot_target_orn=target_orn),
                    operand=s
                )
            def reset_step_update(s):
                return jax.lax.cond(
                    (swing_foot_idx == 0), 
                    lambda s_inner: s_inner.replace(
                        left_foot_target_pos=target_pos, left_foot_target_orn=target_orn,
                        right_foot_target_pos=stance_foot_pos, right_foot_target_orn=stance_foot_orn
                    ),
                    lambda s_inner: s_inner.replace( 
                        right_foot_target_pos=target_pos, right_foot_target_orn=target_orn,
                        left_foot_target_pos=stance_foot_pos, left_foot_target_orn=stance_foot_orn
                    ),
                    operand=s
                )
            state = jax.lax.cond(reset, reset_step_update, normal_step_update, operand=state)

        return state, carry

    def old_sample_goal(
        self, env, data, carry, backend, initial_gait, gait_frequency, 
        distance_range, angle_range_rad, z_distance_range, 
        reset = False, movement_direction = 0.0, feet_direction = 0.0
    ):
        """Sample a new random foot placement goal for a random foot in any direction."""
        # take rotation backend; key for jax randomness; goal state
        R = jnp_R if backend == jnp else np_R
        key = carry.key
        state = getattr(carry.observation_states, self.name)
        
        # verify whether we are at reset time: in that case we initialize the goal to stay still
        hold_still = state.still_phase 

        # ===========================================SWING / STANCE FOOT IDX===========================================
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
        
        # =====================================MODIFY FEET and MOVEMENT DIRECTIONS=====================================
        # define the movement direction
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        sign = backend.where(swing_foot_idx == 0, -1, 1)
        rand_direction_change = sign * jax.random.uniform(
            subkey3, minval=self.change_direction_range_rad[0], maxval=self.change_direction_range_rad[1]
        )
        rand_direction_change = jax.lax.select(hold_still, 0.0, rand_direction_change) # if reset do not change the direction
        mov_dir_rot = R.from_euler('z', rand_direction_change) * R.from_euler('z', movement_direction)
        movement_direction = mov_dir_rot.as_euler('xyz')[2]
        
        # re-assign the feet direction if it is the case
        feet_direction = jax.lax.select(
            self.track_movement_only,
            movement_direction,
            feet_direction
        )

        # ============================================FOOT PLACEMENT TARGET============================================
        # stance foot position in the WORLD
        stance_foot_pos = data.site_xpos[stance_foot_site_id]
        
        # foot orientation in the world
        stance_foot_orn_mat = data.site_xmat[stance_foot_site_id].reshape(3, 3)
        stance_foot_orn = R.from_matrix(stance_foot_orn_mat).as_quat(scalar_first=True)
        current_stance_yaw = R.from_matrix(stance_foot_orn_mat).as_euler('xyz')[2]
        
        # NOTE: if hold still the feet direction has to be the same as the stance foot
        feet_direction = jax.lax.select(hold_still, current_stance_yaw, feet_direction)
        
        swing_foot_pos = data.site_xpos[swing_foot_site_id]
        swing_foot_orn_mat = data.site_xmat[swing_foot_site_id].reshape(3, 3)
        swing_foot_orn = R.from_matrix(swing_foot_orn_mat).as_quat(scalar_first=True)
        current_swing_yaw = R.from_matrix(swing_foot_orn_mat).as_euler('xyz')[2]

        # how far to step
        # sampling for movement
        distance = jax.random.uniform(subkey1, minval=distance_range[0], maxval=distance_range[1])
        # sampling to stay still
        distance = jax.lax.select(hold_still, self.still_feet_distance, distance)
        """NOTE: if we have to stay still we keep the pre-defined feet distance"""
        
        # get the ideal world coordinates
        def _generate_position_target_no_tracking():
            # reference position and orientation
            ref_pos = stance_foot_pos
            ref_yaw = current_stance_yaw
            
            # sample the angle
            angle_rand = jax.random.uniform(subkey2, minval=angle_range_rad[0], maxval=angle_range_rad[1])
            angle = (R.from_euler('z', angle_rand) * mov_dir_rot).as_euler('xyz')[2]
            ideal_world_target = backend.array(
                [distance * backend.cos(angle), distance * backend.sin(angle), 0.0]
            )
            # ideal_world_target = ideal_world_target + ref_pos
            ideal_world_target = ideal_world_target
            
            # clip the coordinates to be in the safe areas
            # NOTE: we define a direction centered in self.feet_distance / 2.0 and with the same orientation of the 
            # NOTE: stance foot orientation, while the ideal targets are generated in the movement direction
            
            # take the rotation of the current stance yaw
            ref_yaw_rot = R.from_euler('z', ref_yaw)
            
            # convert the ideal target into the local frame of the stance foot
            # local_target = ref_yaw_rot.apply(ideal_world_target - ref_pos, inverse=True)
            local_target = ref_yaw_rot.apply(ideal_world_target, inverse=True)

            # FIXME: new code start
            """danger_zone_flag = (backend.abs(local_target[0]) <= self.foot_safe_distance)
            max_lateral = jax.lax.select(
                danger_zone_flag,
                self.foot_safe_distance,
                0.1 # for safety 
            )"""
            # FIXME: new code end

            # define the maximum lateral step
            max_lateral = self.foot_safe_distance  # max allowed distance from foot FIXME
            
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
            target_pos_pre_z = ref_yaw_rot.apply(local_target, inverse=False) + ref_pos
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
        
        def _generate_position_target_hold_still():
            # get the displacement sign
            sign = jnp.where((swing_foot_idx == 0), 1, -1)
            
            # compute the desired foot placement target in local coordinates
            ideal_local_target = backend.array([0.0, sign * distance, 0.0])
            
            # revert into global coordinates
            current_stance_yaw_rot = R.from_euler('z', current_stance_yaw)
            target_pos_pre_z = current_stance_yaw_rot.apply(ideal_local_target, inverse=False) + stance_foot_pos
            
            return target_pos_pre_z
        
        # discriminate whether we have to track the movement only
        target_pos_pre_z = jax.lax.cond(
            self.track_movement_only,
            _generate_position_target_tracking,
            _generate_position_target_no_tracking
        )
        
        # discrimate whether we need to hold still
        target_pos_pre_z = jax.lax.cond(
            hold_still,
            _generate_position_target_hold_still,
            lambda: target_pos_pre_z
        )
        
        # =============================================PILLARS MANAGEMENT=============================================
        
        # management of the three pillars
        num_pillars = backend.astype(getattr(getattr(env, "_terrain", None), "num_pillars", 0), backend.int32)
        use_three_pillars = backend.astype(self.adaptive_terrain & (num_pillars >= 3) & ~hold_still, backend.int32)
        
        def _retrieve_pillar_id_for_goal(state_in):
            foot_pillar_ids = state_in.foot_pillar_ids
            free_pillar_id = state_in.free_pillar_id
            pending_free = state_in.pending_free_pillar_id

            neg_one = backend.array(-1, dtype=backend.int32)

            # Release pending pillar (from previous swing) so it becomes the free pillar now
            has_pending = pending_free >= 0
            free_pillar_id = jax.lax.select(has_pending, pending_free, free_pillar_id)
            pending_free = jax.lax.select(has_pending, neg_one, pending_free)

            # Allocate the free pillar to the NEW swing-foot target
            pillar_id_for_goal = free_pillar_id

            # Mark the OLD swing-foot pillar as pending-free (do NOT reuse immediately)
            old_swing_pillar = foot_pillar_ids[swing_foot_idx]
            foot_pillar_ids = foot_pillar_ids.at[swing_foot_idx].set(pillar_id_for_goal)

            pending_free = old_swing_pillar
            free_pillar_id = neg_one  # no free pillar until next sample (when pending is released)

            # write back updated bookkeeping into state
            state_out = state_in.replace(
                foot_pillar_ids=foot_pillar_ids,
                free_pillar_id=free_pillar_id,
                pending_free_pillar_id=pending_free,
            )
            return pillar_id_for_goal, state_out
        
        def _fallback(state_in):
            return swing_foot_idx, state_in
        
        pillar_id_for_goal, state = jax.lax.cond(
            use_three_pillars,
            _retrieve_pillar_id_for_goal,
            _fallback,
            operand=state
        )
        
        # in the case of adaptive terrains, modify the proposed foot palcement targets when needed
        if self.adaptive_terrain:
            target_pos_pre_z = jax.lax.cond(
                hold_still,
                lambda: target_pos_pre_z,
                lambda: self._push_target_out_of_other_pillars(
                    target_pos_pre_z, carry.terrain_state, pillar_id_for_goal, stance_foot_pos, swing_foot_pos, backend
                )
            )
            # target_pos_pre_z = self._push_target_out_of_other_pillars(
            #     target_pos_pre_z, carry.terrain_state, pillar_id_for_goal, stance_foot_pos, swing_foot_pos, backend
            # )
        
        # =============================================FOOT HEIGHT TARGET=============================================
        key, zkey = jax.random.split(key)
        target_z = env._terrain.get_height_at_xy(carry.terrain_state, target_pos_pre_z[:2], backend)
        if self.adaptive_terrain:
            z_sampled = jax.random.uniform(zkey, minval=z_distance_range[0], maxval=z_distance_range[1])
            z_sampled = backend.where(hold_still, 0.0, z_sampled) 
            target_z = backend.maximum(z_sampled + stance_foot_pos[2], 0.0)

        target_pos = target_pos_pre_z.at[2].set(target_z)
        
        # when the terrain is adaptive, we need to build a pillar below the foot
        terrain_state = carry.terrain_state
        if self.adaptive_terrain:
            terrain_state = env._terrain.set_height_at_xy(carry.terrain_state, target_pos[:2], target_z, pillar_id_for_goal, backend)
        carry = carry.replace(terrain_state=terrain_state)

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
        final_yaw = self.wrap_to_pi(current_stance_yaw + clipped_yaw_displacement, backend)
        target_orn_rot = R.from_euler('z', final_yaw)
        # if hold still then the target should be the orientation of the stance, else the one computed
        target_orn = jax.lax.select(
            hold_still,
            stance_foot_orn,
            target_orn_rot.as_quat(scalar_first=True)
        )

        # ===============================================ASSIGN TARGETS===============================================
        # compute the num_gaits
        num_gaits = jax.lax.select(
            reset,
            1,
            backend.fmod(state.num_gaits + 1, self.max_num_gaits)    
        )
        
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
            feet_direction=feet_direction,
            # still_phase=hold_still,
            num_gaits=num_gaits,
        )

        # if need to hold still, modify the target such that it is the initial position
        if backend == np:
            target_pos = swing_foot_pos if reset else target_pos
            target_orn = swing_foot_orn if reset else target_orn
        else:
            target_pos = jax.lax.select(
                reset,
                swing_foot_pos,
                target_pos
            )
            target_orn = jax.lax.select(
                reset,
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

        return state, carry

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
        
        # manage the curriculum on the z just in case
        if self.curriculum:
            # compute steps elapsed in the curriculum phase
            steps_in_curriculum = state.steps - self.curriculum_start
            # compute the update condition
            should_update = (steps_in_curriculum > 0) & (steps_in_curriculum % self.update_z_every == 0) & (steps_in_curriculum <= self.curriculum_end)
            # prepare the update vector
            update_vector = backend.array([self.delta_pcurr_low, self.delta_pcurr_up])
            # prepare the (redundant but safe) clipping edges
            max_ranges = backend.array([self.max_z_distance_low, self.max_z_distance_up])
            # prepare the new z vector
            new_z_range = backend.minimum(state.z_distance_range + update_vector, max_ranges)
            
            if backend == np:
                state = state.replace(z_distance_range=new_z_range) if should_update else state
            else:
                state = jax.lax.cond(
                    should_update,
                    lambda: state.replace(z_distance_range=new_z_range),
                    lambda: state
                )
        
        # resample goal if needed
        if backend == np:
            # TODO: implement the resampling of the gait parameters
            if resample_goal:
                new_goal, carry = self.sample_goal(
                    env=env, data=data, carry=carry, backend=backend, initial_gait=gp, gait_frequency=state.gait_frequency,
                    distance_range=state.distance_range, angle_range_rad=state.angle_range_rad, 
                    movement_direction=state.movement_direction, feet_direction=state.feet_direction, 
                    z_distance_range=state.z_distance_range
                )
                state = new_goal
        else:
            # manage random keys
            key, sk1, sk2 = jax.random.split(carry.key, 3)
            carry = carry.replace(key=key)
            
            # check whether need to resample gait parameters
            resample_all = resample_goal & (state.num_gaits == 0)
            prev_gait_parameters = (
                state.movement_direction, state.feet_direction, gp, state.gait_frequency, state.distance_range, 
                state.angle_range_rad
            )
            new_gait_parameters = jax.lax.cond(
                resample_all,
                lambda: self._sample_gait_parameters(sk1),
                lambda: prev_gait_parameters
            )
            # movement_dir, feet_dir, gp0, gait_frequency, distance_range, angle_range_rad = new_gait_parameters
            movement_dir = state.movement_direction
            gp0 = gp
            _, feet_dir, _, gait_frequency, distance_range, angle_range_rad = new_gait_parameters
            
            # sample the probability of staying still
            hold_still = jax.lax.select(
                resample_all,
                jax.random.uniform(sk2) < self.still_proportion,
                state.still_phase
            )        
            state = state.replace(still_phase=hold_still)
            observation_states = carry.observation_states.replace(**{self.name: state})
            carry = carry.replace(observation_states=observation_states)
            
            # sample new goal
            state, carry = jax.lax.cond(
                resample_goal,
                lambda: self.sample_goal(
                    env=env, data=data, carry=carry, backend=backend, initial_gait=gp0, gait_frequency=gait_frequency,
                    distance_range=distance_range, angle_range_rad=angle_range_rad, 
                    movement_direction=movement_dir, feet_direction=feet_dir, reset=False, 
                    z_distance_range=state.z_distance_range
                ),
                lambda: (state, carry)
            )
    
            # newly get the gait process details
            gp = state.gait_process
            swing_foot_idx = state.swing_foot_idx

        # retireve info about both feet
        left_pos_w  = data.site_xpos[self._foot_site_id_left]
        left_mat_w  = data.site_xmat[self._foot_site_id_left].reshape(3, 3)
        left_quat_w = R.from_matrix(left_mat_w).as_quat(scalar_first=True)  # (w,x,y,z)

        right_pos_w  = data.site_xpos[self._foot_site_id_right]
        right_mat_w  = data.site_xmat[self._foot_site_id_right].reshape(3, 3)
        right_quat_w = R.from_matrix(right_mat_w).as_quat(scalar_first=True)

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
            lambda: (pos_offset, orn_offset, hold_pos, hold_orn),
            lambda: (hold_pos, hold_orn, pos_offset, orn_offset)
        )
        
        # craft GP array
        gp_info = backend.array([backend.cos(2 * backend.pi * gp), backend.sin(2 * backend.pi * gp)])
        
        # steady still condition 
        # steady_still_flag = state.still_phase & \
        #     (backend.abs(left_pos_w[0] - right_pos_w[0]) <= self.still_threshold) & \
        #     (backend.abs(left_pos_w[1] - right_pos_w[1] - self.still_feet_distance) <= self.still_threshold)
        steady_still_flag = state.still_phase & (state.num_gaits >= 2)
        zero_pos_off_l = backend.array([0.0, self.still_feet_distance, 0.0]) # backend.zeros(3, dtype=backend.float32)
        zero_pos_off_r = backend.array([0.0, - self.still_feet_distance, 0.0])
        zero_orn_off = backend.array([1, 0, 0, 0], dtype=backend.float32)
        gp_both_stance = backend.array([0, 0], dtype=backend.float32)
        left_pos_targ, left_orn_targ, right_pos_targ, right_orn_targ, gp_info = jax.lax.cond(
            steady_still_flag,
            lambda: (zero_pos_off_l, zero_orn_off, zero_pos_off_r, zero_orn_off, gp_both_stance),
            lambda: (left_pos_targ, left_orn_targ, right_pos_targ, right_orn_targ, gp_info)
        )
        """
        NOTE: this condition is verified whenever
        state.still_phase is True and the |x_left[0] - x_right[0]| <= epsilon
        In this case the gait information are [0,0] and the offset are all zero
        """
        
        observation = backend.concatenate(
            [
                left_pos_targ,
                left_orn_targ,
                right_pos_targ,
                right_orn_targ,
                gp_info
            ]
        )

        # verify the goal-based absorbing condition
        absorbing = False
        if self.goal_based_absorbing:
            # xy tracking only
            absorbing = backend.linalg.norm(stance_pos[:2] - stance_pos_target[:2]) > self.stance_absorbing_threshold

        # make the gait process progress
        gp_ada_offset = getattr(carry.control_func_state, "gait_phase_offset", 0.0)
        gp_offset = jax.lax.cond(
            self.is_gp_adaptive,
            lambda: gp_ada_offset,
            lambda: env.dt * state.gait_frequency
        )
        # gp = backend.fmod(gp + env.dt * state.gait_frequency, 1.0)
        gp = backend.fmod(gp + gp_offset, 1.0)
        state = state.replace(gait_process=gp, steps=(state.steps + 1), absorbing=absorbing) # update the steps too
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