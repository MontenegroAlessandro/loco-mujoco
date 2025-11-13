from types import ModuleType
from typing import Any, Union, Dict, Tuple, List
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
import mujoco
from mujoco import MjData, MjModel, MjSpec
from mujoco.mjx import Data, Model

from loco_mujoco.core.terrain import DynamicTerrain
from loco_mujoco.core.utils import mj_jntname2qposid
from loco_mujoco.core.utils.backend import assert_backend_is_supported

from jax.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R
from loco_mujoco.core.utils.math import (
    calculate_relative_site_quatities,
    quat_scalarfirst2scalarlast,
    quat_scalarlast2scalarfirst
)


@struct.dataclass
class ParkourTerrainState:
    """State for the complex obstacles terrain."""
    geom_pos: Union[np.ndarray, jax.Array]
    geom_quat: Union[np.ndarray, jax.Array]
    geom_size: Union[np.ndarray, jax.Array]
    geom_rgba: Union[np.ndarray, jax.Array]


class ParkourTerrain(DynamicTerrain):
    """
    Dynamic terrain that generates complex obstacle courses in one of 3 directions.
    Scenarios:
    1. Stairs (up) -> Slope (down)
    2. Slope (up) -> Stairs (down)
    3. Step (up) -> Flat -> Hole (step down)
    """

    viewer_needs_to_update_hfield: bool = False # the following class is based on geoms and not on heightfield

    def __init__(
        self, env: Any,
        n_obstacle_geoms: int = 50,
        inner_platform_size_in_meters: float = 1,
        obstacle_start_distance: float = 1,
        obstacle_width: float = 1,
        # variables for slopes and stairs
        stair_height: float = 0.1,
        stair_depth: float = 0.3,
        num_stairs: int = 4,
        slope_length: float = 2,
        # variables for the step
        step_height: float = 0.1,
        step_platform_length: float = 1,
        **kwargs: Any
    ):
        """
        Initialize the complex obstacles terrain.

        Args:
            env (Any): The environment instance.
            n_obstacle_geoms (int): number of geoms to be considered as new obstacles
            inner_platform_size_in_meters (float): Size of the flat starting platform.
            obstacle_start_distance (float): Min distance from platform edge to obstacle.
            obstacle_width (float): Width of all obstacles in meters.
            stair_height (float): Height of a single stair.
            stair_depth (float): Depth (length) of a single stair.
            num_stairs (int): Number of stairs to generate.
            slope_length (float): Length of the slope.
            step_height (float): Height of the step in scenario 3.
            step_platform_length (float): Length of the flat area after the step.
            **kwargs (Any): Additional arguments for initialization.
        """
        # store geom pool parameters
        self.n_obstacle_geoms = n_obstacle_geoms
        self._geom_name_prefix = "obstacle_geom_"
        self._geom_names = [f"{self._geom_name_prefix}{i}" for i in range(self.n_obstacle_geoms)]
        self._compile_time_size = jnp.array([0.001, 0.001, 0.001])
        self._hidden_pos = jnp.array([0.0, 0.0, -10.0])
        self._hidden_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        self._hidden_size = jnp.array([0.0, 0.0, 0.0])
        self._default_rgba = jnp.array([0.4, 0.5, 0.6, 1.0])
        self._floor_height = 0.0
        
        # super class initialization
        super().__init__(env, **kwargs)

        # central flat stage
        self.inner_platform_size_in_meters = inner_platform_size_in_meters
        
        # Terrain Generation Parameters 
        self.obstacle_start_distance = obstacle_start_distance
        self.obstacle_width = obstacle_width
        # stairs and slopes
        self.stair_height = stair_height
        self.stair_depth = stair_depth
        self.num_stairs = num_stairs
        self.slope_length = slope_length
        self.slope_height = stair_height * num_stairs
        # steps
        self.step_height = step_height
        self.step_platform_length = step_platform_length

        # Platform cutout
        self.platform_half_size = self.inner_platform_size_in_meters / 2.0 
        
        # pre-compute world dimensions
        self.start_x = self.platform_half_size + self.obstacle_start_distance
        self.center_y = 0.0
        root_free_joint_xml_name = env.root_free_joint_xml_name 
        self._free_jnt_qpos_id = np.array(mj_jntname2qposid(root_free_joint_xml_name, env._model))
        self._geom_ids = np.array([mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in self._geom_names])

    def init_state(
            self, env: Any,
            key: Any,
            model: Union[MjModel, Model],
            data: Union[MjData, Data],
            backend: ModuleType
        ) -> ParkourTerrainState:
        """Initialize the state of the complex obstacles terrain."""
        # check if the backend is supported
        assert_backend_is_supported(backend)
        
        # get the geoms ids
        if backend == jnp:
            geom_ids = jnp.array(
                [mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in self._geom_names]
            )
            self._geom_ids = geom_ids # class variable switched to jax backend
        else:
            geom_ids = self._geom_ids
            
        # initialize the geoms to a hidden state
        init_pos = backend.tile(self._hidden_pos, (self.n_obstacle_geoms, 1))
        init_quat = backend.tile(self._hidden_quat, (self.n_obstacle_geoms, 1))
        init_size = backend.tile(self._hidden_size, (self.n_obstacle_geoms, 1))
        init_rgba = backend.tile(self._default_rgba, (self.n_obstacle_geoms, 1))
        
        # return the state
        return ParkourTerrainState(
            geom_pos=init_pos,
            geom_quat=init_quat,
            geom_size=init_size,
            geom_rgba=init_rgba
        )

    def modify_spec(self, spec: MjSpec) -> MjSpec:
        # delete the initial floor
        for g in spec.geoms:
            if g.name == "floor":
                g.delete()
                break
        
        # add a new flat terrain
        wb = spec.worldbody
        wb.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(4.0, 4.0, 1.0), # 8m x 8m plane
            pos=(0, 0, self._floor_height),
            material="MatPlane",
            group=2
        )
        
        # add the new geoms for the obstacles
        for name in self._geom_names:
            wb.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=self._compile_time_size, # self._hidden_size,
                pos=self._hidden_pos,
                material="MatPlane",
                group=2
            )
        
        return spec

    def reset(
            self, env: Any,
            model: Union[MjModel, Model], data: Union[MjData, Data], carry: Any,
            backend: ModuleType
        ) -> Tuple[Union[MjData, Data], Any]:
        # check if the backend is supported
        assert_backend_is_supported(backend)
        
        # import rotation module
        R = jnp_R if backend == jnp else np_R
        
        # initialize position arrays
        geom_pos = backend.tile(self._hidden_pos, (self.n_obstacle_geoms, 1))
        geom_quat = backend.tile(self._hidden_quat, (self.n_obstacle_geoms, 1))
        geom_size = backend.tile(self._hidden_size, (self.n_obstacle_geoms, 1))
        geom_rgba = backend.tile(self._default_rgba, (self.n_obstacle_geoms, 1))
        
        # Generate the unscaled heightmap
        if backend == jnp:
            key = carry.key
            key, subkey_scenario, subkey_direction, subkey_gen = jax.random.split(key, 4)
            # Sample scenario (0, 1, or 2)
            scenario_idx = jax.random.randint(subkey_scenario, shape=(), minval=0, maxval=3)
            # Sample direction (0=front, 1=left, 2=right)
            direction_idx = jax.random.randint(subkey_direction, shape=(), minval=0, maxval=3)
            # generate geoms
            geom_pos, geom_quat, geom_size, geom_rgba = self._jnp_generate_geoms(
                subkey_gen, scenario_idx, direction_idx,
                geom_pos, geom_quat, geom_size, geom_rgba
            )
            carry = carry.replace(key=key)
        else:
            # Sample scenario (0, 1, or 2)
            scenario_idx = np.random.randint(0, 3)
            # Sample direction (0=front, 1=left, 2=right)
            direction_idx = np.random.randint(0, 3)
            
            geom_pos, geom_quat, geom_size, geom_rgba = self._np_generate_geoms(
                scenario_idx, direction_idx,
                geom_pos, geom_quat, geom_size, geom_rgba
            )
        
        # Store in state
        terrain_state = ParkourTerrainState(
            geom_pos=geom_pos,
            geom_quat=geom_quat,
            geom_size=geom_size,
            geom_rgba=geom_rgba
        )
        carry = carry.replace(terrain_state=terrain_state)

        return data, carry

    def _jnp_generate_geoms(
        self, key: Any, scenario_idx: jax.Array, direction_idx: jax.Array, pos, quat, size, rgba) -> jnp.ndarray:
        """Generates, rotates, and places the chosen JAX terrain scenario."""
        R = jnp_R
        
        # Generate the base scenario (always in +x direction)
        pos, quat, size, rgba = jax.lax.switch(
            scenario_idx,
            [
                lambda p,q,s,r: self._jnp_scenario_1(p,q,s,r),
                lambda p,q,s,r: self._jnp_scenario_2(p,q,s,r),
                lambda p,q,s,r: self._jnp_scenario_3(p,q,s,r)
            ],
            pos, quat, size, rgba
        )
        
        # Rotate the field based on the chosen direction
        # direction_idx: 0=front(no-op), 1=left(rot90), 2=right(rot-90)
        yaw_angle = jax.lax.switch(
            direction_idx,
            [
                lambda: 0.0,
                lambda: jnp.pi / 2.0,   # left
                lambda: -jnp.pi / 2.0   # right
            ]
        )
        scene_rot = R.from_euler('z', yaw_angle)
        
        # apply the rotation to the geoms that are active (i.e., the ones with a size != 0)
        is_active = (size[:, 0] > 0)
        
        rotated_pos = scene_rot.apply(pos)
        pos = jnp.where(is_active[:, jnp.newaxis], rotated_pos, pos)
        
        rotated_quat = (R.from_quat(quat_scalarfirst2scalarlast(quat)) * scene_rot).as_quat(scalar_first=True)
        quat = jnp.where(is_active[:, jnp.newaxis], rotated_quat, quat)
        
        return pos, quat, size, rgba

    def _jnp_scenario_1(self, pos, quat, size, rgba) -> jnp.ndarray:
        """JAX: Stairs (up) -> Slope (down)"""
        R = jnp_R
        
        # generate stairs up
        box_half_size = jnp.array([self.stair_depth/2.0, self.obstacle_width/2.0, self.stair_height/2.0])
        start_x_pos = self.start_x + self.stair_depth/2.0
        
        def stair_body(i, state):
            # unpack the state
            pos, quat, size, rgba = state
            
            # position of the center of the stair box
            x_pos = start_x_pos + i * self.stair_depth
            y_pos = self.center_y
            z_pos = self._floor_height + (i + 0.5) * self.stair_height
            
            pos = pos.at[i].set(jnp.array([x_pos, y_pos, z_pos]))
            quat = quat.at[i].set(self._hidden_quat)
            size = size.at[i].set(box_half_size)
            
            return (pos, quat, size, rgba)
        
        (pos, quat, size, rgba) = jax.lax.fori_loop(
            0, self.num_stairs, stair_body, (pos, quat, size, rgba)
        )
        
        # generate slope down
        slope_i = self.num_stairs
        peak_height = self.num_stairs * self.stair_height
        
        slope_half_size = jnp.array([self.slope_length / 2.0, self.obstacle_width / 2.0, 0.01])
        
        pitch_angle = -jnp.arctan2(peak_height, self.slope_length)
        slope_rot_quat = R.from_euler('z', pitch_angle).as_quat(scalar_first=True)
        
        x_start = self.start_x + self.num_stairs * self.stair_depth
        z_start = self._floor_height + peak_height
        x_center = x_start + (self.slope_length / 2.0) * jnp.cos(pitch_angle)
        y_center = self.center_y
        z_center = z_start + (self.slope_length / 2.0) * jnp.sin(pitch_angle)
        
        pos = pos.at[slope_i].set(jnp.array([x_center, y_center, z_center]))
        quat = quat.at[slope_i].set(slope_rot_quat)
        size = size.at[slope_i].set(slope_half_size.at[2].set(0.05))
        
        return pos, quat, size, rgba

    def _jnp_scenario_2(self, pos, quat, size, rgba) -> jnp.ndarray:
        R = jnp_R
        
        # slope up then stairs down
        # generate the slope
        slope_i = 0
        peak_height = self.slope_height
        
        slope_half_size = jnp.array([self.slope_length / 2.0, self.obstacle_width / 2.0, 0.05])
        
        pitch_angle = -jnp.arctan2(peak_height, self.slope_length)
        slope_rot_quat = R.from_euler('z', pitch_angle).as_quat(scalar_first=True)
        
        x_start = self.start_x
        z_start = self._floor_height
        x_center = x_start + (self.slope_length / 2.0) * jnp.cos(pitch_angle)
        y_center = self.center_y
        z_center = z_start + (self.slope_length / 2.0) * jnp.sin(pitch_angle)
        
        pos = pos.at[slope_i].set(jnp.array([x_center, y_center, z_center]))
        quat = quat.at[slope_i].set(slope_rot_quat)
        size = size.at[slope_i].set(slope_half_size)
        
        # generate stairs 
        box_half_size = jnp.array([self.stair_depth/2.0, self.obstacle_width/2.0, self.stair_height/2.0])
        stair_start_x = self.start_x + self.slope_length
        
        def stair_body(i, state):
            # unpack the state
            pos, quat, size, rgba = state
            geom_idx = i + 1
            
            # position of the center of the stair box
            x_pos = stair_start_x + (i + 0.5) * self.stair_depth
            y_pos = self.center_y
            z_pos = self._floor_height + peak_height - (i + 0.5) * self.stair_height
            
            pos = pos.at[geom_idx].set(jnp.array([x_pos, y_pos, z_pos]))
            quat = quat.at[geom_idx].set(self._hidden_quat)
            size = size.at[geom_idx].set(box_half_size)
            
            return (pos, quat, size, rgba)
        
        (pos, quat, size, rgba) = jax.lax.fori_loop(
            0, self.num_stairs, stair_body, (pos, quat, size, rgba)
        )
        
        return pos, quat, size, rgba
           
    def _jnp_scenario_3(self, pos, quat, size, rgba) -> jnp.ndarray:
        """JAX: Step (up) -> Flat -> step down"""
        step_half_size = jnp.array([self.step_platform_length / 2.0, self.obstacle_width / 2.0, self.step_height / 2.0])
        x_center = self.start_x + self.step_platform_length / 2.0
        z_center = self._floor_height + self.step_height / 2.0
        
        pos = pos.at[0].set(jnp.array([x_center, self.center_y, z_center]))
        quat = quat.at[0].set(self._hidden_quat)
        size = size.at[0].set(step_half_size)
        
        return pos, quat, size, rgba

    def _np_generate_geoms(self, scenario_idx: int, direction_idx: int, pos, quat, size, rgba) -> np.ndarray:
        R = np_R
        """NumPy equivalent of _jnp_generate_geoms."""
        # Generate the base scenario (always in +x direction)
        if scenario_idx == 0:
            pos, quat, size, rgba = self._np_scenario_1(pos, quat, size, rgba)
        elif scenario_idx == 1:
            pos, quat, size, rgba = self._np_scenario_2(pos, quat, size, rgba)
        else:
            pos, quat, size, rgba = self._np_scenario_3(pos, quat, size, rgba)

        # Rotate the field based on the chosen direction
        # direction_idx: 0=front(no-op), 1=left(rot90), 2=right(rot-90)
        if direction_idx == 0:
            yaw_angle = 0.0
        elif direction_idx == 1:
            yaw_angle = np.pi / 2.0
        else:
            yaw_angle = -np.pi / 2.0

        scene_rot = R.from_euler('z', yaw_angle)

        # apply the rotation to the geoms that are active (i.e., the ones with a size != 0)
        is_active = (size[:, 0] > 0)

        rotated_pos = scene_rot.apply(pos)
        pos = np.where(is_active[:, np.newaxis], rotated_pos, pos)

        # rotate quaternions and keep scalar-first ordering as in the rest of the code
        rotated_quat = (R.from_quat(quat_scalarfirst2scalarlast(quat)) * scene_rot).as_quat(scalar_first=True)
        quat = np.where(is_active[:, np.newaxis], rotated_quat, quat)

        return pos, quat, size, rgba

    def _np_scenario_1(self, pos, quat, size, rgba) -> np.ndarray:
        R = np_R
        """NumPy: Stairs (up) -> Slope (down) -- numpy equivalent of _jnp_scenario_1"""
        # generate stairs up
        box_half_size = np.array([self.stair_depth / 2.0, self.obstacle_width / 2.0, self.stair_height / 2.0])
        start_x_pos = self.start_x + self.stair_depth / 2.0

        hidden_quat_np = np.asarray(self._hidden_quat)

        for i in range(self.num_stairs):
            x_pos = start_x_pos + i * self.stair_depth
            y_pos = self.center_y
            z_pos = self._floor_height + (i + 0.5) * self.stair_height

            pos[i] = np.array([x_pos, y_pos, z_pos], dtype=pos.dtype)
            quat[i] = hidden_quat_np
            size[i] = box_half_size

        # generate slope down
        slope_i = self.num_stairs
        peak_height = self.num_stairs * self.stair_height

        slope_half_size = np.array([self.slope_length / 2.0, self.obstacle_width / 2.0, 0.01])
        pitch_angle = -np.arctan2(peak_height, self.slope_length)
        # keep scalar-first quaternion ordering
        slope_rot_quat = R.from_euler('z', pitch_angle).as_quat(scalar_first=True)

        x_start = self.start_x + self.num_stairs * self.stair_depth
        z_start = self._floor_height + peak_height
        x_center = x_start + (self.slope_length / 2.0) * np.cos(pitch_angle)
        y_center = self.center_y
        z_center = z_start + (self.slope_length / 2.0) * np.sin(pitch_angle)

        pos[slope_i] = np.array([x_center, y_center, z_center], dtype=pos.dtype)
        quat[slope_i] = slope_rot_quat
        slope_size = slope_half_size.copy()
        slope_size[2] = 0.05
        size[slope_i] = slope_size

        return pos, quat, size, rgba

    def _np_scenario_2(self, pos, quat, size, rgba) -> np.ndarray:
        R = np_R
        """NumPy: Slope (up) -> Stairs (down) -- numpy equivalent of _jnp_scenario_2"""
        # slope up
        slope_i = 0
        peak_height = self.slope_height

        slope_half_size = np.array([self.slope_length / 2.0, self.obstacle_width / 2.0, 0.05])

        pitch_angle = -np.arctan2(peak_height, self.slope_length)
        slope_rot_quat = R.from_euler('z', pitch_angle).as_quat(scalar_first=True)

        x_start = self.start_x
        z_start = self._floor_height
        x_center = x_start + (self.slope_length / 2.0) * np.cos(pitch_angle)
        y_center = self.center_y
        z_center = z_start + (self.slope_length / 2.0) * np.sin(pitch_angle)

        pos[slope_i] = np.array([x_center, y_center, z_center], dtype=pos.dtype)
        quat[slope_i] = slope_rot_quat
        size[slope_i] = slope_half_size

        # generate stairs down
        box_half_size = np.array([self.stair_depth / 2.0, self.obstacle_width / 2.0, self.stair_height / 2.0])
        stair_start_x = self.start_x + self.slope_length

        hidden_quat_np = np.asarray(self._hidden_quat)

        for i in range(self.num_stairs):
            geom_idx = i + 1
            x_pos = stair_start_x + (i + 0.5) * self.stair_depth
            y_pos = self.center_y
            z_pos = self._floor_height + peak_height - (i + 0.5) * self.stair_height

            pos[geom_idx] = np.array([x_pos, y_pos, z_pos], dtype=pos.dtype)
            quat[geom_idx] = hidden_quat_np
            size[geom_idx] = box_half_size

        return pos, quat, size, rgba

    def _np_scenario_3(self, pos, quat, size, rgba) -> np.ndarray:
        """NumPy: Step (up) -> Flat -> step down -- numpy equivalent of _jnp_scenario_3"""
        step_half_size = np.array([self.step_platform_length / 2.0,
                                   self.obstacle_width / 2.0,
                                   self.step_height / 2.0])
        x_center = self.start_x + self.step_platform_length / 2.0
        z_center = self._floor_height + self.step_height / 2.0

        pos[0] = np.array([x_center, self.center_y, z_center], dtype=pos.dtype)
        quat[0] = np.asarray(self._hidden_quat)
        size[0] = step_half_size

        return pos, quat, size, rgba

    def get_height_at_xy(
            self, 
            terrain_state: ParkourTerrainState, 
            xy_pos: Union[np.ndarray, jnp.ndarray], 
            backend: ModuleType
        ) -> Union[float, jax.Array]:
        """
        Get the terrain height (in meters) at a specific world (x, y) coordinate
        by checking against all active geoms.
        """
        assert_backend_is_supported(backend)
        R = jnp_R if backend == jnp else np_R
        
        # Start with the base floor height
        max_height = backend.full_like(xy_pos[0], self._floor_height)
        
        geom_pos = terrain_state.geom_pos
        geom_quat = terrain_state.geom_quat
        geom_size = terrain_state.geom_size

        def check_geom(i, max_h):
            pos = geom_pos[i]
            quat = geom_quat[i]
            size = geom_size[i]
            
            # --- 1. Check if point is inside the box's 2D footprint ---
            # Get inverse rotation (transpose of matrix)
            R_mat = R.from_quat(quat).as_matrix()
            R_inv = R_mat.T
            
            # Get vector from box center to xy_pos (ignoring z)
            p_world_xy = backend.array([xy_pos[0] - pos[0], xy_pos[1] - pos[1]])
            
            # Rotate this vector into the box's local frame
            # We only need the x and y components of the 3x3 R_inv
            local_x = p_world_xy[0] * R_inv[0, 0] + p_world_xy[1] * R_inv[1, 0]
            local_y = p_world_xy[0] * R_inv[0, 1] + p_world_xy[1] * R_inv[1, 1]
            
            is_inside_x = (backend.abs(local_x) < size[0])
            is_inside_y = (backend.abs(local_y) < size[1])
            is_inside = is_inside_x & is_inside_y & (size[0] > 0.0) # Check for active geom
            
            # --- 2. If inside, calculate the height of the box's top plane ---
            # Plane equation: n · (p - p0) = 0
            # n = box's local Z-axis in world frame (3rd column of R_mat)
            n = R_mat[:, 2]
            
            # p0 = a point on the top plane (center of top face)
            p_top_center = pos + R.from_quat(quat).apply(backend.array([0.0, 0.0, size[2]]))
            
            # Solve for z_world in:
            # n[0]*(x - p0[0]) + n[1]*(y - p0[1]) + n[2]*(z_world - p0[2]) = 0
            n_z_safe = backend.where(n[2] == 0, 1e-6, n[2]) # Avoid divide by zero
            z_world_top = p_top_center[2] - \
                          (n[0] * (xy_pos[0] - p_top_center[0]) + 
                           n[1] * (xy_pos[1] - p_top_center[1])) / n_z_safe
            
            # Check for "hole" (negative height)
            # If hole, we want the *bottom* plane's height
            p_bottom_center = pos + R.from_quat(quat).apply(backend.array([0.0, 0.0, -size[2]]))
            z_world_bottom = p_bottom_center[2] - \
                             (n[0] * (xy_pos[0] - p_bottom_center[0]) + 
                              n[1] * (xy_pos[1] - p_bottom_center[1])) / n_z_safe
                             
            # A "hole" is a box centered below z=0
            is_hole = (pos[2] < self._floor_height)
            
            # If it's a hole, the height is the bottom of the box.
            # If it's a step, the height is the top of the box.
            geom_height = backend.where(is_hole, z_world_bottom, z_world_top)

            return backend.where(is_inside, backend.maximum(max_h, geom_height), max_h)

        if backend == jnp:
            max_height = jax.lax.fori_loop(0, self.n_obstacle_geoms, check_geom, max_height)
        else:
            for i in range(self.n_obstacle_geoms):
                max_height = check_geom(i, max_height)
        
        return max_height

    def update(self, env: Any,
               model: Union[MjModel, Model],
               data: Union[MjData, Data],
               carry: Any,
               backend: ModuleType) -> Tuple[Union[MjModel, Model], Union[MjData, Data], Any]:
        """Update the terrain by setting all geom properties from the state."""
        assert_backend_is_supported(backend)
        terrain_state = carry.terrain_state
        
        if backend == jnp:
            # JAX needs to update the model immutably
            model = model.replace(
                geom_pos=model.geom_pos.at[self._geom_ids].set(terrain_state.geom_pos),
                geom_quat=model.geom_quat.at[self._geom_ids].set(terrain_state.geom_quat),
                geom_size=model.geom_size.at[self._geom_ids].set(terrain_state.geom_size),
                geom_rgba=model.geom_rgba.at[self._geom_ids].set(terrain_state.geom_rgba)
            )
        else:
            # NumPy can update the model in place
            model.geom_pos[self._geom_ids] = terrain_state.geom_pos
            model.geom_quat[self._geom_ids] = terrain_state.geom_quat
            model.geom_size[self._geom_ids] = terrain_state.geom_size
            model.geom_rgba[self._geom_ids] = terrain_state.geom_rgba

        data = self._reset_on_edge(data, backend)
        return model, data, carry

    def get_height_matrix(self, matrix_config: Dict[str, Any],
                          env: Any,
                          model: Union[MjModel, Model],
                          data: Union[MjData, Data],
                          carry: Any,
                          backend: ModuleType) -> Union[np.ndarray, jnp.ndarray]:
        assert_backend_is_supported(backend)
        raise NotImplementedError("get_height_matrix is not implemented for geom-based terrain.")

    def isaac_hf_to_mujoco_hf(self,
                              isaac_hf: Union[np.ndarray, jnp.ndarray],
                              backend: ModuleType) -> Union[np.ndarray, jnp.ndarray]:
        """Not used in this class."""
        assert_backend_is_supported(backend)
        return backend.array([])

    def _reset_on_edge(self, data: Union[MjData, Data],
                       backend: ModuleType) -> Union[MjData, Data]:
        """Reset the robot position if it is on the edge of the terrain."""
        assert_backend_is_supported(backend)

        # Using a simpler platform check now
        com_pos = data.qpos[self._free_jnt_qpos_id][:2]
        
        # Reset if outside a 3.5m radius (for an 8x8m world)
        reached_edge = (com_pos[0]**2 + com_pos[1]**2) > (3.5**2)
        
        free_jnt_xy = self._free_jnt_qpos_id[:2]
        if backend == jnp:
            init_data = data.replace(qpos=data.qpos.at[free_jnt_xy].set(0.0))
            data = jax.lax.cond(reached_edge, lambda _: init_data, lambda _: data, None)
        else:
            if reached_edge:
                data.qpos[free_jnt_xy] = 0.0

        return data