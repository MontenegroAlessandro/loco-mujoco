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


@struct.dataclass
class ParkourTerrainState:
    """State for the complex obstacles terrain."""
    height_field_raw: Union[np.ndarray, jax.Array] # The raw, scaled hfield data for MuJoCo (80x80 flattened).
    height_field_unscaled: Union[np.ndarray, jax.Array] # The unscaled heightmap in meters (80x80).


class ParkourTerrain(DynamicTerrain):
    """
    Dynamic terrain that generates complex obstacle courses in one of 3 directions.
    Scenarios:
    1. Stairs (up) -> Slope (down)
    2. Slope (up) -> Stairs (down)
    3. Step (up) -> Flat -> Hole (step down)
    """

    viewer_needs_to_update_hfield: bool = True

    def __init__(
        self, env: Any,
        inner_platform_size_in_meters: float = 1.0,
        obstacle_start_distance: float = 0.5,
        obstacle_width: float = 1.0,
        # Scenario 1 & 2 (Stairs/Slope)
        stair_height: float = 0.1,
        stair_depth: float = 0.3,
        num_stairs: int = 4,
        slope_length: float = 2.0,
        # Scenario 3 (Step/Hole)
        step_height: float = 0.15,
        step_platform_length: float = 1.0,
        hole_depth: float = -0.15,
        obstacle_length: float = 0.5, # Length of the step/hole itself
        **kwargs: Any
    ):
        """
        Initialize the complex obstacles terrain.

        Args:
            env (Any): The environment instance.
            inner_platform_size_in_meters (float): Size of the flat starting platform.
            obstacle_start_distance (float): Min distance from platform edge to obstacle.
            obstacle_width (float): Width of all obstacles in meters.
            stair_height (float): Height of a single stair.
            stair_depth (float): Depth (length) of a single stair.
            num_stairs (int): Number of stairs to generate.
            slope_length (float): Length of the slope.
            step_height (float): Height of the step in scenario 3.
            step_platform_length (float): Length of the flat area after the step.
            hole_depth (float): Depth of the hole in scenario 3 (should be negative).
            obstacle_length (float): Length of the step-up and step-down sections.
            **kwargs (Any): Additional arguments for initialization.
        """
        super().__init__(env, **kwargs)

        self.inner_platform_size_in_meters = inner_platform_size_in_meters
        
        # --- Terrain Generation Parameters ---
        self.obstacle_start_distance = obstacle_start_distance
        self.obstacle_width = obstacle_width
        # Scenario 1 & 2
        self.stair_height = stair_height
        self.stair_depth = stair_depth
        self.num_stairs = num_stairs
        self.slope_length = slope_length
        self.slope_height = stair_height * num_stairs # Ensure slope/stairs match
        # Scenario 3
        self.step_height = step_height
        self.step_platform_length = step_platform_length
        self.hole_depth = hole_depth
        self.obstacle_length = obstacle_length
        # --- End Terrain Params ---

        # Hfield parameters (copied from StonesHolesTerrain)
        self.hfield_size = (4, 4, 30.0, 0.125) # (half_x, half_y, z_scale, sample_space)
        self.hfield_length = 80 # hfield resolution (pixels)
        self.hfield_half_length_in_meters = self.hfield_size[0]
        self.max_possible_height = self.hfield_size[2]
        self.one_meter_length = int(self.hfield_length / (self.hfield_half_length_in_meters * 2))
        self.hfield_half_length = self.hfield_length // 2
        self.mujoco_height_scaling = self.max_possible_height

        # Platform cutout (copied from StonesHolesTerrain)
        platform_size = int(self.inner_platform_size_in_meters * self.one_meter_length)
        self.x1 = self.hfield_half_length - (platform_size // 2)
        self.y1 = self.hfield_half_length - (platform_size // 2)
        self.x2 = self.hfield_half_length + (platform_size // 2)
        self.y2 = self.hfield_half_length + (platform_size // 2)
        
        # Pre-calculate pixel dimensions for generation functions
        self.obstacle_width_px = int(self.obstacle_width * self.one_meter_length)
        self.start_i_px = self.hfield_half_length + int(self.obstacle_start_distance * self.one_meter_length)
        self.start_j_px = self.hfield_half_length - (self.obstacle_width_px // 2)
        self.end_j_px = self.start_j_px + self.obstacle_width_px
        
        self.stair_depth_px = int(self.stair_depth * self.one_meter_length)
        self.slope_length_px = int(self.slope_length * self.one_meter_length)
        self.obstacle_length_px = int(self.obstacle_length * self.one_meter_length)
        self.step_platform_length_px = int(self.step_platform_length * self.one_meter_length)
        
        root_free_joint_xml_name = env.root_free_joint_xml_name
        self._free_jnt_qpos_id = np.array(mj_jntname2qposid(root_free_joint_xml_name, env._model))

    def init_state(
            self, env: Any,
            key: Any,
            model: Union[MjModel, Model],
            data: Union[MjData, Data],
            backend: ModuleType
        ) -> ParkourTerrainState:
        """Initialize the state of the complex obstacles terrain."""
        assert_backend_is_supported(backend)
        return ParkourTerrainState(
            height_field_raw=backend.zeros(self.hfield_length * self.hfield_length),
            height_field_unscaled=backend.zeros((self.hfield_length, self.hfield_length))
        )

    def modify_spec(self, spec: MjSpec) -> MjSpec:
        """Modify the simulation specification (Identical to RoughTerrain)."""
        file_name = Path(__file__).resolve().parent.parent.parent / "models" / "common" / "default_hfield_80.png"
        spec.add_hfield(name='complex_obstacles_terrain', size=self.hfield_size, file=str(file_name))
        for i, field in enumerate(spec.hfields):
            if field.name == 'complex_obstacles_terrain':
                self.hfield_id = i
                break

        for g in spec.geoms:
            if g.name == 'floor':
                g.delete()
                break

        wb = spec.worldbody
        wb.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname='complex_obstacles_terrain', group=2,
                    pos=(0, 0, -0.06), material="MatPlane", rgba=(0.8, 0.9, 0.8, 1))
        return spec

    def reset(
            self, env: Any,
            model: Union[MjModel, Model], data: Union[MjData, Data], carry: Any,
            backend: ModuleType
        ) -> Tuple[Union[MjData, Data], Any]:
        """Reset the terrain by generating a new random scenario in a random direction."""
        assert_backend_is_supported(backend)
        
        # Generate the unscaled heightmap
        if backend == jnp:
            key = carry.key
            key, subkey_scenario, subkey_direction, subkey_gen = jax.random.split(key, 4)
            # Sample scenario (0, 1, or 2)
            scenario_idx = jax.random.randint(subkey_scenario, shape=(), minval=0, maxval=3)
            # Sample direction (0=front, 1=left, 2=right)
            direction_idx = jax.random.randint(subkey_direction, shape=(), minval=0, maxval=3)
            
            height_field_unscaled = self._jnp_generate_terrain(
                subkey_gen, scenario_idx, direction_idx
            )
            carry = carry.replace(key=key)
        else:
            # Sample scenario (0, 1, or 2)
            scenario_idx = np.random.randint(0, 3)
            # Sample direction (0=front, 1=left, 2=right)
            direction_idx = np.random.randint(0, 3)
            
            height_field_unscaled = self._np_generate_terrain(
                scenario_idx, direction_idx
            )

        # Cut out the flat starting platform (applied after rotation)
        if backend == jnp:
            height_field_unscaled = height_field_unscaled.at[self.x1:self.x2, self.y1:self.y2].set(0.0)
        else:
            height_field_unscaled[self.x1:self.x2, self.y1:self.y2] = 0.0

        # Convert to MuJoCo-scaled format
        height_field_raw = self.isaac_hf_to_mujoco_hf(height_field_unscaled, backend)
        
        # Store in state
        terrain_state = ParkourTerrainState(
            height_field_raw=height_field_raw,
            height_field_unscaled=height_field_unscaled
        )
        carry = carry.replace(terrain_state=terrain_state)

        return data, carry

    # --- JAX Generation ---

    def _jnp_generate_terrain(self, key: Any, scenario_idx: jax.Array, direction_idx: jax.Array) -> jnp.ndarray:
        """Generates, rotates, and places the chosen JAX terrain scenario."""
        
        # 1. Generate the base scenario (always in +x direction)
        scenario_field = jax.lax.switch(
            scenario_idx,
            [self._jnp_scenario_1, self._jnp_scenario_2, self._jnp_scenario_3],
            key
        )
        
        # 2. Rotate the field based on the chosen direction
        # direction_idx: 0=front(no-op), 1=left(rot90), 2=right(rot-90)
        final_field = jax.lax.switch(
            direction_idx,
            [
                lambda x: x,                     # 0: Front
                lambda x: jnp.rot90(x, k=1),     # 1: Left
                lambda x: jnp.rot90(x, k=-1)     # 2: Right
            ],
            scenario_field
        )
        
        return final_field

    def _jnp_scenario_1(self, key: Any) -> jnp.ndarray:
        """JAX: Stairs (up) -> Slope (down)"""
        hf = jnp.zeros((self.hfield_length, self.hfield_length))
        
        # 1. Generate Stairs (Up)
        def stair_body(i, hf_and_i):
            hf, current_i = hf_and_i
            current_height = (i + 1) * self.stair_height
            next_i = current_i + self.stair_depth_px
            # Clip indices to prevent out-of-bounds
            idx_start = jnp.clip(current_i, 0, self.hfield_length)
            idx_end = jnp.clip(next_i, 0, self.hfield_length)
            hf = hf.at[idx_start:idx_end, self.start_j_px:self.end_j_px].set(current_height)
            return (hf, next_i)
        
        (hf, slope_start_i) = jax.lax.fori_loop(
            0, self.num_stairs, stair_body, (hf, self.start_i_px)
        )
        
        # 2. Generate Slope (Down)
        slope_end_i = slope_start_i + self.slope_length_px
        peak_height = self.num_stairs * self.stair_height
        
        # Create a linear ramp from peak_height to 0.0
        x = jnp.linspace(peak_height, 0.0, self.slope_length_px)
        ramp = x[:, jnp.newaxis] # Broadcast across width
        
        # Clip indices
        idx_start = jnp.clip(slope_start_i, 0, self.hfield_length)
        idx_end = jnp.clip(slope_end_i, 0, self.hfield_length)
        # Adjust ramp length if clipped
        ramp = ramp[:(idx_end - idx_start), :] 
        
        hf = hf.at[idx_start:idx_end, self.start_j_px:self.end_j_px].set(ramp)
        return hf

    def _jnp_scenario_2(self, key: Any) -> jnp.ndarray:
        """JAX: Slope (up) -> Stairs (down)"""
        hf = jnp.zeros((self.hfield_length, self.hfield_length))

        # 1. Generate Slope (Up)
        slope_start_i = self.start_i_px
        slope_end_i = slope_start_i + self.slope_length_px
        peak_height = self.slope_height
        
        x = jnp.linspace(0.0, peak_height, self.slope_length_px)
        ramp = x[:, jnp.newaxis]
        
        # Clip indices
        idx_start = jnp.clip(slope_start_i, 0, self.hfield_length)
        idx_end = jnp.clip(slope_end_i, 0, self.hfield_length)
        ramp = ramp[:(idx_end - idx_start), :]
        
        hf = hf.at[idx_start:idx_end, self.start_j_px:self.end_j_px].set(ramp)
        
        # 2. Generate Stairs (Down)
        def stair_body(i, hf_and_i):
            hf, current_i = hf_and_i
            # Height decreases
            current_height = peak_height - (i * self.stair_height)
            next_i = current_i + self.stair_depth_px
            
            idx_start = jnp.clip(current_i, 0, self.hfield_length)
            idx_end = jnp.clip(next_i, 0, self.hfield_length)
            
            hf = hf.at[idx_start:idx_end, self.start_j_px:self.end_j_px].set(current_height)
            return (hf, next_i)

        (hf, _) = jax.lax.fori_loop(
            0, self.num_stairs, stair_body, (hf, slope_end_i)
        )
        return hf
        
    def _jnp_scenario_3(self, key: Any) -> jnp.ndarray:
        """JAX: Step (up) -> Flat -> Hole (step down)"""
        hf = jnp.zeros((self.hfield_length, self.hfield_length))

        # 1. Generate Step (Up)
        step_start_i = self.start_i_px
        step_end_i = step_start_i + self.obstacle_length_px
        idx_start = jnp.clip(step_start_i, 0, self.hfield_length)
        idx_end = jnp.clip(step_end_i, 0, self.hfield_length)
        hf = hf.at[idx_start:idx_end, self.start_j_px:self.end_j_px].set(self.step_height)
        
        # 2. Generate Flat Platform
        platform_start_i = step_end_i
        platform_end_i = platform_start_i + self.step_platform_length_px
        idx_start = jnp.clip(platform_start_i, 0, self.hfield_length)
        idx_end = jnp.clip(platform_end_i, 0, self.hfield_length)
        hf = hf.at[idx_start:idx_end, self.start_j_px:self.end_j_px].set(self.step_height)

        # 3. Generate Hole (Down)
        hole_start_i = platform_end_i
        hole_end_i = hole_start_i + self.obstacle_length_px
        idx_start = jnp.clip(hole_start_i, 0, self.hfield_length)
        idx_end = jnp.clip(hole_end_i, 0, self.hfield_length)
        hf = hf.at[idx_start:idx_end, self.start_j_px:self.end_j_px].set(self.hole_depth)
        
        return hf

    # --- NumPy Generation ---

    def _np_generate_terrain(self, scenario_idx: int, direction_idx: int) -> np.ndarray:
        """Generates, rotates, and places the chosen NumPy terrain scenario."""
        
        # 1. Generate the base scenario
        if scenario_idx == 0:
            scenario_field = self._np_scenario_1()
        elif scenario_idx == 1:
            scenario_field = self._np_scenario_2()
        else: # scenario_idx == 2
            scenario_field = self._np_scenario_3()
            
        # 2. Rotate the field
        if direction_idx == 0: # Front
            final_field = scenario_field
        elif direction_idx == 1: # Left
            final_field = np.rot90(scenario_field, k=1)
        else: # direction_idx == 2, Right
            final_field = np.rot90(scenario_field, k=-1)
            
        return final_field

    def _np_scenario_1(self) -> np.ndarray:
        """NumPy: Stairs (up) -> Slope (down)"""
        hf = np.zeros((self.hfield_length, self.hfield_length))
        
        # 1. Generate Stairs (Up)
        current_i = self.start_i_px
        for i in range(self.num_stairs):
            current_height = (i + 1) * self.stair_height
            next_i = current_i + self.stair_depth_px
            # Clip indices
            idx_start = np.clip(current_i, 0, self.hfield_length)
            idx_end = np.clip(next_i, 0, self.hfield_length)
            if idx_start >= idx_end: continue
            
            hf[idx_start:idx_end, self.start_j_px:self.end_j_px] = current_height
            current_i = next_i
        
        slope_start_i = current_i
        
        # 2. Generate Slope (Down)
        slope_end_i = slope_start_i + self.slope_length_px
        peak_height = self.num_stairs * self.stair_height
        
        x = np.linspace(peak_height, 0.0, self.slope_length_px)
        ramp = x[:, np.newaxis]
        
        idx_start = np.clip(slope_start_i, 0, self.hfield_length)
        idx_end = np.clip(slope_end_i, 0, self.hfield_length)
        if idx_start >= idx_end: return hf
        
        ramp = ramp[:(idx_end - idx_start), :]
        hf[idx_start:idx_end, self.start_j_px:self.end_j_px] = ramp
        
        return hf

    def _np_scenario_2(self) -> np.ndarray:
        """NumPy: Slope (up) -> Stairs (down)"""
        hf = np.zeros((self.hfield_length, self.hfield_length))
        
        # 1. Generate Slope (Up)
        slope_start_i = self.start_i_px
        slope_end_i = slope_start_i + self.slope_length_px
        peak_height = self.slope_height
        
        x = np.linspace(0.0, peak_height, self.slope_length_px)
        ramp = x[:, np.newaxis]
        
        idx_start = np.clip(slope_start_i, 0, self.hfield_length)
        idx_end = np.clip(slope_end_i, 0, self.hfield_length)
        if idx_start < idx_end:
            ramp = ramp[:(idx_end - idx_start), :]
            hf[idx_start:idx_end, self.start_j_px:self.end_j_px] = ramp
        
        # 2. Generate Stairs (Down)
        current_i = slope_end_i
        for i in range(self.num_stairs):
            current_height = peak_height - (i * self.stair_height)
            next_i = current_i + self.stair_depth_px
            
            idx_start = np.clip(current_i, 0, self.hfield_length)
            idx_end = np.clip(next_i, 0, self.hfield_length)
            if idx_start >= idx_end: continue
            
            hf[idx_start:idx_end, self.start_j_px:self.end_j_px] = current_height
            current_i = next_i
            
        return hf

    def _np_scenario_3(self) -> np.ndarray:
        """NumPy: Step (up) -> Flat -> Hole (step down)"""
        hf = np.zeros((self.hfield_length, self.hfield_length))

        # 1. Generate Step (Up)
        step_start_i = self.start_i_px
        step_end_i = step_start_i + self.obstacle_length_px
        idx_start = np.clip(step_start_i, 0, self.hfield_length)
        idx_end = np.clip(step_end_i, 0, self.hfield_length)
        if idx_start < idx_end:
            hf[idx_start:idx_end, self.start_j_px:self.end_j_px] = self.step_height
        
        # 2. Generate Flat Platform
        platform_start_i = step_end_i
        platform_end_i = platform_start_i + self.step_platform_length_px
        idx_start = np.clip(platform_start_i, 0, self.hfield_length)
        idx_end = np.clip(platform_end_i, 0, self.hfield_length)
        if idx_start < idx_end:
            hf[idx_start:idx_end, self.start_j_px:self.end_j_px] = self.step_height

        # 3. Generate Hole (Down)
        hole_start_i = platform_end_i
        hole_end_i = hole_start_i + self.obstacle_length_px
        idx_start = np.clip(hole_start_i, 0, self.hfield_length)
        idx_end = np.clip(hole_end_i, 0, self.hfield_length)
        if idx_start < idx_end:
            hf[idx_start:idx_end, self.start_j_px:self.end_j_px] = self.hole_depth
            
        return hf

    # --- Standard Methods (Copied from StonesHolesTerrain) ---

    def get_height_at_xy(
            self, 
            terrain_state: ParkourTerrainState, 
            xy_pos: Union[np.ndarray, jnp.ndarray], 
            backend: ModuleType
        ) -> Union[float, jax.Array]:
        """
        Get the terrain height (in meters) at a specific world (x, y) coordinate.
        """
        assert_backend_is_supported(backend)
        
        height_map = terrain_state.height_field_unscaled

        i = (xy_pos[0] + self.hfield_half_length_in_meters) * self.one_meter_length
        j = (xy_pos[1] + self.hfield_half_length_in_meters) * self.one_meter_length
        
        i_clipped = backend.clip(backend.astype(i, 'int32'), 0, self.hfield_length - 1)
        j_clipped = backend.clip(backend.astype(j, 'int32'), 0, self.hfield_length - 1)
        
        height = height_map[i_clipped, j_clipped]
        
        return height

    def update(self, env: Any,
               model: Union[MjModel, Model],
               data: Union[MjData, Data],
               carry: Any,
               backend: ModuleType) -> Tuple[Union[MjModel, Model], Union[MjData, Data], Any]:
        """Update the rough terrain and simulation state."""
        assert_backend_is_supported(backend)
        terrain_state = carry.terrain_state
        model = self._set_attribute_in_model(model, "hfield_data", terrain_state.height_field_raw, backend)
        data = self._reset_on_edge(data, backend)
        return model, data, carry

    def get_height_matrix(self, matrix_config: Dict[str, Any],
                          env: Any,
                          model: Union[MjModel, Model],
                          data: Union[MjData, Data],
                          carry: Any,
                          backend: ModuleType) -> Union[np.ndarray, jnp.ndarray]:
        assert_backend_is_supported(backend)
        raise NotImplementedError

    def isaac_hf_to_mujoco_hf(self,
                              isaac_hf: Union[np.ndarray, jnp.ndarray],
                              backend: ModuleType) -> Union[np.ndarray, jnp.ndarray]:
        """
        Convert Isaac height field data to MuJoCo-compatible height field data.
        """
        assert_backend_is_supported(backend)

        hf = isaac_hf + backend.abs(backend.min(isaac_hf))
        hf /= self.mujoco_height_scaling
        return hf.reshape(-1)

    def _reset_on_edge(self, data: Union[MjData, Data],
                       backend: ModuleType) -> Union[MjData, Data]:
        """Reset the robot position if it is on the edge of the terrain."""
        assert_backend_is_supported(backend)

        min_edge = self.hfield_half_length_in_meters - 0.5
        max_edge = self.hfield_half_length_in_meters
        com_pos = data.qpos[self._free_jnt_qpos_id][:2]
        reached_edge = backend.array(((min_edge < backend.abs(com_pos[0])) & (backend.abs(com_pos[0]) < max_edge)) | (
                    (min_edge < backend.abs(com_pos[1])) & (backend.abs(com_pos[1]) < max_edge)))
        free_jnt_xy = self._free_jnt_qpos_id[:2]
        if backend == jnp:
            init_data = data.replace(qpos=data.qpos.at[free_jnt_xy].set(0.0))
            data = jax.lax.cond(reached_edge, lambda _: init_data, lambda _: data, None)
        else:
            if reached_edge:
                data.qpos[free_jnt_xy] = 0.0

        return data