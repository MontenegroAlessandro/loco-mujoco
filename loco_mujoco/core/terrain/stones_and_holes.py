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
class StonesHolesTerrainState:
    height_field_raw: Union[np.ndarray, jax.Array] # The raw, scaled hfield data for MuJoCo (80x80 flattened).
    height_field_unscaled: Union[np.ndarray, jax.Array] # The unscaled heightmap in meters (80x80).


class StonesHolesTerrain(DynamicTerrain):
    """
    Dynamic terrain with discrete patches ("stepping stones and holes") of varying heights.
    """

    viewer_needs_to_update_hfield: bool = True

    def __init__(
        self, env: Any,
        patch_size: float = 0.3, # Size of each patch in meters (the side of the squared zones with varying height)
        height_range: List[float] = [-0.1, 0.1], # Min/max height in meters
        inner_platform_size_in_meters: float = 1.0,
        **kwargs: Any
    ):
        """
        Initialize the stepping stones terrain.

        Args:
            env (Any): The environment instance.
            patch_size (float): The side length of each square patch.
            height_range (List[float]): [min, max] height of the patches.
            inner_platform_size_in_meters (float): Size of the flat starting platform.
            **kwargs (Any): Additional arguments for initialization.
        """
        super().__init__(env, **kwargs)

        self.height_range = height_range
        self.inner_platform_size_in_meters = inner_platform_size_in_meters
        
        # ... from RoughTerrain
        self.hfield_size = (4, 4, 30.0, 0.125) # (half_x, half_y, z_scale, sample_space)
        self.hfield_length = 80 # hfield resolution (pixels)
        self.hfield_half_length_in_meters = self.hfield_size[0]
        self.max_possible_height = self.hfield_size[2]
        self.one_meter_length = int(self.hfield_length / (self.hfield_half_length_in_meters * 2))
        self.hfield_half_length = self.hfield_length // 2
        self.mujoco_height_scaling = self.max_possible_height

        # Params for patches
        self.patch_size_pixels = int(patch_size * self.one_meter_length)
        if self.patch_size_pixels == 0:
            raise ValueError(f"patch_size {patch_size} is too small for hfield resolution.")
        
        # Number of patches in the grid
        self.num_patches_x = self.hfield_length // self.patch_size_pixels
        self.num_patches_y = self.hfield_length // self.patch_size_pixels

        # Platform cutout (... from RoughTerrain)
        platform_size = int(self.inner_platform_size_in_meters * self.one_meter_length)
        self.x1 = self.hfield_half_length - (platform_size // 2)
        self.y1 = self.hfield_half_length - (platform_size // 2)
        self.x2 = self.hfield_half_length + (platform_size // 2)
        self.y2 = self.hfield_half_length + (platform_size // 2)
        
        root_free_joint_xml_name = env.root_free_joint_xml_name
        self._free_jnt_qpos_id = np.array(mj_jntname2qposid(root_free_joint_xml_name, env._model))

    def init_state(
        self, env: Any,
        key: Any,
        model: Union[MjModel, Model],
        data: Union[MjData, Data],
        backend: ModuleType
    ) -> StonesHolesTerrainState:
        """Initialize the state of the stepping stones terrain."""
        assert_backend_is_supported(backend)
        return StonesHolesTerrainState(
            height_field_raw=backend.zeros(self.hfield_length * self.hfield_length),
            height_field_unscaled=backend.zeros((self.hfield_length, self.hfield_length))
        )

    def modify_spec(self, spec: MjSpec) -> MjSpec:
        """Modify the simulation specification (Identical to RoughTerrain)."""
        file_name = Path(__file__).resolve().parent.parent.parent / "models" / "common" / "default_hfield_80.png"
        spec.add_hfield(name='stepping_stones_terrain', size=self.hfield_size, file=str(file_name))
        for i, field in enumerate(spec.hfields):
            if field.name == 'stepping_stones_terrain':
                self.hfield_id = i
                break

        for g in spec.geoms:
            if g.name == 'floor':
                g.delete()
                break

        wb = spec.worldbody
        wb.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname='stepping_stones_terrain', group=2,
                    pos=(0, 0, -0.06), material="MatPlane", rgba=(0.8, 0.9, 0.8, 1))
        return spec

    def reset(
            self, env: Any,
            model: Union[MjModel, Model], data: Union[MjData, Data], carry: Any,
            backend: ModuleType
        ) -> Tuple[Union[MjData, Data], Any]:
        """Reset the terrain by generating a new patch layout."""
        assert_backend_is_supported(backend)
        
        # Generate the unscaled heightmap
        if backend == jnp:
            key = carry.key
            key, subkey = jax.random.split(key)
            height_field_unscaled = self._jnp_generate_patches(subkey)
            carry = carry.replace(key=key)
        else:
            height_field_unscaled = self._np_generate_patches()

        # Convert to MuJoCo-scaled format
        height_field_raw = self.isaac_hf_to_mujoco_hf(height_field_unscaled, backend)
        
        # Store in state
        terrain_state = StonesHolesTerrainState(
            height_field_raw=height_field_raw,
            height_field_unscaled=height_field_unscaled
        )
        carry = carry.replace(terrain_state=terrain_state)

        return data, carry

    def _jnp_generate_patches(self, key: Any) -> jnp.ndarray:
        """Generate random patches using JAX."""
        # Create low-resolution grid of random heights
        low_res_grid = jax.random.uniform(
            key,
            shape=(self.num_patches_x, self.num_patches_y),
            minval=self.height_range[0],
            maxval=self.height_range[1]
        )
        
        # Upsample to full hfield resolution using jnp.repeat
        height_field = jnp.repeat(low_res_grid, self.patch_size_pixels, axis=0)
        height_field = jnp.repeat(height_field, self.patch_size_pixels, axis=1)

        # Cut out the flat starting platform
        height_field = height_field.at[self.x1:self.x2, self.y1:self.y2].set(0.0)
        return height_field

    def _np_generate_patches(self) -> np.ndarray:
        """Generate random patches using NumPy."""
        # Create low-resolution grid of random heights
        low_res_grid = np.random.uniform(
            low=self.height_range[0],
            high=self.height_range[1],
            size=(self.num_patches_x, self.num_patches_y)
        )
        
        # Upsample to full hfield resolution using np.repeat
        height_field = np.repeat(low_res_grid, self.patch_size_pixels, axis=0)
        height_field = np.repeat(height_field, self.patch_size_pixels, axis=1)

        # Cut out the flat starting platform
        height_field[self.x1:self.x2, self.y1:self.y2] = 0.0
        return height_field

    def get_height_at_xy(
            self, 
            terrain_state: StonesHolesTerrainState, 
            xy_pos: Union[np.ndarray, jnp.ndarray], 
            backend: ModuleType
        ) -> Union[float, jax.Array]:
        """
        Get the terrain height (in meters) at a specific world (x, y) coordinate.
        """
        assert_backend_is_supported(backend)
        
        # Get the unscaled heightmap
        height_map = terrain_state.height_field_unscaled

        # Convert world coordinates (x, y) to hfield indices (i, j)
        # World coords range from -4.0 to +4.0
        # Hfield indices range from 0 to 79
        
        # Add half-length (meters) to shift origin from (0,0) to (-4, -4)
        # Then multiply by pixels-per-meter
        i = (xy_pos[0] + self.hfield_half_length_in_meters) * self.one_meter_length
        j = (xy_pos[1] + self.hfield_half_length_in_meters) * self.one_meter_length
        
        # Clip indices to be within the hfield bounds [0, 79]
        i_clipped = backend.clip(backend.astype(i, 'int32'), 0, self.hfield_length - 1)
        j_clipped = backend.clip(backend.astype(j, 'int32'), 0, self.hfield_length - 1)
        
        # Index the heightmap to get the height
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