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
    height_field_raw: Union[np.ndarray, jax.Array]


class ParkourTerrain(DynamicTerrain):
    viewer_needs_to_update_hfield: bool = True 
    # NOTE: even if the following class is based on geoms and not on heightfield, we need to update a dummy hfield 
    # NOTE: for backtrack compatibility with MJX, since also here we have to update the view

    def __init__(
        self, env: Any,
        inner_platform_size_in_meters: float = 1,
        step_height: float = 0.1,
        step_distance: float = 1.0,
        step_length: float = 0.5,
        step_width: float = 1.0,
        feet_collision: List[str] = [],
        **kwargs: Any
    ):
        # super class initialization
        super().__init__(env, **kwargs)
        
        # store parameters
        self.inner_platform_size_in_meters = inner_platform_size_in_meters
        self.step_height = step_height
        self.step_distance = step_distance
        self.step_length = step_length
        self.step_width = step_width
        self.feet_collision = feet_collision
        
        # hfield parameters
        self.hfield_size = (4, 4, 0.01, 0.1)  # (sx, sy, sz, thickness)
        self.hfield_length = 80                 # resolution: 80 x 80
        self.hfield_half_length_in_meters = self.hfield_size[0]
        self.max_possible_height = self.hfield_size[2]
        
        # scaling used by isaac_hf_to_mujoco_hf
        self.one_meter_length = int(
            self.hfield_length / (self.hfield_half_length_in_meters * 2)
        )
        self.hfield_half_length = self.hfield_length // 2
        self.mujoco_height_scaling = self.max_possible_height

        # flat floor height (same as hfield geom pos.z)
        self._floor_height = 0.0
        self._hfield_height = 0.0

        # We still want to be able to reset on edge as in rough.py
        root_free_joint_xml_name = env.root_free_joint_xml_name
        self._free_jnt_qpos_id = np.array(
            mj_jntname2qposid(root_free_joint_xml_name, env._model)
        )

    def init_state(
        self,
        env: Any,
        key: Any,
        model: Union[MjModel, Model],
        data: Union[MjData, Data],
        backend: ModuleType,
    ) -> ParkourTerrainState:
        """
        Initialize the terrain state: start with a flat heightfield (all zeros).
        """
        assert_backend_is_supported(backend)

        return ParkourTerrainState(
            height_field_raw=backend.zeros((self.hfield_length, self.hfield_length))
        )

    def modify_spec(self, spec: MjSpec) -> MjSpec:
        """
        Modify the Mujoco spec:

        - add a heightfield 'parkour_terrain'
        - replace original floor with a hfield-based floor
        - add one box geom as the step in front of the robot
        """
        # heightfield
        file_name = (
            Path(__file__).resolve().parent.parent.parent
            / "models"
            / "common"
            / "default_hfield_80.png"
        )
        spec.add_hfield(
            name="parkour_terrain",
            size=self.hfield_size,
            file=str(file_name),
        )
        for i, field in enumerate(spec.hfields):
            if field.name == "parkour_terrain":
                self.hfield_id = i
                break

        # remove any existing floor geom
        for g in spec.geoms:
            if g.name == "floor":
                g.delete()
                break

        # retrieve the worldbody
        wb = spec.worldbody
        
        # add hfield
        wb.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname="parkour_terrain",
            group=2,
            pos=(0, 0, self._hfield_height),
            material="MatPlane",
            rgba=(0.8, 0.9, 0.8, 1),
            contype=0,
            conaffinity=0,
        )

        # add the step as a box geom
        # MuJoCo expects half sizes in geom.size
        hx = self.step_length / 2.0
        hy = self.step_width / 2.0
        hz = self.step_height / 2.0

        # place step: front face at x = step_distance
        # so center = step_distance + hx
        step_center_x = self.step_distance + hx
        step_center_y = 0.0
        step_center_z = self._floor_height + hz

        wb.add_geom(
            name="step",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(hx, hy, hz),
            pos=(step_center_x, step_center_y, step_center_z),
            material="MatPlane",
            group=2,
            rgba=(1.0, 0.2, 0.2, 1.0),
            contype=0,
            conaffinity=0,
        )
        
        # add collisions
        # spec.add_pair(geomname1="floor", geomname2="step")
        if len(self.feet_collision) > 0:
            for foot_geom in self.feet_collision:
                spec.add_pair(geomname1=foot_geom, geomname2="step")
                spec.add_pair(geomname1=foot_geom, geomname2="floor")

        return spec

    def reset(
        self,
        env: Any,
        model: Union[MjModel, Model],
        data: Union[MjData, Data],
        carry: Any,
        backend: ModuleType,
    ) -> Tuple[Union[MjData, Data], Any]:
        """
        Reset the terrain.

        For parkour: we use a flat heightfield (all zeros).
        The step is static (defined in modify_spec), so no geom changes here.
        """
        assert_backend_is_supported(backend)
        terrain_state = carry.terrain_state

        if backend == jnp:
            # flat heightfield = zeros
            height_matrix = jnp.zeros((self.hfield_length, self.hfield_length))
            height_field_raw = self.isaac_hf_to_mujoco_hf(height_matrix, backend)
        else:
            height_matrix = np.zeros((self.hfield_length, self.hfield_length))
            height_field_raw = self.isaac_hf_to_mujoco_hf(height_matrix, backend)

        terrain_state = terrain_state.replace(height_field_raw=height_field_raw)
        carry = carry.replace(terrain_state=terrain_state)

        return data, carry

    def update(
        self,
        env: Any,
        model: Union[MjModel, Model],
        data: Union[MjData, Data],
        carry: Any,
        backend: ModuleType,
    ) -> Tuple[Union[MjModel, Model], Union[MjData, Data], Any]:
        """
        Update the Mujoco MJX model with the current heightfield.

        The step geom is static (only in the spec), so we don't touch geoms here.
        """
        assert_backend_is_supported(backend)
        terrain_state = carry.terrain_state

        # push heightfield into model.hfield_data
        model = self._set_attribute_in_model(
            model, "hfield_data", terrain_state.height_field_raw, backend
        )
        data = self._reset_on_edge(data, backend)
        return model, data, carry

    def get_height_at_xy(
        self,
        terrain_state: ParkourTerrainState,
        xy_pos: Union[np.ndarray, jnp.ndarray],
        backend: ModuleType
    ) -> Union[float, jax.Array]:
        """
        Return the terrain height at world coordinates (x, y), considering:
        - the flat heightfield (uniform floor height)
        - the step box placed in front of the robot

        No slopes, no rotations — this is axis-aligned.
        """
        assert_backend_is_supported(backend)

        x = xy_pos[0]
        y = xy_pos[1]

        floor_h = self._floor_height          # e.g., -0.06
        step_h  = self.step_height            # height in meters
        step_len = self.step_length           # along x direction
        step_w   = self.step_width            # along y direction

        # Box half sizes
        hx = step_len / 2
        hy = step_w  / 2

        # Box center position
        cx = self.step_distance + hx      # front face at step_distance
        cy = 0.0

        # Compute footprint test
        inside_x = backend.abs(x - cx) <= hx
        inside_y = backend.abs(y - cy) <= hy
        inside_box = inside_x & inside_y

        # Height of the top of the box
        step_top_height = floor_h + step_h

        # Conditional return
        return backend.where(inside_box, step_top_height, floor_h)

    def get_height_matrix(
        self,
        matrix_config: Dict[str, Any],
        env: Any,
        model: Union[MjModel, Model],
        data: Union[MjData, Data],
        carry: Any,
        backend: ModuleType,
    ) -> Union[np.ndarray, jnp.ndarray]:
        """
        Not implemented: would return the height matrix around the robot if needed.
        """
        assert_backend_is_supported(backend)
        raise NotImplementedError

    def isaac_hf_to_mujoco_hf(
        self, isaac_hf: Union[np.ndarray, jnp.ndarray], backend: ModuleType
    ) -> Union[np.ndarray, jnp.ndarray]:
        """
        Convert height matrix into MuJoCo-compatible flattened hfield_data.
        Same as RoughTerrain, but here we typically pass in flat zeros.
        """
        assert_backend_is_supported(backend)
        hf = isaac_hf + backend.abs(backend.min(isaac_hf))
        hf /= self.mujoco_height_scaling
        return hf.reshape(-1)

    def _reset_on_edge(
        self, data: Union[MjData, Data], backend: ModuleType
    ) -> Union[MjData, Data]:
        """
        Reset the robot position if it is on the edge of the terrain.
        Same logic as in RoughTerrain.
        """
        assert_backend_is_supported(backend)

        min_edge = self.hfield_half_length_in_meters - 0.5
        max_edge = self.hfield_half_length_in_meters
        com_pos = data.qpos[self._free_jnt_qpos_id][:2]
        reached_edge = backend.array(
            ((min_edge < backend.abs(com_pos[0])) & (backend.abs(com_pos[0]) < max_edge))
            | ((min_edge < backend.abs(com_pos[1])) & (backend.abs(com_pos[1]) < max_edge))
        )
        free_jnt_xy = self._free_jnt_qpos_id[:2]
        if backend == jnp:
            init_data = data.replace(qpos=data.qpos.at[free_jnt_xy].set(0.0))
            data = jax.lax.cond(reached_edge, lambda _: init_data, lambda _: data, None)
        else:
            if reached_edge:
                data.qpos[free_jnt_xy] = 0.0

        return data