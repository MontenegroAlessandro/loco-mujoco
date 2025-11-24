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
    sizes: Union[np.ndarray, jax.Array]
    positions: Union[np.ndarray, jax.Array]
    quats: Union[np.ndarray, jax.Array]


class ParkourTerrain(DynamicTerrain):
    viewer_needs_to_update_hfield: bool = True 
    # NOTE: even if the following class is based on geoms and not on heightfield, we need to update a dummy hfield 
    # NOTE: for backtrack compatibility with MJX, since also here we have to update the view

    def __init__(
        self, env: Any,
        num_boxes: int = 5,
        box_length_range: List[float] = [0.2, 1.0],
        box_width_range: List[float] = [0.1, 0.8],
        box_height_range: List[float] = [0.05, 0.5],
        box_x_range: List[float] = [1.0, 5.0],
        box_y_range: List[float] = [-1.0, 1.0],
        box_yaw_range: List[float] = [-np.pi, np.pi],
        feet_collision: List[str] = [],
        **kwargs: Any
    ):
        # super class initialization
        super().__init__(env, **kwargs)
        
        # store parameters
        self.num_boxes = num_boxes
        self.box_length_range = box_length_range
        self.box_width_range = box_width_range
        self.box_height_range = box_height_range
        self.box_x_range = box_x_range
        self.box_y_range = box_y_range
        self.box_yaw_range = box_yaw_range
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
        
        # store the geoms for the obstacles
        self._obstacle_geom_ids = []  # NOTE: these will be added in modify_spec!

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
            height_field_raw=backend.zeros((self.hfield_length, self.hfield_length)),
            positions=backend.zeros((self.num_boxes, 3)),
            sizes=backend.zeros((self.num_boxes, 3)),
            quats=backend.zeros((self.num_boxes, 4))
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
        
        # add plaemhoder geoms for the boxes
        # this part is only needed for initialization, then the boxes will be added when doing the reset
        for i in range(self.num_boxes):
            wb.add_geom(
                name=f"obstacle_{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(0.1, 0.1, 0.1),      # placeholder; will be overwritten at reset()
                pos=(0, 0, 0),             # placeholder
                quat=(1, 0, 0, 0),         # no rotation
                group=2,
                rgba=(0.8, 0.2, 0.2, 1.0),
                contype=0,
                conaffinity=0,
            )
        
        # add ids
        self._obstacle_geom_ids = []
        for i, geom in enumerate(spec.geoms):
            if geom.name.startswith("obstacle_"):
                self._obstacle_geom_ids.append(i)
        
        # add collisions
        if len(self.feet_collision) > 0:
            for foot_geom in self.feet_collision:
                spec.add_pair(geomname1=foot_geom, geomname2="floor")
                for i in range(self.num_boxes):
                    spec.add_pair(geomname1=foot_geom, geomname2=f"obstacle_{i}")

        # NOTE: remove
        # # add the step as a box geom
        # # MuJoCo expects half sizes in geom.size
        # hx = self.step_length / 2.0
        # hy = self.step_width / 2.0
        # hz = self.step_height / 2.0

        # # place step: front face at x = step_distance
        # # so center = step_distance + hx
        # step_center_x = self.step_distance + hx
        # step_center_y = 0.0
        # step_center_z = self._floor_height + hz

        # wb.add_geom(
        #     name="step",
        #     type=mujoco.mjtGeom.mjGEOM_BOX,
        #     size=(hx, hy, hz),
        #     pos=(step_center_x, step_center_y, step_center_z),
        #     material="MatPlane",
        #     group=2,
        #     rgba=(1.0, 0.2, 0.2, 1.0),
        #     contype=0,
        #     conaffinity=0,
        # )
        
        # # add collisions
        # # spec.add_pair(geomname1="floor", geomname2="step")
        # if len(self.feet_collision) > 0:
        #     for foot_geom in self.feet_collision:
        #         spec.add_pair(geomname1=foot_geom, geomname2="step")
        #         spec.add_pair(geomname1=foot_geom, geomname2="floor")
        # NOTE: end remove

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
        
        # sample boxes dimensions
        if backend == jnp:
            # extract keys
            key = carry.key
            key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
            
            # sample dimensions
            sizes_x = jax.random.uniform(
                subkey1, (self.num_boxes,), minval=self.box_length_range[0], maxval=self.box_length_range[1]
            )
            sizes_y = jax.random.uniform(
                subkey2, (self.num_boxes,), minval=self.box_width_range[0], maxval=self.box_width_range[1]
            )
            sizes_z = jax.random.uniform(
                subkey3, (self.num_boxes,), minval=self.box_height_range[0], maxval=self.box_height_range[1]
            )
            
            # sample positions
            key, subkey1, subkey2 = jax.random.split(key, 3)
            pos_x = jax.random.uniform(
                subkey1, (self.num_boxes,), minval=self.box_x_range[0], maxval=self.box_x_range[1] 
            )
            pos_y = jax.random.uniform(
                subkey2, (self.num_boxes,), minval=self.box_y_range[0], maxval=self.box_y_range[1] 
            )
            pos_z = sizes_z / 2 # make them stay on the ground
            
            # sample yaws
            key, subkey = jax.random.split(key, 2)
            yaws = jax.random.uniform(
                subkey, (self.num_boxes,), minval=self.box_yaw_range[0], maxval=self.box_yaw_range[1]
            )
            # quats = jnp.array([jnp_R.from_euler('z', yw).as_quat(scalar_first=True) for yw in yaws])
            half_yaw = yaws / 2.0
            qw = jnp.cos(half_yaw)
            qz = jnp.sin(half_yaw)
            q0 = jnp.zeros_like(qw)
            quats = jnp.stack([qw, q0, q0, qz], axis=1)  # (N, 4) scalar-first
            
            # flat heightfield = zeros
            height_matrix = jnp.zeros((self.hfield_length, self.hfield_length))
            height_field_raw = self.isaac_hf_to_mujoco_hf(height_matrix, backend)
            
            # set the new key
            carry = carry.replace(key=key)
        else:
            # do the same as before but in np
            # ... sizes
            sizes_x = np.random.uniform(
                self.box_length_range[0], self.box_length_range[1], size=(self.num_boxes,)
            )
            sizes_y = np.random.uniform(
                self.box_width_range[0], self.box_width_range[1], size=(self.num_boxes,)
            )
            sizes_z = np.random.uniform(
                self.box_height_range[0], self.box_height_range[1], size=(self.num_boxes,)
            )
            
            # ... positions
            pos_x = np.random.uniform(
               self.box_x_range[0], self.box_x_range[1], size=(self.num_boxes,)
            )
            pos_y = np.random.uniform(
               self.box_y_range[0], self.box_y_range[1], size=(self.num_boxes,)
            )
            pos_z = sizes_z / 2 # make them stay on the ground
            
            #... quats
            yaws = np.random.uniform(
                self.box_yaw_range[0], self.box_yaw_range[1], size=(self.num_boxes,)
            )
            quats = np.array([np_R.from_euler('z', yw).as_quat(scalar_first=True) for yw in yaws])
            
            # flat height field
            height_matrix = np.zeros((self.hfield_length, self.hfield_length))
            height_field_raw = self.isaac_hf_to_mujoco_hf(height_matrix, backend)

        # update the terrain state
        terrain_state = terrain_state.replace(
            height_field_raw=height_field_raw,
            positions=backend.stack([pos_x, pos_y, pos_z], axis=1),
            sizes=backend.stack([sizes_x, sizes_y, sizes_z], axis=1),
            quats=quats
        )
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
        Update the Mujoco MJX model with the current heightfield and obstacle geoms.
        """
        assert_backend_is_supported(backend)
        terrain_state = carry.terrain_state

        # ----- update obstacle geoms -----
        if backend == jnp:
            # JAX: use .at[...] to keep things JIT-friendly
            geom_size = model.geom_size
            geom_pos = model.geom_pos
            geom_quat = model.geom_quat

            for local_id, geom_id in enumerate(self._obstacle_geom_ids):
                # MuJoCo stores half-sizes in geom_size
                fullsize = terrain_state.sizes[local_id] / 2.0
                pos_xyz = terrain_state.positions[local_id]
                quat = terrain_state.quats[local_id]

                geom_size = geom_size.at[geom_id].set(fullsize)
                geom_pos = geom_pos.at[geom_id].set(pos_xyz)
                geom_quat = geom_quat.at[geom_id].set(quat)

            model = model.replace(
                geom_pos=geom_pos, 
                geom_size=geom_size, 
                geom_quat=geom_quat, 
                hfield_data=terrain_state.height_field_raw
            )
        else:
            # NumPy backend: just assign in-place
            for local_id, geom_id in enumerate(self._obstacle_geom_ids):
                fullsize = terrain_state.sizes[local_id] / 2.0
                pos_xyz = terrain_state.positions[local_id]
                quat = terrain_state.quats[local_id]

                model.geom_size[geom_id] = np.asarray(fullsize)
                model.geom_pos[geom_id] = np.asarray(pos_xyz)
                model.geom_quat[geom_id] = np.asarray(quat)
            model.hfield_data = np.asarray(terrain_state.height_field_raw)

        data = self._reset_on_edge(data, backend)
        return model, data, carry

    def get_height_at_xy(
        self,
        terrain_state: ParkourTerrainState,
        xy_pos: Union[np.ndarray, jnp.ndarray],
        backend: ModuleType,
    ):
        assert_backend_is_supported(backend)

        x, y = xy_pos[0], xy_pos[1]
        floor_h = self._floor_height

        centers = terrain_state.positions      # (N, 3)
        sizes   = terrain_state.sizes          # (N, 3)
        quats   = terrain_state.quats          # (N, 4), scalar-first

        if centers.shape[0] == 0:
            # no obstacles
            return floor_h

        # extract yaw from quats (assuming z-rotation only)
        w = quats[:, 0]
        z = quats[:, 3]
        yaw = 2.0 * backend.atan2(z, w)        # (N,)

        # rotate world point into each box local frame (inverse yaw)
        cos_y = backend.cos(-yaw)
        sin_y = backend.sin(-yaw)

        dx = x - centers[:, 0]
        dy = y - centers[:, 1]

        lx = dx * cos_y - dy * sin_y
        ly = dx * sin_y + dy * cos_y

        hx = sizes[:, 0] / 2.0
        hy = sizes[:, 1] / 2.0
        hz = sizes[:, 2] / 2.0

        inside_x = backend.abs(lx) <= hx
        inside_y = backend.abs(ly) <= hy
        inside   = inside_x & inside_y

        top = centers[:, 2] + hz

        # mask boxes that don't cover (x,y)
        masked_tops = backend.where(inside, top, -1e9)
        max_top = backend.max(masked_tops)

        return backend.maximum(floor_h, max_top)

    def old_get_height_at_xy(
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