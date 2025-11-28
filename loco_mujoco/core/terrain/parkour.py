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
import copy

# ====================================================Parkour State====================================================
@struct.dataclass
class ParkourTerrainState:
    sizes: Union[np.ndarray, jax.Array]
    positions: Union[np.ndarray, jax.Array]
    quats: Union[np.ndarray, jax.Array]

# ====================================================Parkour Class====================================================
class ParkourTerrain(DynamicTerrain):
    viewer_needs_to_update_hfield: bool = False 
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
        inner_platform_size: float = 1.0,
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
        self.inner_platform_size = inner_platform_size

        # flat floor height 
        self._floor_height = 0.0
        
        # store the geoms for the obstacles
        self._obstacle_geom_ids = []  # NOTE: this will be added in modify_spec!

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
            positions=backend.zeros((self.num_boxes, 3)),
            sizes=backend.zeros((self.num_boxes, 3)),
            quats=backend.zeros((self.num_boxes, 4)),
        )

    def modify_spec(self, spec: MjSpec) -> MjSpec:
        # retrieve the worldbody
        wb = spec.worldbody
        
        # ==================================================OBSTACLEs==================================================
        # this part is only needed for initialization, then the boxes will be added when doing the reset
        for i in range(self.num_boxes):
            wb.add_geom(
                name=f"obstacle_{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(0.1, 0.1, 0.1),   # *      
                pos=(0., 0., 0.1),      # **             
                quat=(1, 0, 0, 0),     
                group=0,
                rgba=(0.8, 0.2, 0.2, 1.0),
                contype=0,
                conaffinity=0,
            )
        
        """
        Footnotes:
        *: if the specified size would have been (0,0,0), then mujoco would have complained. We set a "fake" size and 
        then we will update it at reset time.
        
        **: the initial position cannot be (0,0,0), otherwise we would have rendering issues. In particular, if the 
        initial position is (0,0,0), then mujoco would add to that geom an attirbute "SAME_FRAME=1". In that case,
        when we modify the model "geom_pos", "geom_size", and "geom_quat", they will not be updated in the data, since 
        the "forward" method called by the visualizer is filtering out certain geometries with certain values for
        the "SAME_FRAME". If the initial position is not exactly (0,0,0), then "SAME_FRAME=3", and we are happy.
        """
        
        # add ids
        self._obstacle_geom_ids = []
        for i, geom in enumerate(spec.geoms):
            if geom.name.startswith("obstacle_"):
                self._obstacle_geom_ids.append(i)
        
        # =================================================COLLISIONs=================================================
        # add collisions
        if len(self.feet_collision) > 0:
            for foot_geom in self.feet_collision:
                for i in range(self.num_boxes):
                    spec.add_pair(geomname1=foot_geom, geomname2=f"obstacle_{i}")
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
            pos_x = self.clip_to_platform(pos_x, backend)
            pos_y = jax.random.uniform(
                subkey2, (self.num_boxes,), minval=self.box_y_range[0], maxval=self.box_y_range[1] 
            )
            pos_y = self.clip_to_platform(pos_y, backend)
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
            pos_x = self.clip_to_platform(pos_x, backend)
            pos_y = np.random.uniform(
               self.box_y_range[0], self.box_y_range[1], size=(self.num_boxes,)
            )
            pos_y = self.clip_to_platform(pos_y, backend)
            pos_z = sizes_z / 2 # make them stay on the ground
            
            #... quats
            yaws = np.random.uniform(
                self.box_yaw_range[0], self.box_yaw_range[1], size=(self.num_boxes,)
            )
            quats = np.array([np_R.from_euler('z', yw).as_quat(scalar_first=True) for yw in yaws])

        # update the terrain state
        terrain_state = ParkourTerrainState(
            positions=backend.stack([pos_x, pos_y, pos_z], axis=1),
            sizes=backend.stack([sizes_x, sizes_y, sizes_z], axis=1),
            quats=quats,
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

        if backend == jnp:
            idx = jnp.array(self._obstacle_geom_ids, dtype=jnp.int32)
            geom_size = (model.geom_size).at[idx].set(terrain_state.sizes / 2.0)
            geom_pos = (model.geom_pos).at[idx].set(terrain_state.positions)
            geom_quat = (model.geom_quat).at[idx].set(terrain_state.quats)
            model = self._set_attribute_in_model(model, "geom_pos", geom_pos, backend)
            model = self._set_attribute_in_model(model, "geom_size", geom_size, backend)
            model = self._set_attribute_in_model(model, "geom_quat", geom_quat, backend)
        else:
            for local_id, geom_id in enumerate(self._obstacle_geom_ids):
                fullsize = terrain_state.sizes[local_id] / 2.0
                pos_xyz = terrain_state.positions[local_id]
                quat = terrain_state.quats[local_id]
                model.geom_size[geom_id] = np.asarray(fullsize)
                model.geom_pos[geom_id] = np.asarray(pos_xyz)
                model.geom_quat[geom_id] = np.asarray(quat)
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

    def clip_to_platform(self, v: Union[np.ndarray, jax.Array], backend) -> Union[np.ndarray, jax.Array]:
        """
        This func outputs a position vector v clipped to respect the flat area at the robot initialization specified 
        by the attribute "self.inner_platform_size."
        """
        return backend.sign(v) * backend.clip(backend.abs(v), a_min=self.inner_platform_size, a_max=None)
