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

# ===================================================AdaPillar State===================================================
@struct.dataclass
class AdaPillarsTerrainState:
    sizes: Union[np.ndarray, jax.Array]
    positions: Union[np.ndarray, jax.Array]
    quats: Union[np.ndarray, jax.Array]
    desired_z: float 

# ===================================================AdaPillar Class===================================================
class AdaPillarsTerrain(DynamicTerrain):
    """
    This tarrain is thought to be used with Foot Placement Goals. In particular, whenever the generated goal has
    a z-traget which is > 0, then pillars with that height are spawned under the proposed foot placement height.
    """
    
    viewer_needs_to_update_hfield: bool = False 
    
    def __init__(
        self, env: Any,
        num_pillars: int = 2, # thought for humanoids, but can be used with quadrupeds too, just put 4
        diameter: float = 0.3, # the foot should be able to properly stand on the pillar
        feet_collision: List[str] = [],
        foot_dimension: List[float] = [0.1, 0.04, 0.01],
        remove_floor: bool = False,
        pillar_height: float = 0.2,
        **kwargs: Any
    ):
        # super class initialization
        super().__init__(env, **kwargs)

        # store parameters
        self.num_pillars = num_pillars + 1
        """NOTE: we need one more pillar in order not to leave any foot without floor."""
        self.diameter = diameter
        self.feet_collision = feet_collision
        self.foot_dimension = foot_dimension
        self.remove_floor = remove_floor
        self.pillar_height = pillar_height

        # flat floor height used as the pillar hide-position in reset() and as the fallback
        # in get_height_at_xy. For adaptive terrain the height terminal handler uses desired_z
        # instead, so this value does not affect the termination condition.
        self._floor_height = 0.0
        
        # store the geoms for the obstacles
        self._obstacle_geom_ids = []
        """NOTE: this will be added in modify_spec!"""

    def init_state(
        self,
        env: Any,
        key: Any,
        model: Union[MjModel, Model],
        data: Union[MjData, Data],
        backend: ModuleType,
    ) -> AdaPillarsTerrainState:
        """
        Initialize the terrain state: start with a flat heightfield (all zeros).
        """
        assert_backend_is_supported(backend)

        return AdaPillarsTerrainState(
            positions=backend.zeros((self.num_pillars, 3)),
            sizes=backend.zeros((self.num_pillars, 3)),
            quats=backend.zeros((self.num_pillars, 4)),
            desired_z=0.0
        )

    def modify_spec(self, spec: MjSpec) -> MjSpec:
        # retrieve the worldbody
        wb = spec.worldbody

        # remove the floor if requested
        if self.remove_floor:
            for g in spec.geoms:
                if g.name == 'floor':
                    g.delete()
                    break
            # remove any collision pairs that referenced the floor geom
            for pair in list(spec.pairs):
                if pair.geomname1 == 'floor' or pair.geomname2 == 'floor':
                    pair.delete()
        
        # ==================================================OBSTACLEs==================================================
        # this part is only needed for initialization, then the boxes will be added when doing the reset
        for i in range(self.num_pillars):
            wb.add_geom(
                name=f"pillar_{i}",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=(0.3, 0.1, 0.0),   # *      
                pos=(-5.0, 0.0, -0.5),    # **             
                quat=(0.0, 0.0, 0.0, 1.0),     
                group=0,
                rgba=(0.26, 0.4, 0.82, 1.0),
                contype=0,
                conaffinity=0,
                priority=1,
                friction=(1.1,0.5,0.001)
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
            if geom.name.startswith("pillar_"):
                self._obstacle_geom_ids.append(i)
        
        # =================================================COLLISIONs=================================================
        # add collisions
        if len(self.feet_collision) > 0:
            for foot_geom in self.feet_collision:
                for i in range(self.num_pillars):
                    spec.add_pair(geomname1=foot_geom, geomname2=f"pillar_{i}")
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
        
        """
        NOTE: reset has no power here, this terrain is thought for being queried by a foot placement goal generator, 
        the pillars are thought to be generated when a target z is sampled.
        """

        # define the sizes (which has not to be modified)
        sizes_x = backend.ones(self.num_pillars, dtype=backend.float32) * (self.diameter / 2.0)
        sizes_y = backend.zeros(self.num_pillars, dtype=backend.float32) # half of the height
        sizes_z = backend.ones(self.num_pillars, dtype=backend.float32) * self._floor_height # will be ingored for cylilnders
        
        # define the positions (at reset we hide the pillars for safety reasons)
        pos_x = backend.zeros(self.num_pillars, dtype=backend.float32)
        pos_y = backend.zeros(self.num_pillars, dtype=backend.float32)
        pos_z = backend.ones(self.num_pillars, dtype=backend.float32) * self._floor_height

        quat = backend.array([1.0, 0.0, 0.0, 0.0])
        quats = backend.tile(quat, (self.num_pillars, 1))

        # update the terrain state
        terrain_state = AdaPillarsTerrainState(
            positions=backend.stack([pos_x, pos_y, pos_z], axis=1),
            sizes=backend.stack([sizes_x, sizes_y, sizes_z], axis=1),
            quats=quats,
            desired_z=0.0
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
            geom_size = (model.geom_size).at[idx].set(terrain_state.sizes)
            geom_pos = (model.geom_pos).at[idx].set(terrain_state.positions)
            model = self._set_attribute_in_model(model, "geom_pos", geom_pos, backend)
            model = self._set_attribute_in_model(model, "geom_size", geom_size, backend)
        else:
            idx = np.array(self._obstacle_geom_ids, dtype=np.int32)
            model.geom_size[idx] = terrain_state.sizes
            model.geom_pos[idx] = terrain_state.positions
            model = self._set_attribute_in_model(model, "geom_pos", geom_pos, backend)
            model = self._set_attribute_in_model(model, "geom_size", geom_size, backend)
            
        return model, data, carry

    def set_height_at_xy(
        self,
        terrain_state: AdaPillarsTerrainState,
        xy_pos: Union[np.ndarray, jnp.ndarray],
        desired_z: float,
        pillar_id: int, 
        backend: ModuleType,
    ) -> AdaPillarsTerrainState:
        # check if the tarrain is supported
        assert_backend_is_supported(backend)
        
        # retrieve state information
        sizes = terrain_state.sizes
        poss = terrain_state.positions
        
        raw_des_z = desired_z

        if self.remove_floor:
            # Fixed-shape pillar: always pillar_height tall, top aligned with the foot
            # contact point (= foot-site z minus foot half-thickness).  The pillar hangs
            # downward from there, so it works for any target_z — including negative values.
            half_height = self.pillar_height / 2.0
            foot_contact_z = desired_z - self.foot_dimension[2]
            pos_z = foot_contact_z - half_height
        else:
            # Floor-based: pillar grows from z=0 up to the target.
            # pos_z == half_height encodes a cylinder whose bottom is at z=0 and
            # top is at 2*pos_z ≈ target_z − foot_thickness.
            pos_z = backend.maximum((desired_z / 2.0), 0.0) - (self.foot_dimension[2] / 2.0)
            # MuJoCo requires non-negative half-height; clamp so degenerate pillars
            # near z=0 stay valid.
            pos_z = backend.maximum(pos_z, 0.0)
            half_height = pos_z          # same value: bottom at 0, top at 2*pos_z

        # set at x and y coordinates the new cylinder height to be the desired_z for the desired foot
        new_pos = backend.concatenate([xy_pos, backend.array([pos_z])], dtype=backend.float32)
        if backend == jnp:
            # modify the size (diameter not touched)
            sizes = sizes.at[pillar_id, 1].set(half_height)

            # modify the positions
            poss = poss.at[pillar_id].set(new_pos)
        else:
            # modify the size (diameter not touched)
            sizes[pillar_id, 1] = half_height

            # modify the positions
            poss[pillar_id] = new_pos
        
        # replace the attributes in the state
        terrain_state = terrain_state.replace(positions=poss, sizes=sizes, desired_z=raw_des_z)
        
        return terrain_state
    
    def get_height_at_xy(
        self,
        terrain_state: AdaPillarsTerrainState,
        xy_pos: Union[np.ndarray, jnp.ndarray],
        backend: ModuleType,
    ) -> float:
        assert_backend_is_supported(backend)
        
        # Extract pillar data
        pillar_pos = terrain_state.positions
        pillar_sizes = terrain_state.sizes
        
        d_sq = backend.sum((pillar_pos[:, :2] - xy_pos)**2, axis=1) # squared distances
        r_sq = pillar_sizes[:, 0]**2 # squared radius
        
        # Check if point is inside the pillar (dist^2 <= radius^2)
        is_on_pillar = d_sq <= r_sq
        
        # Retrieve heights
        pillar_top_heights = pillar_pos[:, 2] + pillar_sizes[:, 1]
        
        # If we are on multiple pillars (overlap), we typically take the max height
        heights_under_point = backend.where(is_on_pillar, pillar_top_heights, self._floor_height)
        max_height = backend.max(heights_under_point)
        
        return max_height
