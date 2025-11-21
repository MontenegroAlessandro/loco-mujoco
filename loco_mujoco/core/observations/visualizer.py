from typing import Any, Union, List, Tuple
from types import ModuleType
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R
import mujoco
from mujoco import MjModel, MjData
from mujoco.mjx import Model, Data

from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast


def extract_z_rotation(global_rotation, backend):
    if backend == np:
        R = np_R
    else:
        R = jnp_R

    # Convert to Euler angles
    euler_angles = global_rotation.as_euler('xyz', degrees=False)

    # Isolate yaw (z-axis rotation)
    yaw_only = [0, 0, euler_angles[2]]  # Set roll and pitch to zero, keep yaw

    # Convert back to rotation object
    z_rotation = R.from_euler('xyz', yaw_only, degrees=False)

    return z_rotation


class RootVelocityArrowVisualizer:
    """
    A class to visualize the root velocity and rotational velocity arrows in
    a carry state to be visualized in simulation.

    Attributes:
        _arrow_n_visual_geoms (int): Number of visual geometries for arrows.
        _z_offset (np.ndarray): Z-axis offset for visualization.
        _arrow_color (np.ndarray): Color of the linear velocity arrow.
        _arrow_type (int): Geometry type for the arrow.
        _rot_vel_arrow_type (int): Geometry type for the rotational velocity arrow.
        _rot_vel_arrow_size (np.ndarray): Size of the rotational velocity arrow.
        _rot_vel_arrow_color (np.ndarray): Color of the rotational velocity arrow.
        _rot_vel_point_size (np.ndarray): Size of the rotational velocity point.
        _center_sphere_size (np.ndarray): Size of the center sphere.
        _center_sphere_type (int): Geometry type for the center sphere.
        _center_sphere_mat (np.ndarray): Material of the center sphere.
        _center_sphere_color (np.ndarray): Color of the center sphere.
    """

    def __init__(self, info_props, visualize_rot_vel: bool = False):
        self._arrow_n_visual_geoms = 3 if visualize_rot_vel else 2
        self._visualize_rot_vel = visualize_rot_vel
        self._z_offset = np.array([0.0, 0.0, 0.3]) + info_props["goal_visualization_arrow_offset"]
        self._arrow_color = np.array([1.0, 0.0, 0.0, 0.75])
        self._arrow_type = int(mujoco.mjtGeom.mjGEOM_ARROW)
        self._rot_vel_arrow_type = int(mujoco.mjtGeom.mjGEOM_ARROW1)
        self._rot_vel_arrow_size = np.array([0.015, 0.015, 0.5])
        self._rot_vel_arrow_color = np.array([1.0, 1.0, 0.0, 0.75])
        self._rot_vel_point_size = np.array([0.025, 0.025, 0.025])
        self._center_sphere_size = np.array([0.05, 0.0, 0.0])
        self._center_sphere_type = int(mujoco.mjtGeom.mjGEOM_SPHERE)
        self._center_sphere_mat = np.eye(3).reshape(-1)
        self._center_sphere_color = np.array([0.063, 0.0, 0.541, 1.0])

    def set_visuals(self,
                    goal: Union[np.ndarray, jnp.ndarray],
                    env: Any,
                    model: Union[MjModel, Model],
                    data: Union[MjData, Data],
                    carry: Any,
                    root_body_id: int,
                    free_jnt_qposid: Union[np.ndarray, jnp.ndarray],
                    visual_geoms_idx: List[int],
                    backend: ModuleType) -> Any:
        """
        Sets the visuals for root velocity and rotational velocity in the carry.

        Args:
            goal (Union[np.ndarray, jnp.ndarray]): Target velocity in global frame.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[Model, Data]): The simulation data.
            carry (Any): Additional state information for rendering.
            root_body_id (int): ID of the root body in the simulation.
            free_jnt_qposid (Union[np.ndarray, jnp.ndarray],): Index for the root joint position in `qpos`.
            visual_geoms_idx (List[int]): Indices of visual geometries.
            backend (ModuleType): Backend module (e.g., `np` or `jnp`) for calculations.

        Returns:
            Any: Updated carry with the new visualizations set.
        """
        if backend == np:
            R = np_R
        else:
            R = jnp_R

        user_scene = carry.user_scene
        geoms = user_scene.geoms

        # Get root orientation
        root_qpos = backend.squeeze(data.qpos[free_jnt_qposid])
        root_quat = R.from_quat(quat_scalarfirst2scalarlast(root_qpos[3:7]))
        root_quat = extract_z_rotation(root_quat, backend)
        root_mat = root_quat.as_matrix()

        # Get root position
        root_pos = data.xpos[root_body_id]

        # Goal velocity in local frame
        goal_lin_vel = backend.concatenate([goal[:3]])
        goal_vel_local = root_mat @ goal_lin_vel
        goal_rot_vel = goal[5]

        # Base rotation arrow (add 90-degree rotation around y-axis)
        arrow_mat = root_mat @ R.from_euler("y", 90, degrees=True).as_matrix()

        # Calculate angle between two arrays
        v1 = backend.array([0.0, 0.0, 1.0])
        reorder = np.array([2, 1, 0])
        goal_lin_vel_norm = goal_lin_vel[reorder] / backend.linalg.norm(goal_lin_vel[reorder])
        cross = backend.cross(v1, goal_lin_vel_norm)
        dot = backend.dot(v1, goal_lin_vel_norm)
        angle = backend.arctan2(cross[0], dot)  # Use x-component for sign
        arrow_mat = arrow_mat @ R.from_euler("x", angle, degrees=False).as_matrix()
        arrow_mat = arrow_mat.reshape(-1)

        min_arrow_length = 0.1
        arrow_length = backend.linalg.norm(goal_lin_vel) + min_arrow_length

        # Calculate position of the point indicating rotational velocity
        scaling = 0.15
        local_x_y_rot_vel = backend.array([
            backend.cos(goal_rot_vel * (backend.pi / 2)) * scaling,
            backend.sin(goal_rot_vel * (backend.pi / 2)) * scaling,
            self._z_offset[2]
        ])

        # Rotational velocity arrow matrix
        if self._visualize_rot_vel:
            rot_vel_arrow_mat = root_mat @ R.from_euler("y", 90, degrees=True).as_matrix()
            local_x_y_rot_vel_norm = local_x_y_rot_vel[reorder] / backend.linalg.norm(local_x_y_rot_vel[reorder])
            cross = backend.cross(v1, local_x_y_rot_vel_norm)
            dot = backend.dot(v1, local_x_y_rot_vel_norm)
            angle = backend.arctan2(cross[0], dot)  # Use x-component for sign
            rot_vel_arrow_mat = rot_vel_arrow_mat @ R.from_euler("x", angle, degrees=False).as_matrix()
            rot_vel_arrow_mat = rot_vel_arrow_mat.reshape(-1)

        # Arrow size
        arrow_size = backend.array([0.025, 0.025, arrow_length])

        if backend == jnp:
            # Set visualizations using JAX's array operations
            arrow_idx = visual_geoms_idx[0]
            geom_pos = user_scene.geoms.pos.at[arrow_idx].set(root_pos + self._z_offset)
            geom_mat = user_scene.geoms.mat.at[arrow_idx].set(arrow_mat)
            geom_type = user_scene.geoms.type.at[arrow_idx].set(self._arrow_type)
            geom_size = user_scene.geoms.size.at[arrow_idx].set(arrow_size)
            geom_rgba = user_scene.geoms.rgba.at[arrow_idx].set(self._arrow_color)

            center_sphere_idx = visual_geoms_idx[1]
            geom_pos = geom_pos.at[center_sphere_idx].set(root_pos + self._z_offset)
            geom_mat = geom_mat.at[center_sphere_idx].set(self._center_sphere_mat)
            geom_type = geom_type.at[center_sphere_idx].set(self._center_sphere_type)
            geom_size = geom_size.at[center_sphere_idx].set(self._center_sphere_size)
            geom_rgba = geom_rgba.at[center_sphere_idx].set(self._center_sphere_color)

            if self._visualize_rot_vel:
                rot_vel_arrow_idx = visual_geoms_idx[2]
                geom_pos = geom_pos.at[rot_vel_arrow_idx].set(root_pos + self._z_offset)
                geom_mat = geom_mat.at[rot_vel_arrow_idx].set(rot_vel_arrow_mat)
                geom_type = geom_type.at[rot_vel_arrow_idx].set(self._rot_vel_arrow_type)
                geom_size = geom_size.at[rot_vel_arrow_idx].set(self._rot_vel_arrow_size)
                geom_rgba = geom_rgba.at[rot_vel_arrow_idx].set(self._rot_vel_arrow_color)

        else:
            # Set visualizations using NumPy's array operations
            arrow_idx = visual_geoms_idx[0]
            user_scene.geoms.pos[arrow_idx] = root_pos + self._z_offset
            user_scene.geoms.mat[arrow_idx] = arrow_mat
            user_scene.geoms.type[arrow_idx] = self._arrow_type
            user_scene.geoms.size[arrow_idx] = arrow_size
            user_scene.geoms.rgba[arrow_idx] = self._arrow_color

            center_sphere_idx = visual_geoms_idx[1]
            user_scene.geoms.pos[center_sphere_idx] = root_pos + self._z_offset
            user_scene.geoms.mat[center_sphere_idx] = self._center_sphere_mat
            user_scene.geoms.type[center_sphere_idx] = self._center_sphere_type
            user_scene.geoms.size[center_sphere_idx] = self._center_sphere_size
            user_scene.geoms.rgba[center_sphere_idx] = self._center_sphere_color

            if self._visualize_rot_vel:
                rot_vel_arrow_idx = visual_geoms_idx[2]
                user_scene.geoms.pos[rot_vel_arrow_idx] = root_pos + self._z_offset
                user_scene.geoms.mat[rot_vel_arrow_idx] = rot_vel_arrow_mat
                user_scene.geoms.type[rot_vel_arrow_idx] = self._rot_vel_arrow_type
                user_scene.geoms.size[rot_vel_arrow_idx] = self._rot_vel_arrow_size
                user_scene.geoms.rgba[rot_vel_arrow_idx] = self._rot_vel_arrow_color

            geom_pos = user_scene.geoms.pos[self.visual_geoms_idx]
            geom_mat = user_scene.geoms.mat[self.visual_geoms_idx]
            geom_type = user_scene.geoms.type[self.visual_geoms_idx]
            geom_size = user_scene.geoms.size[self.visual_geoms_idx]
            geom_rgba = user_scene.geoms.rgba[self.visual_geoms_idx]

        # Update carry
        new_user_scene = user_scene.replace(geoms=user_scene.geoms.replace(
            pos=geom_pos, mat=geom_mat, size=geom_size, type=geom_type, rgba=geom_rgba))
        carry = carry.replace(user_scene=new_user_scene)

        return carry


class FootPlacementVisualizer:
    """
    Visualizes both stance and swing foot placement targets as semi-transparent boxes.
    - Left foot target shown in BLUE
    - Right foot target shown in ORANGE
    """

    def __init__(self,
                 box_size: Tuple[float, float, float] = (0.1, 0.04, 0.01),
                 left_foot_color: Tuple[float, float, float, float] = (0.2, 0.6, 1.0, 0.7),     # BLUE
                 right_foot_color: Tuple[float, float, float, float] = (1.0, 0.6, 0.2, 0.7)):   # ORANGE
        
        # Two geoms: one for stance, one for swing
        self._n_visual_geoms = 2
        self._box_size = np.array(box_size)
        self._box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
        self._colors = np.array([left_foot_color, right_foot_color])

    def set_visuals(self,
                    goal: Union[np.ndarray, jnp.ndarray],
                    env: Any,
                    model: Union[MjModel, Model],
                    data: Union[MjData, Data],
                    carry: Any,
                    visual_geoms_idx: List[int],
                    backend: ModuleType) -> Any:
        """
        Draws both the swing and stance target boxes.

        Goal vector layout:
            [0:3]  swing_target_pos
            [3:7]  swing_target_orn
            [7:10] stance_target_pos
            [10:14] stance_target_orn
            [14:16] swing_foot_one_hot
        """

        R = np_R if backend == np else jnp_R
        user_scene = carry.user_scene
        geoms = user_scene.geoms

        # --- Parse goal vector ---
        goal_state = getattr(carry.observation_states, self.name)
        swing_pos = goal_state.swing_target_pos 
        swing_orn = goal_state.swing_target_orn 
        swing_one_hot = jax.nn.one_hot(goal_state.swing_foot_idx, 2)

        # --- Determine which foot is swing / stance ---
        swing_color = backend.sum(swing_one_hot[:, None] * backend.array(self._colors), axis=0)

        # Convert quaternions to rotation matrices
        swing_mat = R.from_quat(quat_scalarfirst2scalarlast(swing_orn)).as_matrix().reshape(-1)

        # --- Update both geoms ---
        if backend == jnp:
            # JAX backend (immutable)
            geom_pos = geoms.pos
            geom_mat = geoms.mat
            geom_type = geoms.type
            geom_size = geoms.size
            geom_rgba = geoms.rgba

            # Swing target
            geom_pos = geom_pos.at[visual_geoms_idx[0]].set(swing_pos)
            geom_mat = geom_mat.at[visual_geoms_idx[0]].set(swing_mat)
            geom_type = geom_type.at[visual_geoms_idx[0]].set(self._box_type)
            geom_size = geom_size.at[visual_geoms_idx[0]].set(self._box_size)
            geom_rgba = geom_rgba.at[visual_geoms_idx[0]].set(swing_color)

        else:
            # NumPy backend (mutable)
            geom_pos = geoms.pos.copy()
            geom_mat = geoms.mat.copy()
            geom_type = geoms.type.copy()
            geom_size = geoms.size.copy()
            geom_rgba = geoms.rgba.copy()

            # Swing target
            geom_pos[visual_geoms_idx[0]] = swing_pos
            geom_mat[visual_geoms_idx[0]] = swing_mat
            geom_type[visual_geoms_idx[0]] = self._box_type
            geom_size[visual_geoms_idx[0]] = self._box_size
            geom_rgba[visual_geoms_idx[0]] = np.array(swing_color)

        # Create new geoms and update carry
        new_geoms = geoms.replace(
            pos=geom_pos, mat=geom_mat, size=geom_size, type=geom_type, rgba=geom_rgba
        )
        new_user_scene = user_scene.replace(geoms=new_geoms)
        return carry.replace(user_scene=new_user_scene)


class OldDoubleFootPlacementVisualizer:
    """
    Visualizes both stance and swing foot placement targets as semi-transparent boxes.
    - Left foot target shown in BLUE
    - Right foot target shown in ORANGE
    """

    def __init__(self,
                 box_size: Tuple[float, float, float] = (0.1, 0.04, 0.01),
                 left_foot_color: Tuple[float, float, float, float] = (0.2, 0.6, 1.0, 0.7),     # BLUE
                 right_foot_color: Tuple[float, float, float, float] = (1.0, 0.6, 0.2, 0.7)):   # ORANGE
        
        # Two geoms: one for stance, one for swing
        self._n_visual_geoms = 2
        self._box_size = np.array(box_size)
        self._box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
        self._colors = np.array([left_foot_color, right_foot_color])

    def set_visuals(self,
                    goal: Union[np.ndarray, jnp.ndarray],
                    env: Any,
                    model: Union[MjModel, Model],
                    data: Union[MjData, Data],
                    carry: Any,
                    visual_geoms_idx: List[int],
                    backend: ModuleType) -> Any:
        """
        Draws both the swing and stance target boxes.
        """

        R = np_R if backend == np else jnp_R
        user_scene = carry.user_scene
        geoms = user_scene.geoms

        # Parse goal vector
        goal_state = getattr(carry.observation_states, self.name)
        left_pos = goal_state.left_foot_target_pos 
        left_orn = goal_state.left_foot_target_orn 
        right_pos = goal_state.right_foot_target_pos 
        right_orn = goal_state.right_foot_target_orn 
        # swing_one_hot = jax.nn.one_hot(goal_state.swing_foot_idx, 2)
        
        # take the movement direction yaw
        mov_dir_yaw = goal_state.movement_direction
        dir_x = backend.cos(mov_dir_yaw)
        dir_y = backend.sin(mov_dir_yaw)
        direction_vector = backend.array([dir_x, dir_y, 0])

        # Convert quaternions to rotation matrices
        left_mat = R.from_quat(quat_scalarfirst2scalarlast(left_orn)).as_matrix().reshape(-1)
        right_mat = R.from_quat(quat_scalarfirst2scalarlast(right_orn)).as_matrix().reshape(-1)

        # --- Update both geoms ---
        if backend == jnp:
            # JAX backend (immutable)
            geom_pos = geoms.pos
            geom_mat = geoms.mat
            geom_type = geoms.type
            geom_size = geoms.size
            geom_rgba = geoms.rgba

            # Left foot target (index 0)
            geom_pos = geom_pos.at[visual_geoms_idx[0]].set(left_pos)
            geom_mat = geom_mat.at[visual_geoms_idx[0]].set(left_mat)
            geom_type = geom_type.at[visual_geoms_idx[0]].set(self._box_type)
            geom_size = geom_size.at[visual_geoms_idx[0]].set(self._box_size)
            geom_rgba = geom_rgba.at[visual_geoms_idx[0]].set(self._colors[0])

            # Right foot target (index 1)
            geom_pos = geom_pos.at[visual_geoms_idx[1]].set(right_pos)
            geom_mat = geom_mat.at[visual_geoms_idx[1]].set(right_mat)
            geom_type = geom_type.at[visual_geoms_idx[1]].set(self._box_type)
            geom_size = geom_size.at[visual_geoms_idx[1]].set(self._box_size)
            geom_rgba = geom_rgba.at[visual_geoms_idx[1]].set(self._colors[1])

        else:
            # NumPy backend (mutable)
            geom_pos = geoms.pos.copy()
            geom_mat = geoms.mat.copy()
            geom_type = geoms.type.copy()
            geom_size = geoms.size.copy()
            geom_rgba = geoms.rgba.copy()

            # Left foot target (index 0)
            geom_pos[visual_geoms_idx[0]] = left_pos
            geom_mat[visual_geoms_idx[0]] = left_mat
            geom_type[visual_geoms_idx[0]] = self._box_type
            geom_size[visual_geoms_idx[0]] = self._box_size
            geom_rgba[visual_geoms_idx[0]] = self._colors[0]

            # Right foot target (index 1)
            geom_pos[visual_geoms_idx[1]] = right_pos
            geom_mat[visual_geoms_idx[1]] = right_mat
            geom_type[visual_geoms_idx[1]] = self._box_type
            geom_size[visual_geoms_idx[1]] = self._box_size
            geom_rgba[visual_geoms_idx[1]] = self._colors[1]

        # --- Create new geoms and update carry ---
        new_geoms = geoms.replace(
            pos=geom_pos, mat=geom_mat, size=geom_size, type=geom_type, rgba=geom_rgba
        )
        new_user_scene = user_scene.replace(geoms=new_geoms)
        return carry.replace(user_scene=new_user_scene)

class DoubleFootPlacementVisualizer:
    """
    Visualizes both stance and swing foot placement targets as semi-transparent boxes
    and the desired movement direction as an arrow.
    - Left foot target shown in BLUE
    - Right foot target shown in ORANGE
    - Movement direction shown in GREEN
    - Feet direction shown in MAGENTA
    """

    def __init__(self,
                 box_size: Tuple[float, float, float] = (0.1, 0.04, 0.01),
                 left_foot_color: Tuple[float, float, float, float] = (0.2, 0.6, 1.0, 0.7),     # BLUE
                 right_foot_color: Tuple[float, float, float, float] = (1.0, 0.6, 0.2, 0.7)):   # ORANGE
        
        # Two geoms (boxes) + one geom (arrow) + another geom for the feet direction arrow
        self._n_visual_geoms = 6 # 4 # 3
        
        # Box properties
        self._box_size = np.array(box_size)
        self._box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
        self._colors = np.array([left_foot_color, right_foot_color])

        # Arrow properties
        self._dir_arrow_type = int(mujoco.mjtGeom.mjGEOM_ARROW)
        self._dir_arrow_size = np.array([0.015, 0.03, 1.0]) # [shaft_radius, head_radius, length]
        self._dir_arrow_color = np.array([0.0, 1.0, 0.0, 0.25]) # Green
        
        # Arrow 2 properties
        self._feet_arrow_type = int(mujoco.mjtGeom.mjGEOM_ARROW)
        self._feet_arrow_size = np.array([0.015, 0.03, 0.5]) # [shaft_radius, head_radius, length]
        self._feet_arrow_color = np.array([1.0, 0.0, 1.0, 0.25]) # Magenta
        
        # Toe arrow properties
        self._toe_arrow_type = int(mujoco.mjtGeom.mjGEOM_ARROW)
        self._toe_arrow_size = np.array([0.01, 0.02, 0.25])  # small
        self._left_toe_color  = np.array([1.0, 0.0, 0.0, 0.7])   # RED
        self._right_toe_color = np.array([1.0, 0.0, 0.0, 0.7])   # RED

    def set_visuals(self,
                    goal: Union[np.ndarray, jnp.ndarray],
                    env: Any,
                    model: Union[MjModel, Model],
                    data: Union[MjData, Data],
                    carry: Any,
                    visual_geoms_idx: List[int],
                    backend: ModuleType) -> Any:
        """
        Draws both the swing and stance target boxes.
        """

        R = np_R if backend == np else jnp_R
        user_scene = carry.user_scene
        geoms = user_scene.geoms

        # Parse goal vector
        goal_state = getattr(carry.observation_states, self.name)
        left_pos = goal_state.left_foot_target_pos 
        left_orn = goal_state.left_foot_target_orn 
        right_pos = goal_state.right_foot_target_pos 
        right_orn = goal_state.right_foot_target_orn 
        
        # take the movement direction yaw
        mov_dir_yaw = goal_state.movement_direction
        
        # take the feet direction yaw
        feet_dir_yaw = goal_state.feet_direction

        # Convert quaternions to rotation matrices
        left_mat = R.from_quat(quat_scalarfirst2scalarlast(left_orn)).as_matrix().reshape(-1)
        right_mat = R.from_quat(quat_scalarfirst2scalarlast(right_orn)).as_matrix().reshape(-1)

        # Calculate direction arrow properties
        dir_arrow_pos = (left_pos + right_pos) / 2.0 + backend.array([0.0, 0.0, 0.02])
        R_y_90_mat = R.from_euler("y", 90, degrees=True).as_matrix()
        R_z_yaw_mat = R.from_euler("z", mov_dir_yaw, degrees=False).as_matrix()
        # Combine rotations: first, point Z-up arrow to X-forward, then rotate by yaw
        dir_arrow_mat = (R_z_yaw_mat @ R_y_90_mat).reshape(-1)
        
        # do the same for the feet directiona rrow
        feet_arrow_pos = (left_pos + right_pos) / 2.0 + backend.array([0.0, 0.0, 0.02])
        fR_y_90_mat = R.from_euler("y", 90, degrees=True).as_matrix()
        fR_z_yaw_mat = R.from_euler("z", feet_dir_yaw, degrees=False).as_matrix()
        # Combine rotations: first, point Z-up arrow to X-forward, then rotate by yaw
        feet_arrow_mat = (fR_z_yaw_mat @ fR_y_90_mat).reshape(-1)
        
        # orientation arrows for feet
        # Extract yaw of left and right foot targets
        left_yaw  = R.from_matrix(left_mat.reshape(3,3)).as_euler("xyz")[2]
        right_yaw = R.from_matrix(right_mat.reshape(3,3)).as_euler("xyz")[2]

        # Create arrow rotation: Z-up → X-forward, then apply yaw
        arrow_align = R.from_euler("y", 90, degrees=True).as_matrix()

        left_arrow_rot  = (R.from_euler("z", left_yaw).as_matrix() @ arrow_align).reshape(-1)
        right_arrow_rot = (R.from_euler("z", right_yaw).as_matrix() @ arrow_align).reshape(-1)

        # Toe arrow positions (placed slightly above ground)
        left_toe_pos  = left_pos + backend.array([0.5 * self._box_size[0], 0., 0.5 * self._box_size[2]])
        right_toe_pos = right_pos + backend.array([0.5 * self._box_size[0], 0., 0.5 * self._box_size[2]])

        # --- Update all geoms ---
        if backend == jnp:
            # JAX backend (immutable)
            geom_pos = geoms.pos
            geom_mat = geoms.mat
            geom_type = geoms.type
            geom_size = geoms.size
            geom_rgba = geoms.rgba

            # Left foot target (index 0)
            geom_pos = geom_pos.at[visual_geoms_idx[0]].set(left_pos)
            geom_mat = geom_mat.at[visual_geoms_idx[0]].set(left_mat)
            geom_type = geom_type.at[visual_geoms_idx[0]].set(self._box_type)
            geom_size = geom_size.at[visual_geoms_idx[0]].set(self._box_size)
            geom_rgba = geom_rgba.at[visual_geoms_idx[0]].set(self._colors[0])

            # Right foot target (index 1)
            geom_pos = geom_pos.at[visual_geoms_idx[1]].set(right_pos)
            geom_mat = geom_mat.at[visual_geoms_idx[1]].set(right_mat)
            geom_type = geom_type.at[visual_geoms_idx[1]].set(self._box_type)
            geom_size = geom_size.at[visual_geoms_idx[1]].set(self._box_size)
            geom_rgba = geom_rgba.at[visual_geoms_idx[1]].set(self._colors[1])
            
            # Direction arrow (index 2)
            geom_pos = geom_pos.at[visual_geoms_idx[2]].set(dir_arrow_pos)
            geom_mat = geom_mat.at[visual_geoms_idx[2]].set(dir_arrow_mat)
            geom_type = geom_type.at[visual_geoms_idx[2]].set(self._dir_arrow_type)
            geom_size = geom_size.at[visual_geoms_idx[2]].set(self._dir_arrow_size)
            geom_rgba = geom_rgba.at[visual_geoms_idx[2]].set(self._dir_arrow_color)
            
            # Direction arrow (index 3)
            geom_pos = geom_pos.at[visual_geoms_idx[3]].set(feet_arrow_pos)
            geom_mat = geom_mat.at[visual_geoms_idx[3]].set(feet_arrow_mat)
            geom_type = geom_type.at[visual_geoms_idx[3]].set(self._feet_arrow_type)
            geom_size = geom_size.at[visual_geoms_idx[3]].set(self._feet_arrow_size)
            geom_rgba = geom_rgba.at[visual_geoms_idx[3]].set(self._feet_arrow_color)
            
            # Toe arrow positions (placed slightly above ground)
            # Left toe arrow (index 4)
            geom_pos = geom_pos.at[visual_geoms_idx[4]].set(left_toe_pos)
            geom_mat = geom_mat.at[visual_geoms_idx[4]].set(left_arrow_rot)
            geom_type = geom_type.at[visual_geoms_idx[4]].set(self._toe_arrow_type)
            geom_size = geom_size.at[visual_geoms_idx[4]].set(self._toe_arrow_size)
            geom_rgba = geom_rgba.at[visual_geoms_idx[4]].set(self._left_toe_color)

            # Right toe arrow (index 5)
            geom_pos = geom_pos.at[visual_geoms_idx[5]].set(right_toe_pos)
            geom_mat = geom_mat.at[visual_geoms_idx[5]].set(right_arrow_rot)
            geom_type = geom_type.at[visual_geoms_idx[5]].set(self._toe_arrow_type)
            geom_size = geom_size.at[visual_geoms_idx[5]].set(self._toe_arrow_size)
            geom_rgba = geom_rgba.at[visual_geoms_idx[5]].set(self._right_toe_color)

        else:
            # NumPy backend (mutable)
            geom_pos = geoms.pos.copy()
            geom_mat = geoms.mat.copy()
            geom_type = geoms.type.copy()
            geom_size = geoms.size.copy()
            geom_rgba = geoms.rgba.copy()

            # Left foot target (index 0)
            geom_pos[visual_geoms_idx[0]] = left_pos
            geom_mat[visual_geoms_idx[0]] = left_mat
            geom_type[visual_geoms_idx[0]] = self._box_type
            geom_size[visual_geoms_idx[0]] = self._box_size
            geom_rgba[visual_geoms_idx[0]] = self._colors[0]

            # Right foot target (index 1)
            geom_pos[visual_geoms_idx[1]] = right_pos
            geom_mat[visual_geoms_idx[1]] = right_mat
            geom_type[visual_geoms_idx[1]] = self._box_type
            geom_size[visual_geoms_idx[1]] = self._box_size
            geom_rgba[visual_geoms_idx[1]] = self._colors[1]

            # Direction arrow (index 2)
            geom_pos[visual_geoms_idx[2]] = dir_arrow_pos
            geom_mat[visual_geoms_idx[2]] = dir_arrow_mat
            geom_type[visual_geoms_idx[2]] = self._dir_arrow_type
            geom_size[visual_geoms_idx[2]] = self._dir_arrow_size
            geom_rgba[visual_geoms_idx[2]] = self._dir_arrow_color
            
            # Direction arrow (index 3)
            geom_pos[visual_geoms_idx[3]] = feet_arrow_pos
            geom_mat[visual_geoms_idx[3]] = feet_arrow_mat
            geom_type[visual_geoms_idx[3]] = self._feet_arrow_type
            geom_size[visual_geoms_idx[3]] = self._feet_arrow_size
            geom_rgba[visual_geoms_idx[3]] = self._feet_arrow_color
            
            # Toe arrow positions (placed slightly above ground)
            # Left toe arrow (index 4)
            geom_pos[visual_geoms_idx[4]] = left_toe_pos
            geom_mat[visual_geoms_idx[4]] = left_arrow_rot
            geom_type[visual_geoms_idx[4]] = self._toe_arrow_type
            geom_size[visual_geoms_idx[4]] = self._toe_arrow_size
            geom_rgba[visual_geoms_idx[4]] = self._left_toe_color

            # Right toe arrow (index 5)
            geom_pos[visual_geoms_idx[5]] = right_toe_pos
            geom_mat[visual_geoms_idx[5]] = right_arrow_rot
            geom_type[visual_geoms_idx[5]] = self._toe_arrow_type
            geom_size[visual_geoms_idx[5]] = self._toe_arrow_size
            geom_rgba[visual_geoms_idx[5]] = self._right_toe_color

        # --- Create new geoms and update carry ---
        new_geoms = geoms.replace(
            pos=geom_pos, mat=geom_mat, size=geom_size, type=geom_type, rgba=geom_rgba
        )
        new_user_scene = user_scene.replace(geoms=new_geoms)
        return carry.replace(user_scene=new_user_scene)