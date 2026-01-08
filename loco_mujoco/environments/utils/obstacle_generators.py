import mujoco, numpy as np, jax.numpy as jnp
from typing import Dict, List, Tuple, Any, Union
from jax._src.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R

def add_box(
        world_body: Any, 
        name: str,
        coordinates: Union[jnp.array, np.array, List, Tuple] = [0,0,0], 
        length: float = 1,
        height: float = 1,
        width: float = 1,
        color: Union[jnp.array, np.array, List, Tuple] = [0,0,0,1],
        orientation_yaw_deg: float = 0.0,
        friction: Union[jnp.array, np.array, List, Tuple] = [1.0, 0.005, 0.0001],
        priority: int = 0,
        backend: Any = np
        ):
    """
    Add a box geom to the world body and return it.

    NOTE: 
    Size of the box is (half x, half y, half z).
    The position is the center of mass.
    """
    R = np_R if backend == np else jnp_R
    yaw_rad = backend.deg2rad(orientation_yaw_deg)
    des_quat = (R.from_euler('z', yaw_rad)).as_quat(scalar_first=True)

    world_body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(length / 2, width / 2, height / 2), 
        pos=(coordinates[0], coordinates[1], coordinates[2]), 
        quat=(des_quat[0], des_quat[1], des_quat[2], des_quat[3]),
        rgba=(color[0], color[1], color[2], color[3]),
        priority=priority,
        friction=(friction[0],friction[1],friction[2])
    )
    return world_body

def add_stair(
        world_body: Any, 
        name: str,
        first_step_coordinates: Union[jnp.array, np.array, List, Tuple] = [0,0,0], 
        num_steps: int = 1,
        step_height: float = 1,
        step_length: float = 1,
        step_width: float = 1,
        down: bool = False,
        color: Union[jnp.array, np.array, List, Tuple] = [0,0,0,1],
        orientation_yaw_deg: float = 0.0,
        friction: Union[jnp.array, np.array, List, Tuple] = [1.0, 0.005, 0.0001],
        priority: int = 0,
        backend: Any = np
        ):
    """
    Add a box geoms to form a stair, based on add_box function
    """
    R = np_R if backend == np else jnp_R
    yaw_rad = backend.deg2rad(orientation_yaw_deg)
    des_quat = (R.from_euler('z', yaw_rad)).as_quat(scalar_first=True)

    run_coo = backend.array(first_step_coordinates)
    
    # Calculate the vector to move to the next step
    # X/Y move forward based on rotation
    dx = step_length * backend.cos(yaw_rad)
    dy = step_length * backend.sin(yaw_rad)
    
    # Z moves up or down
    dz = -step_height if down else step_height

    for i in range(num_steps):
        world_body.add_geom(
            name=f"{name}_{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(step_length/2, step_width / 2, step_height / 2), 
            pos=(run_coo[0], run_coo[1], run_coo[2]), 
            quat=(des_quat[0], des_quat[1], des_quat[2], des_quat[3]),
            rgba=(color[0], color[1], color[2], color[3]),
            priority=priority,
            friction=(friction[0],friction[1],friction[2])
        )

        # Move coordinate for the NEXT step
        run_coo = run_coo + backend.array([dx, dy, dz])
        
    return world_body

def add_slope(
        world_body: Any, 
        name: str,
        coordinates: Union[jnp.array, np.array, List, Tuple] = [0,0,0], 
        run: float = 1,          # Horizontal length
        rise: float = 1,         # Vertical height
        width: float = 1,
        thickness: float = 0.05, # Thickness of the ramp board
        down: bool = False,
        color: Union[jnp.array, np.array, List, Tuple] = [0,0,0,1],
        orientation_yaw_deg: float = 0.0,
        friction: Union[jnp.array, np.array, List, Tuple] = [1.0, 0.005, 0.0001],
        priority: int = 0,
        backend: Any = np
        ):
    """
    Add a rotated box geom to act as a slope (ramp).
    
    NOTE:
    'coordinates' specifies the BOTTOM-CENTER start point of the ramp surface.
    The function automatically calculates the geometric center and rotation.
    """
    R = np_R if backend == np else jnp_R
    
    # Determine direction of the height change
    # If down is True, we go DOWN (-rise). If False, we go UP (+rise).
    height_delta = -rise if down else rise
    
    # Geometry
    hypotenuse = backend.sqrt(run**2 + rise**2)
    slope_angle_rad = backend.arctan2(height_delta, run) 
    
    # Orientation
    r_pitch = R.from_euler('y', -slope_angle_rad) 
    r_yaw = R.from_euler('z', backend.deg2rad(orientation_yaw_deg))
    
    des_quat = (r_yaw * r_pitch).as_quat(scalar_first=True)

    # Position
    start_pos = backend.array(coordinates)
    yaw_rad = backend.deg2rad(orientation_yaw_deg)
    dx = (run / 2.0) * backend.cos(yaw_rad)
    dy = (run / 2.0) * backend.sin(yaw_rad)
    dz = height_delta / 2.0 
    
    final_pos = start_pos + backend.array([dx, dy, dz])

    world_body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(hypotenuse / 2, width / 2, thickness / 2), 
        pos=(final_pos[0], final_pos[1], final_pos[2]), 
        quat=(des_quat[0], des_quat[1], des_quat[2], des_quat[3]),
        rgba=(color[0], color[1], color[2], color[3]),
        priority=priority,
        friction=(friction[0], friction[1], friction[2])
    )
    return world_body

def add_ramp_platform_ramp(
        world_body: Any, 
        name: str,
        coordinates: Union[jnp.array, np.array, List, Tuple] = [0,0,0], 
        run: float = 1.0,           # Horizontal length of the slope part
        rise: float = 0.5,          # Height of the slope
        platform_length: float = 1.0, # Length of the flat top section
        platform_width: float = 1.0, # Length of the flat top section
        width: float = 1.0,
        thickness: float = 0.05,
        orientation_yaw_deg: float = 0.0,
        color: Union[jnp.array, np.array, List, Tuple] = [0.2, 0.2, 0.2, 1],
        friction: Union[jnp.array, np.array, List, Tuple] = [1.0, 0.005, 0.0001],
        backend: Any = np
        ):
    """
    Creates a structure: Slope Up -> Flat Platform -> Slope Down.
    """
    add_slope(
        world_body=world_body,
        name=f"{name}_up",
        coordinates=coordinates,
        run=run,
        rise=rise,
        width=width,
        thickness=thickness,
        down=False,
        orientation_yaw_deg=orientation_yaw_deg,
        color=color,
        friction=friction,
        backend=backend
    )
    yaw_rad = backend.deg2rad(orientation_yaw_deg)
    
    dx_slope = run * backend.cos(yaw_rad)
    dy_slope = run * backend.sin(yaw_rad)
    
    plat_start_x = coordinates[0] + dx_slope
    plat_start_y = coordinates[1] + dy_slope
    plat_start_z = coordinates[2] + rise
    
    dx_plat_half = (platform_length / 2.0) * backend.cos(yaw_rad)
    dy_plat_half = (platform_length / 2.0) * backend.sin(yaw_rad)
    
    plat_center_x = plat_start_x + dx_plat_half
    plat_center_y = plat_start_y + dy_plat_half
    
    plat_center_z = plat_start_z - (thickness / 2.0)

    add_box(
        world_body=world_body,
        name=f"{name}_platform",
        coordinates=[plat_center_x, plat_center_y, plat_center_z],
        length=platform_length,
        width=platform_width,
        height=thickness, # "Height" of a box is its Z-thickness here
        orientation_yaw_deg=orientation_yaw_deg,
        color=color,
        friction=friction,
        backend=backend
    )
    
    dx_plat_full = platform_length * backend.cos(yaw_rad)
    dy_plat_full = platform_length * backend.sin(yaw_rad)
    
    down_start_x = plat_start_x + dx_plat_full
    down_start_y = plat_start_y + dy_plat_full
    down_start_z = plat_start_z 

    add_slope(
        world_body=world_body,
        name=f"{name}_down",
        coordinates=[down_start_x, down_start_y, down_start_z],
        run=run,
        rise=rise,
        width=width,
        thickness=thickness,
        down=True, 
        orientation_yaw_deg=orientation_yaw_deg,
        color=color,
        friction=friction,
        backend=backend
    )

    return world_body