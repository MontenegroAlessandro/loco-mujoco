from typing import Any, Dict, List, Union, Tuple
from types import ModuleType

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from flax import struct
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.utils.backend import assert_backend_is_supported
from loco_mujoco.core.control_functions import ControlFunction
from loco_mujoco.core.utils import mj_jntname2qposid, mj_jntname2qvelid


@struct.dataclass
class PDControlState:
    """
    This state can be used by the domain randomizer to add noise to the PD-controller.
    """
    p_gain_noise: Union[np.ndarray, jax.Array]
    d_gain_noise: Union[np.ndarray, jax.Array]
    pos_offset: Union[np.ndarray, jax.Array]
    ctrl_mult: Union[np.ndarray, jax.Array]


class PDControl(ControlFunction):

    """
    PD controller function setting positions. This controller internally normalizes the action space to [-1, 1]
    for the agent but uses the joint position limits for the environment.

    """

    def __init__(self,
                 env: Any,
                 p_gain: Union[float, np.ndarray],
                 d_gain: Union[float, np.ndarray],
                 nominal_joint_positions: np.ndarray = None,
                 scale_action_to_jnt_limits: bool = False,
                 action_scale: float = 1.0,
                 clip_actions: float = 1.0,
                 expand_action_space_by: int = 0,
                 **kwargs: Any):
        """
        Initialize the PDControl class.

        Args:
            env (Any): The environment instance containing model and specifications.
            p_gain (Union[float, np.ndarray]): Proportional gain for the PD controller.
            d_gain (Union[float, np.ndarray]): Derivative gain for the PD controller.
            nominal_joint_positions (np.ndarray, optional): Default joint positions. If not provided, uses qpos0.
            scale_action_to_jnt_limits (bool): If true, the actions are scaled to the joint limits.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        self._init_p_gain = np.array(p_gain)
        self._init_d_gain = np.array(d_gain)
        self._ctrl_ranges = []
        self._jnt_ranges = []
        self._jnt_names = []

        self._scale_action_to_jnt_limits = scale_action_to_jnt_limits
        self._action_scale = action_scale
        self._clip_actions = clip_actions
        
        for actuator in env.mjspec.actuators:
            jnt_name = actuator.target
            ctrl_range = actuator.ctrlrange if actuator.ctrllimited else np.array([-np.inf, np.inf])
            self._ctrl_ranges.append(ctrl_range)
            for j in env.mjspec.joints:
                if j.name == jnt_name:
                    assert j.type == mujoco.mjtJoint.mjJNT_HINGE or j.type == mujoco.mjtJoint.mjJNT_SLIDE, \
                        "Only Hinge and Slide joints are supported for PDControl."
                    self._jnt_names.append(jnt_name)
                    self._jnt_ranges.append(j.range)

        # sort according to action ind
        self._ctrl_ranges = np.concatenate([self._ctrl_ranges[i].reshape(1, 2) for i in env._action_indices])
        self._jnt_ranges = np.concatenate([self._jnt_ranges[i].reshape(1, 2) for i in env._action_indices])
        self._jnt_names = [self._jnt_names[i] for i in env._action_indices]

        # get qpos and qvel ids
        self._qpos_ids = np.concatenate([mj_jntname2qposid(name, env._model) for name in self._jnt_names])
        self._qvel_ids = np.concatenate([mj_jntname2qvelid(name, env._model) for name in self._jnt_names])

        if nominal_joint_positions is None:
            self._nominal_joint_positions = env._model.qpos0[self._qpos_ids]
        else:
            self._nominal_joint_positions = np.array(nominal_joint_positions)

        self._high_pos_target = self._jnt_ranges[:, 1] - self._nominal_joint_positions
        self._low_pos_target = self._jnt_ranges[:, 0] - self._nominal_joint_positions

        # calculate mean and delta
        self.norm_act_mean = (self._high_pos_target + self._low_pos_target) / 2.0
        self.norm_act_delta = (self._high_pos_target - self._low_pos_target) / 2.0

        # set the action space limits for the agent to -1 and 1
        # low = -np.ones_like(self.norm_act_mean)
        # high = np.ones_like(self.norm_act_mean)

        low = self._low_pos_target
        high = self._high_pos_target

        if expand_action_space_by > 0:
            low = np.concatenate([low, -np.ones(expand_action_space_by)], axis=0)
            high = np.concatenate([high, np.ones(expand_action_space_by)], axis=0)

        super(PDControl, self).__init__(env, low, high, **kwargs)

    def init_state(self,
                   env: Any,
                   key: Union[jax.random.PRNGKey, Any],
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType) -> PDControlState:
        """
        Initialize the state for PDControl.

        Args:
            env (Any): The environment instance.
            key (Union[jax.random.PRNGKey, Any]): Random key for noise generation, specific to JAX.
            model (Union[MjModel, Model]): The simulation model instance.
            data (Union[MjData, Data]): The simulation data instance.
            backend (ModuleType): Backend module (e.g., numpy or jax.numpy) for computation.

        Returns:
            PDControlState: The initialized PD control state containing zeroed noise values, offsets, and multipliers.

        Raises:
            ValueError: If the backend module is not supported.
        """
        assert_backend_is_supported(backend)
        return PDControlState(p_gain_noise=backend.zeros_like(self._nominal_joint_positions),
                              d_gain_noise=backend.zeros_like(self._nominal_joint_positions),
                              pos_offset=backend.zeros_like(self._nominal_joint_positions),
                              ctrl_mult=backend.ones_like(self._nominal_joint_positions))

    def generate_action(self, env: Any,
                        action: Union[np.ndarray, jax.Array],
                        model: Union[MjModel, Model],
                        data: Union[MjData, Data],
                        carry: Any,
                        backend: ModuleType) -> Tuple[Union[np.ndarray, jax.Array], Any]:
        """
        Generate the action using the PD controller. This function expects the action to be in the range [-1, 1].

        Args:
            env (Any): The environment instance.
            action (Union[np.ndarray, jax.Array]): The action to be updated.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jax.Array], Any]: The updated action and carry.

        Raises:
            ValueError: If the backend module is not supported.
        """
        assert_backend_is_supported(backend)

        action = backend.clip(action, -self._clip_actions, self._clip_actions)

        if self._scale_action_to_jnt_limits:
            unnormalized_action = self._unnormalize_action(action)
        else:
            unnormalized_action = action

        unnormalized_action = unnormalized_action * self._action_scale

        pd_state = carry.control_func_state

        p_gain = pd_state.p_gain_noise + self._init_p_gain
        d_gain = pd_state.d_gain_noise + self._init_d_gain
        offsets = pd_state.pos_offset

        target_joint_pos = backend.clip(self._nominal_joint_positions + unnormalized_action + offsets*0.0,
                                        self._jnt_ranges[:, 0], self._jnt_ranges[:, 1])

        ctrl = p_gain * (target_joint_pos - data.qpos[self._qpos_ids]) - d_gain * data.qvel[self._qvel_ids]
        ctrl = backend.clip(ctrl * pd_state.ctrl_mult, self._ctrl_ranges[:, 0], self._ctrl_ranges[:, 1])

        return ctrl, carry

    def _unnormalize_action(self, action: Union[np.ndarray, jax.Array]) -> Union[np.ndarray, jax.Array]:
        """
        Rescale the action from [-1, 1] to the desired action space.

        Args:
            action (Union[np.ndarray, jax.Array]): The action to be unnormalized.

        Returns:
            Union[np.ndarray, jax.Array]: The unnormalized action

        """
        unnormalized_action = ((action * self.norm_act_delta) + self.norm_act_mean)
        return unnormalized_action

    @property
    def run_with_simulation_frequency(self):
        """
        If true, the control function is called with the simulation frequency.
        """
        return True


@struct.dataclass
class PDControlGaitState(PDControlState):
    """
    Extended state for PDControlGait that includes gait phase information.
    """
    gait_phase_offset: Union[np.ndarray, jax.Array]  # Store the gait offset for use in goals/rewards
    
class PDControlGait(PDControl):
    """
    Extended PD controller handling an additional output of the policy that will serve as an offset
    to update the gait phase indicator in other points of the training pipeline.
    """

    def __init__(
            self,
            env: Any,
            gait_phase_delta_max: float = 0.1,
            gait_action_name: str = "gait_phase_offset", 
            **kwargs: Any
        ):
        
        self.gait_phase_delta_max = gait_phase_delta_max
        self.gait_action_name = gait_action_name
        
        super().__init__(env, expand_action_space_by=1, **kwargs)

        self._num_physical_actions = len(env._action_indices)
        
        if hasattr(env, '_full_actuation_spec'):
            try:
                self._gait_action_idx = env._full_actuation_spec.index(gait_action_name)
            except ValueError:
                raise ValueError(f"Gait action '{gait_action_name}' not found in actuation_spec!")
        else:
            self._gait_action_idx = self._num_physical_actions
        
        # Verify the index is correct
        total_actions = self._num_physical_actions + 1 
        if self._gait_action_idx >= total_actions:
            raise ValueError(f"Gait action index {self._gait_action_idx} exceeds total actions {total_actions}")

    def init_state(self,
                   env: Any,
                   key: Union[jax.random.PRNGKey, Any],
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType) -> PDControlGaitState:
        """
        Initialize the state for PDControlGait, extending the base state with gait phase offset.
        
        Args:
            env (Any): The environment instance.
            key (Union[jax.random.PRNGKey, Any]): Random key for noise generation.
            model (Union[MjModel, Model]): The simulation model instance.
            data (Union[MjData, Data]): The simulation data instance.
            backend (ModuleType): Backend module (e.g., numpy or jax.numpy).
            
        Returns:
            PDControlGaitState: The initialized state with gait phase information.
        """
        assert_backend_is_supported(backend)
        return PDControlGaitState(
            p_gain_noise=backend.zeros_like(self._nominal_joint_positions),
            d_gain_noise=backend.zeros_like(self._nominal_joint_positions),
            pos_offset=backend.zeros_like(self._nominal_joint_positions),
            ctrl_mult=backend.ones_like(self._nominal_joint_positions),
            gait_phase_offset=backend.array(0.0) 
        )

    def generate_action(self, env: Any,
                        action: Union[np.ndarray, jax.Array],
                        model: Union[MjModel, Model],
                        data: Union[MjData, Data],
                        carry: Any,
                        backend: ModuleType) -> Tuple[Union[np.ndarray, jax.Array], Any]:
        
        assert_backend_is_supported(backend)
        
        if self._gait_action_idx == len(action) - 1:
            pd_action = action[:-1]
            gait_raw = action[-1]
        else:
            mask = backend.ones(len(action), dtype=bool)
            mask = mask.at[self._gait_action_idx].set(False)
            pd_action = action[mask]
            gait_raw = action[self._gait_action_idx]

        gait_offset = self.gait_phase_delta_max / (1 + backend.exp(-gait_raw)) 
        
        updated_control_state = carry.control_func_state.replace(gait_phase_offset=gait_offset)
        carry = carry.replace(control_func_state=updated_control_state)
        
        ctrl, carry = super().generate_action(env, pd_action, model, data, carry, backend)
        
        return ctrl, carry