from typing import Any, Union, Tuple
from types import ModuleType

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
import mujoco
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.domain_randomizer import DomainRandomizer
from loco_mujoco.core.control_functions import ControlFunction, PDControl
from loco_mujoco.core.utils.backend import assert_backend_is_supported


@struct.dataclass
class CustomRandomizerState:
    """
    Represents the state of the default randomizer.

    """

    gravity: Union[np.ndarray, jax.Array]
    geom_friction: Union[np.ndarray, jax.Array]
    floor_friction: Union[np.ndarray, jax.Array]
    geom_stiffness: Union[np.ndarray, jax.Array]
    geom_damping: Union[np.ndarray, jax.Array]
    base_mass_to_add: float
    com_displacement: Union[np.ndarray, jax.Array]
    link_mass_multipliers: Union[np.ndarray, jax.Array]
    joint_friction_loss: Union[np.ndarray, jax.Array]
    joint_damping: Union[np.ndarray, jax.Array]
    joint_armature: Union[np.ndarray, jax.Array]
    prev_filtered_action: Union[np.ndarray, jax.Array]
    
    active_root_push_force: Union[np.ndarray, jax.Array]
    push_counter: int = 100000

    curriculum_coefficient: float = 0.0
    kicked: bool = False
    
    filter_alpha: float = 0.0

class CustomRandomizer(DomainRandomizer):
    """
    A domain randomizer class that modifies typical simulation parameters.

    """

    def __init__(self, env, **kwargs):

        # store initial values for reset (only needed for numpy backend)
        self._init_gravity = None
        self._init_geom_friction = None
        self._init_geom_solref = None
        self._init_body_ipos = None
        self._init_body_mass = None
        self._init_dof_frictionloss = None
        self._init_dof_damping = None
        self._init_dof_armature = None

        info_props = env._get_all_info_properties()
        root_body_name = info_props["root_body_name"]
        self._root_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)

        self._floor_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        self._other_body_masks = np.ones(env.model.nbody, dtype=bool)
        self._other_body_masks[0] = False # exclude worldbody
        self._other_body_masks[self._root_body_id] = False

        # some observations are not allowed to be randomized, filter them out
        self._allowed_to_be_randomized = env.obs_container.get_randomizable_obs_indices()

        super().__init__(env, **kwargs)

    def init_state(self,
                   env: Any,
                   key: Any,
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType) -> CustomRandomizerState:
        """
        Initialize the randomizer state.

        Args:
            env (Any): The environment instance.
            key (Any): Random seed key.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            CustomRandomizerState: The initialized randomizer state.

        """

        assert_backend_is_supported(backend)
        if self.rand_conf["apply_action_filtering"]:
            filter_action_alpha = self.rand_conf["filter_action_alpha"]
        else:
            filter_action_alpha = 0.0

        return CustomRandomizerState(gravity=backend.array([0.0, 0.0, -9.81]),
                                      geom_friction=backend.array(model.geom_friction.copy()),
                                      floor_friction=backend.array(model.geom_friction[self._floor_geom_id].copy()),
                                      geom_stiffness=backend.zeros(model.ngeom),
                                      geom_damping=backend.zeros(model.ngeom),
                                      base_mass_to_add=0.0,
                                      com_displacement=backend.array([0.0, 0.0, 0.0]),
                                      link_mass_multipliers=backend.array([1.0] * (model.nbody-1)), #exclude worldbody
                                      joint_friction_loss=backend.array([0.0] * (model.nv-6)), #exclude freejoint 6 dofs
                                      joint_damping=backend.array([0.0] * (model.nv-6)), #exclude freejoint 6 dofs
                                    #   joint_armature=backend.array([0.0] * (model.nv-6)), #exclude freejoint 6 dofs
                                      joint_armature=backend.array(model.dof_armature[6:].copy()), #exclude freejoint 6 dofs
                                      push_counter=100000,
                                      active_root_push_force=backend.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                                      kicked=False,
                                      prev_filtered_action=backend.array([0.0] * (env.action_dim)),
                                      filter_alpha=filter_action_alpha,
                                      )

    def reset(self,
              env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType) -> Tuple[Union[MjData, Data], Any]:
        """
        Reset the randomizer, applying domain randomization.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated simulation data and carry.

        """

        assert_backend_is_supported(backend)
        domain_randomizer_state = carry.domain_randomizer_state

        if backend == np and self._init_body_ipos is None:
            # store initial values for reset
            self._init_gravity = model.opt.gravity.copy()
            self._init_geom_solref = model.geom_solref.copy()
            self._init_body_ipos = model.body_ipos.copy()
            self._init_body_mass = model.body_mass.copy()
            self._init_dof_frictionloss = model.dof_frictionloss.copy()
            self._init_dof_damping = model.dof_damping.copy()
            self._init_dof_armature = model.dof_armature.copy()

        # update different randomization parameters
        gravity, carry = self._sample_gravity(model, carry, backend)
        geom_friction, carry = self._sample_geom_friction(model, carry, backend)
        # floor_friction, carry = self._sample_floor_friction(model, carry, backend)
        geom_damping, geom_stiffness, carry = self._sample_geom_damping_and_stiffness(model, carry, backend)
        base_mass_to_add, carry = self._sample_base_mass(model, carry, backend)
        com_displacement, carry = self._sample_com_displacement(model, carry, backend)
        link_mass_multipliers, carry = self._sample_link_mass_multipliers(model, carry, backend)
        joint_friction_loss, carry = self._sample_joint_friction_loss(model, carry, backend)
        joint_damping, carry = self._sample_joint_damping(model, carry, backend)
        joint_armature, carry = self._sample_joint_armature(model, carry, backend)

        if isinstance(env._control_func, ControlFunction):
            control_func_state = carry.control_func_state

            p_noise, carry = self._sample_p_gains_noise(env, model, carry, backend)
            d_noise, carry = self._sample_d_gains_noise(env, model, carry, backend)
            carry = carry.replace(control_func_state=control_func_state.replace(p_gain_noise=p_noise,
                                                                                d_gain_noise=d_noise,
                                                                                pos_offset=backend.zeros_like(env._control_func._nominal_joint_positions),
                                                                                ctrl_mult=backend.ones_like(env._control_func._nominal_joint_positions)))



        carry = carry.replace(domain_randomizer_state=domain_randomizer_state.replace(gravity=gravity,
                                                                                      geom_friction=geom_friction,
                                                                                      floor_friction=geom_friction[self._floor_geom_id],
                                                                                      geom_stiffness=geom_stiffness,
                                                                                      geom_damping=geom_damping,
                                                                                      base_mass_to_add=base_mass_to_add,
                                                                                      com_displacement=com_displacement,
                                                                                      link_mass_multipliers=link_mass_multipliers,
                                                                                      joint_friction_loss=joint_friction_loss,
                                                                                      joint_damping=joint_damping,
                                                                                      joint_armature=joint_armature,
                                                                                      push_counter=100000,
                                                                                      active_root_push_force=backend.zeros_like(domain_randomizer_state.active_root_push_force),
                                                                                      kicked=False,
                                                                                      prev_filtered_action=backend.zeros_like(domain_randomizer_state.prev_filtered_action)
                                                                                      ))

        return data, carry
    
    def _compute_curriculum_coeff(self, t, num_steps, total_steps, num_envs):
        def single_step(_):
            return 1.0

        def multi_step(_):
            timesteps_per_step = total_steps // num_steps
            current_step = jnp.minimum((t * num_envs) // timesteps_per_step, num_steps - 1)
            return current_step / (num_steps - 1)

        return jax.lax.cond(num_steps == 1, single_step, multi_step, operand=None)

    def update(self,
               env: Any,
               model: Union[MjModel, Model],
               data: Union[MjData, Data],
               carry: Any,
               backend: ModuleType) -> Tuple[Union[MjModel, Model], Union[MjData, Data], Any]:
        """
        Update the randomizer by applying the state changes to the model.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjModel, Model], Union[MjData, Data], Any]: The updated simulation model, data, and carry.

        """

        assert_backend_is_supported(backend)

        domrand_state = carry.domain_randomizer_state

        sampled_base_mass_multiplier = domrand_state.link_mass_multipliers[0]
        sampled_other_bodies_mass_multipliers = domrand_state.link_mass_multipliers[1:]

        root_body_id = self._root_body_id
        other_body_masks = self._other_body_masks

        # increment the curriculum coefficient
        new_curriculum_coefficient = domrand_state.curriculum_coefficient + (1 / self.rand_conf["total_timesteps"])
        domrand_state = domrand_state.replace(curriculum_coefficient=new_curriculum_coefficient)


        if backend == jnp:
            geom_solref = model.geom_solref.at[:, 0].set(-domrand_state.geom_stiffness)
            geom_solref = geom_solref.at[:, 1].set(-domrand_state.geom_damping)
            body_ipos = model.body_ipos.at[root_body_id].set(model.body_ipos[root_body_id] + domrand_state.com_displacement)
            body_mass = model.body_mass.at[root_body_id].set(model.body_mass[root_body_id] * sampled_base_mass_multiplier)
            body_mass = body_mass.at[root_body_id].set(body_mass[root_body_id] + domrand_state.base_mass_to_add)
            body_mass = body_mass.at[other_body_masks].set(body_mass[other_body_masks] * sampled_other_bodies_mass_multipliers)
            dof_frictionloss = model.dof_frictionloss.at[6:].set(domrand_state.joint_friction_loss)
            dof_damping = model.dof_damping.at[6:].set(domrand_state.joint_damping)
            dof_armature = model.dof_armature.at[6:].set(domrand_state.joint_armature)
            if self.rand_conf["randomize_gravity"]:
                model = self._set_attribute_in_model(model, "opt.gravity", domrand_state.gravity, backend)
        else:
            geom_solref = self._init_geom_solref.copy()
            geom_solref[:, 0] = -domrand_state.geom_stiffness
            geom_solref[:, 1] = -domrand_state.geom_damping
            body_ipos = self._init_body_ipos.copy()
            body_ipos[root_body_id] += domrand_state.com_displacement
            body_mass = self._init_body_mass.copy()
            body_mass[root_body_id] *= sampled_base_mass_multiplier
            body_mass[root_body_id] += domrand_state.base_mass_to_add
            body_mass[other_body_masks] *= sampled_other_bodies_mass_multipliers
            dof_frictionloss = self._init_dof_frictionloss.copy()
            dof_frictionloss[6:] = domrand_state.joint_friction_loss
            dof_damping = self._init_dof_damping.copy()
            dof_damping[6:] = domrand_state.joint_damping
            dof_armature = self._init_dof_armature.copy()
            dof_armature[6:] = domrand_state.joint_armature
            if self.rand_conf["randomize_gravity"]:
                model.opt.gravity = domrand_state.gravity

        if (self.rand_conf["randomize_model_geom_friction_tangential"] 
                or self.rand_conf["randomize_model_geom_friction_torsional"]
                or self.rand_conf["randomize_model_geom_friction_rolling"] 
                or self.rand_conf["randomize_floor_geom_friction_tangential"] 
                or self.rand_conf["randomize_floor_geom_friction_torsional"]
                or self.rand_conf["randomize_floor_geom_friction_rolling"]):
            model = self._set_attribute_in_model(model, "geom_friction", domrand_state.geom_friction, backend)
        if self.rand_conf["randomize_geom_damping"] or self.rand_conf["randomize_geom_stiffness"]:
            model = self._set_attribute_in_model(model, "geom_solref", geom_solref, backend)
        if self.rand_conf["randomize_com_displacement"]:
            model = self._set_attribute_in_model(model, "body_ipos", body_ipos, backend)
        if self.rand_conf["randomize_link_mass"] or self.rand_conf["randomize_base_mass"]:
            model = self._set_attribute_in_model(model, "body_mass", body_mass, backend)
        if self.rand_conf["randomize_joint_friction_loss"]:
            model = self._set_attribute_in_model(model, "dof_frictionloss", dof_frictionloss, backend)
        if self.rand_conf["randomize_joint_damping"]:
            model = self._set_attribute_in_model(model, "dof_damping", dof_damping ,backend)
        if self.rand_conf["randomize_joint_armature"]:
            model = self._set_attribute_in_model(model, "dof_armature", dof_armature, backend)


        # robot kicking
        # if robot kicking has been enabled, edit the qvel of the root body with some probability
        if self.rand_conf["kick_robots"]:
            # get the qvel of the root body
            qvel = data.qvel.copy()

            # get the randomization parameters
            # print(domrand_state.curriculum_coefficient)
            kick_max_vel = self.rand_conf["kick_max_vel"] #* domrand_state.curriculum_coefficient
            kick_prob = self.rand_conf["kick_prob"]
            kick_at_reset = self.rand_conf["kick_at_reset"]
            kick_min_vel = self.rand_conf["kick_min_vel"]

            num_curriculum_steps = self.rand_conf["num_curriculum_steps"]
            num_environments = self.rand_conf["num_environments"]
            num_total_timesteps = self.rand_conf["num_total_timesteps"]

            if backend == jnp:
                key = carry.key
                key, _k = jax.random.split(key)
                random_value = jax.random.uniform(_k, shape=(1,))
                carry = carry.replace(key=key)
                
                curriculum_coeff = self._compute_curriculum_coeff(
                    t=carry.total_timestep,
                    num_steps=num_curriculum_steps,
                    total_steps=num_total_timesteps,
                    num_envs=num_environments
                )

                key, _k2 = jax.random.split(key)
                random_kick_vel = jax.random.uniform(_k2, shape=(3,)) * 2 - 1 # random kick velocity in [-1, 1]^3
                random_kick_vel = random_kick_vel * kick_max_vel
                clamped_kick_magnitude = jnp.sign(random_kick_vel) * jnp.clip(
                    jnp.abs(random_kick_vel),
                    a_min=kick_min_vel,
                    a_max=kick_max_vel
                )
                
                clamped_kick_magnitude = clamped_kick_magnitude * curriculum_coeff.reshape(())

                going_to_kick = random_value[0] < kick_prob
                # also, if kick_at_reset is true, then kick the robot at reset
                going_to_kick = jax.lax.cond(
                    kick_at_reset,
                    lambda _: jax.numpy.logical_or(going_to_kick, carry.cur_step_in_episode == 1),
                    lambda _: going_to_kick,
                    operand=None
                )
                qvel = jax.lax.cond(
                    going_to_kick,
                    lambda _: qvel.at[0:3].add(clamped_kick_magnitude),
                    lambda _: qvel,
                    operand=None
                )
                # Create a new data instance with updated qvel
                data = data.replace(qvel=qvel)
                carry = carry.replace(key=key)

                kicked = going_to_kick
                domrand_state = domrand_state.replace(kicked=kicked)
            else:
                going_to_kick = random_value[0] < kick_prob
                # also, if kick_at_reset is true, then kick the robot at reset
                if kick_at_reset:
                    going_to_kick = going_to_kick or (carry.cur_step_in_episode == 1)

                random_value = np.random.uniform(size=(1,))
                if going_to_kick:
                    random_kick_vel = np.random.uniform(-1, 1, size=(3,)) * kick_max_vel
                    clamped_kick_magnitude = np.sign(random_kick_vel) * np.clip(
                        np.abs(random_kick_vel),
                        a_min=kick_min_vel,
                        a_max=kick_max_vel
                    )
                    clamped_kick_magnitude *= domrand_state.curriculum_coefficient

                    qvel[0:3] += clamped_kick_magnitude

                kicked = going_to_kick
                domrand_state = domrand_state.replace(kicked=kicked)


        if self.rand_conf["push_robots"]:
            # Pushing
            # push_robots: true
            # push_force_amplitude: 20 # Newtons
            # push_torque_amplitude: 2.0 # Newton-meters
            # push_active_duration: 1.0 # seconds
            # push_relief_duration: 3.0 # seconds
            # Rely on carry.cur_step_in_episode

            push_prob = self.rand_conf["push_prob"]
            push_force_amplitude = self.rand_conf["push_force_amplitude"]
            push_active_duration = backend.array(self.rand_conf["push_active_duration"] / env.dt, dtype=backend.int32)
            push_relief_duration = backend.array(self.rand_conf["push_relief_duration"] / env.dt, dtype=backend.int32)
            push_torque_amplitude = self.rand_conf["push_torque_amplitude"]

            is_push_active = (domrand_state.push_counter < push_active_duration)
            domrand_state = domrand_state.replace(push_counter=domrand_state.push_counter + 1)

            # if push is inactive, and a push is sampled, then randomly generate a push force (cartesian and torque), and save it in domrand state
            # if push is active, then use the saved push force from domrand state. so just increment the push counter
            if backend == jnp:
                key = carry.key
                key, _k = jax.random.split(key)
                random_value = jax.random.uniform(_k, shape=(1,))
                carry = carry.replace(key=key)
                
                def inactive_push_branch(_):
                    # Generate random push force
                    random_push_force = jax.random.uniform(_k, shape=(6,), minval=-1.0, maxval=1.0)
                    random_push_force = random_push_force * backend.array([push_force_amplitude, push_force_amplitude, push_force_amplitude,
                                                                            push_torque_amplitude, push_torque_amplitude, push_torque_amplitude])
                    
                    def apply_push(_):
                        new_state = domrand_state.replace(active_root_push_force=random_push_force)
                        return new_state.replace(push_counter=0)  # reset push counter
                    
                    def no_push(_):
                        return domrand_state.replace(active_root_push_force=backend.zeros_like(domrand_state.active_root_push_force))
                    
                    return jax.lax.cond(random_value[0] < push_prob, apply_push, no_push, operand=None)
                
                def active_push_branch(_):
                    return domrand_state
                
                domrand_state = jax.lax.cond(~is_push_active, inactive_push_branch, active_push_branch, operand=None)

            else:
                if not is_push_active:
                    # sample a push force with a random prob
                    random_value = np.random.uniform(size=(1,))
                    if random_value[0] < push_prob:
                        random_push_force = np.random.uniform(-1.0, 1.0, size=(6,))
                        random_push_force = random_push_force * backend.array([push_force_amplitude, push_force_amplitude, push_force_amplitude,
                                                                            push_torque_amplitude, push_torque_amplitude, push_torque_amplitude])
                        domrand_state = domrand_state.replace(active_root_push_force=random_push_force)
                        domrand_state = domrand_state.replace(push_counter=0)  # reset push counter
                else:
                    domrand_state = domrand_state.replace(active_root_push_force=backend.zeros_like(domrand_state.active_root_push_force))

            push_force = domrand_state.active_root_push_force[:3]
            push_torque = domrand_state.active_root_push_force[3:]

            if backend == jnp:
                # Apply the push force to the root body (body index 0)
                data = data.replace(xfrc_applied=data.xfrc_applied.at[self._root_body_id, :3].set(push_force))
                data = data.replace(xfrc_applied=data.xfrc_applied.at[self._root_body_id, 3:].set(push_torque))                
            else:
                # Apply the push force to the root body (body index 0)
                data.xfrc_applied[self._root_body_id, :3] = push_force
                data.xfrc_applied[self._root_body_id, 3:] = push_torque


        carry = carry.replace(domain_randomizer_state=domrand_state)

        return model, data, carry

    def update_observation(self,
                           env: Any,
                           obs: Union[np.ndarray, jnp.ndarray],
                           model: Union[MjModel, Model],
                           data: Union[MjData, Data],
                           carry: Any,
                           backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Update the observation with randomization effects.

        Args:
            env (Any): The environment instance.
            obs (Union[np.ndarray, jnp.ndarray]): The observation to be updated.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The updated observation and carry.

        """

        assert_backend_is_supported(backend)

        # get indices of all the observation components
        total_len_noise_vec = 0
        ind_of_all_joint_pos = env._obs_indices.JointPosArray
        ind_of_all_joint_pos_noise = np.arange(total_len_noise_vec, total_len_noise_vec+len(ind_of_all_joint_pos))
        total_len_noise_vec += len(ind_of_all_joint_pos)
        ind_of_all_joint_vel = env._obs_indices.JointVelArray
        ind_of_all_joint_vel_noise = np.arange(total_len_noise_vec, total_len_noise_vec+len(ind_of_all_joint_vel))
        total_len_noise_vec += len(ind_of_all_joint_vel)
        ind_of_gravity_vec = env._obs_indices.ProjectedGravityVector
        ind_of_gravity_vec_noise = np.arange(total_len_noise_vec, total_len_noise_vec+len(ind_of_gravity_vec))
        total_len_noise_vec += len(ind_of_gravity_vec)
        ind_of_lin_vel = env._obs_indices.FreeJointVel[:3]
        ind_of_lin_vel_noise = np.arange(total_len_noise_vec, total_len_noise_vec+len(ind_of_lin_vel))
        total_len_noise_vec += len(ind_of_lin_vel)
        ind_of_ang_vel = env._obs_indices.FreeJointVel[3:]
        ind_of_ang_vel_noise = np.arange(total_len_noise_vec, total_len_noise_vec+len(ind_of_ang_vel))
        total_len_noise_vec += len(ind_of_ang_vel)
        ind_of_policy_angvel_entries = env._obs_indices.EntryFromFreeJointVel
        ind_of_policy_angvel_entries_noise = np.arange(total_len_noise_vec, total_len_noise_vec+len(ind_of_policy_angvel_entries))
        total_len_noise_vec += len(ind_of_policy_angvel_entries)


        print(ind_of_all_joint_pos)
        print(ind_of_all_joint_vel)
        print(ind_of_gravity_vec)
        print(ind_of_lin_vel)
        print(ind_of_ang_vel)
        print(ind_of_policy_angvel_entries)

        print("ALLOED TO BE RANDOMIZED: ", self._allowed_to_be_randomized)


        # get randomization parameters
        joint_pos_noise_scale = self.rand_conf["joint_pos_noise_scale"]
        joint_vel_noise_scale = self.rand_conf["joint_vel_noise_scale"]
        gravity_noise_scale = self.rand_conf["gravity_noise_scale"]
        lin_vel_noise_scale = self.rand_conf["lin_vel_noise_scale"]
        ang_vel_noise_scale = self.rand_conf["ang_vel_noise_scale"]

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            noise = jax.random.normal(_k, shape=(total_len_noise_vec,))

            randomized_obs = obs.copy()

            norm_obs = self.rand_conf["normalize_observations"]
            norm_factors = self.rand_conf["norm_factors"]

            # Add noise to joint positions
            if self.rand_conf["add_joint_pos_noise"]:
                randomized_obs = randomized_obs.at[ind_of_all_joint_pos].add(noise[ind_of_all_joint_pos_noise] * joint_pos_noise_scale)
            
            # Add noise to joint velocities
            if self.rand_conf["add_joint_vel_noise"]:
                randomized_obs = randomized_obs.at[ind_of_all_joint_vel].add(noise[ind_of_all_joint_vel_noise] * joint_vel_noise_scale)
            
            # Add noise to gravity vector
            if self.rand_conf["add_gravity_noise"]:
                randomized_obs = randomized_obs.at[ind_of_gravity_vec].add(noise[ind_of_gravity_vec_noise] * gravity_noise_scale)
            
            # Add noise to linear velocities
            if self.rand_conf["add_free_joint_lin_vel_noise"]:
                randomized_obs = randomized_obs.at[ind_of_lin_vel].add(noise[ind_of_lin_vel_noise] * lin_vel_noise_scale)

            # Add noise to angular velocities
            if self.rand_conf["add_free_joint_ang_vel_noise"]:
                randomized_obs = randomized_obs.at[ind_of_ang_vel].add(noise[ind_of_ang_vel_noise] * ang_vel_noise_scale)

            # Add noise to policy angular velocity entries
            if self.rand_conf["add_policy_ang_vel_noise"]:
                randomized_obs = randomized_obs.at[ind_of_policy_angvel_entries].add(noise[ind_of_policy_angvel_entries_noise] * ang_vel_noise_scale)
                

            if norm_obs:
                # print("asked to normalize observations")
                # print(norm_factors)
                randomized_obs = randomized_obs.at[ind_of_all_joint_pos].multiply(norm_factors["joint_pos"])
                # cast to numpy to print

                # # print("randomized_obs at joint pos: ", np.array(randomized_obs[ind_of_all_joint_vel]))
                # jax.debug.print("ONE randomized_obs at joint pos: {r}", r=randomized_obs[ind_of_all_joint_vel])
                randomized_obs = randomized_obs.at[ind_of_all_joint_vel].multiply(norm_factors["joint_vel"])
                # print("randomized_obs at joint pos: ", np.array(randomized_obs[ind_of_all_joint_vel]))
                # jax.debug.print("TWO randomized_obs at joint pos: {r}", r=randomized_obs[ind_of_all_joint_vel])
                # print('----------------')
                randomized_obs = randomized_obs.at[ind_of_gravity_vec].multiply(norm_factors["gravity"])
                randomized_obs = randomized_obs.at[ind_of_lin_vel].multiply(norm_factors["lin_vel"])
                randomized_obs = randomized_obs.at[ind_of_ang_vel].multiply(norm_factors["ang_vel"])

            obs = obs.at[self._allowed_to_be_randomized].set(randomized_obs[self._allowed_to_be_randomized])
            carry = carry.replace(key=key)

        else:
            noise = np.random.normal(size=(total_len_noise_vec,))

            randomized_obs = obs.copy()

            norm_obs = self.rand_conf["normalize_observations"]
            norm_factors = self.rand_conf["norm_factors"]

             # Add noise to joint positions
            if self.rand_conf["add_joint_pos_noise"]:
                randomized_obs[ind_of_all_joint_pos] += noise[ind_of_all_joint_pos_noise] * joint_pos_noise_scale

            # Add noise to joint velocities
            if self.rand_conf["add_joint_vel_noise"]:
                randomized_obs[ind_of_all_joint_vel] += noise[ind_of_all_joint_vel_noise] * joint_vel_noise_scale

            # Add noise to gravity vector
            if self.rand_conf["add_gravity_noise"]:
                randomized_obs[ind_of_gravity_vec] += noise[ind_of_gravity_vec_noise] * gravity_noise_scale

            # Add noise to linear velocities
            if self.rand_conf["add_free_joint_lin_vel_noise"]:
                randomized_obs[ind_of_lin_vel] += noise[ind_of_lin_vel_noise] * lin_vel_noise_scale

            # Add noise to angular velocities
            if self.rand_conf["add_free_joint_ang_vel_noise"]:
                randomized_obs[ind_of_ang_vel] += noise[ind_of_ang_vel_noise] * ang_vel_noise_scale

            if norm_obs:
                randomized_obs[ind_of_all_joint_pos] *= norm_factors["joint_pos"]
                randomized_obs[ind_of_all_joint_vel] *= norm_factors["joint_vel"]
                randomized_obs[ind_of_gravity_vec] *= norm_factors["gravity"]
                randomized_obs[ind_of_lin_vel] *= norm_factors["lin_vel"]
                randomized_obs[ind_of_ang_vel] *= norm_factors["ang_vel"]

            obs[self._allowed_to_be_randomized] = randomized_obs[self._allowed_to_be_randomized]

        return obs, carry

    def update_action(self,
                      env: Any,
                      action: Union[np.ndarray, jnp.ndarray],
                      model: Union[MjModel, Model],
                      data: Union[MjData, Data],
                      carry: Any,
                      backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Update the action with randomization effects.

        Args:
            env (Any): The environment instance.
            action (Union[np.ndarray, jnp.ndarray]): The action to be updated.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The updated action and carry.

        """

        assert_backend_is_supported(backend)

        alpha = carry.domain_randomizer_state.filter_alpha
        is_first = (carry.cur_step_in_episode == 1)

        if backend == jnp:
            action = jnp.asarray(action)
            prev = jnp.asarray(carry.domain_randomizer_state.prev_filtered_action)

            filtered_action = jax.lax.select(
                is_first,
                action,
                alpha * prev + (1.0 - alpha) * action
            )

            # Update carry with new filtered action
            domain_randomizer_state = carry.domain_randomizer_state.replace(
                prev_filtered_action=filtered_action
            )
            carry = carry.replace(domain_randomizer_state=domain_randomizer_state)

        else:
            action = np.asarray(action)
            prev = np.asarray(carry.domain_randomizer_state.prev_filtered_action)

            if is_first:
                filtered_action = action
            else:
                filtered_action = alpha * prev + (1.0 - alpha) * action

            # Update carry with new filtered action
            domain_randomizer_state=carry.domain_randomizer_state.replace(
                    prev_filtered_action=filtered_action
            )
            carry = carry.replace(domain_randomizer_state=domain_randomizer_state)


        return filtered_action, carry

    def _sample_geom_friction(self, 
                            model: Union[MjModel, Model],
                            carry: Any,
                            backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples geometry friction parameters, handling floor and non-floor geometries separately.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information (e.g., JAX key).
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: Randomized geometry friction parameters and carry.
        """
        assert_backend_is_supported(backend)

        floor_id = self._floor_geom_id

        ngeom = len(model.geom_friction)
        floor_friction = model.geom_friction[floor_id]

        any_model_flags = (
            self.rand_conf["randomize_model_geom_friction_tangential"]
            or self.rand_conf["randomize_model_geom_friction_torsional"]
            or self.rand_conf["randomize_model_geom_friction_rolling"]
        )
        any_floor_flags = (
            self.rand_conf["randomize_floor_geom_friction_tangential"]
            or self.rand_conf["randomize_floor_geom_friction_torsional"]
            or self.rand_conf["randomize_floor_geom_friction_rolling"]
        )

        if backend == jnp:
            key = carry.key
            if any_model_flags:
                key, subkey = jax.random.split(key)
                interp_model = jax.random.uniform(subkey, shape=(ngeom,))
            else:
                interp_model = backend.zeros(ngeom)

            if any_floor_flags:
                key, subkey = jax.random.split(key)
                interp_floor = jax.random.uniform(subkey)
            else:
                interp_floor = backend.zeros(())
            carry = carry.replace(key=key)
        else:
            interp_model = np.random.uniform(size=(ngeom,)) if any_model_flags else np.zeros(ngeom)
            interp_floor = np.random.uniform() if any_floor_flags else 0.0

        # Start from original values
        geom_friction = backend.array(model.geom_friction)

        # Sample non-floor friction if needed
        if any_model_flags:
            tan = model.geom_friction[:, 0]
            tor = model.geom_friction[:, 1]
            roll = model.geom_friction[:, 2]

            if self.rand_conf["randomize_model_geom_friction_tangential"]:
                min_, max_ = self.rand_conf["model_geom_friction_tangential_range"]
                tan = min_ + (max_ - min_) * interp_model

            if self.rand_conf["randomize_model_geom_friction_torsional"]:
                min_, max_ = self.rand_conf["model_geom_friction_torsional_range"]
                tor = min_ + (max_ - min_) * interp_model

            if self.rand_conf["randomize_model_geom_friction_rolling"]:
                min_, max_ = self.rand_conf["model_geom_friction_rolling_range"]
                roll = min_ + (max_ - min_) * interp_model

            updated = backend.stack([tan, tor, roll], axis=1)

            # Leave floor friction untouched
            if backend == jnp:
                geom_friction = geom_friction.at[:].set(
                    backend.where(
                        backend.arange(ngeom)[:, None] != floor_id,
                        updated,
                        geom_friction
                    )
                )
            else:
                geom_friction[np.arange(ngeom) != floor_id] = updated[np.arange(ngeom) != floor_id]

        # Sample floor friction if needed
        if any_floor_flags:
            tan = floor_friction[0]
            tor = floor_friction[1]
            roll = floor_friction[2]

            if self.rand_conf["randomize_floor_geom_friction_tangential"]:
                min_, max_ = self.rand_conf["floor_geom_friction_tangential_range"]
                tan = min_ + (max_ - min_) * interp_floor

            if self.rand_conf["randomize_floor_geom_friction_torsional"]:
                min_, max_ = self.rand_conf["floor_geom_friction_torsional_range"]
                tor = min_ + (max_ - min_) * interp_floor

            if self.rand_conf["randomize_floor_geom_friction_rolling"]:
                min_, max_ = self.rand_conf["floor_geom_friction_rolling_range"]
                roll = min_ + (max_ - min_) * interp_floor

            new_floor_friction = backend.array([tan, tor, roll])

            if backend == jnp:
                geom_friction = geom_friction.at[floor_id].set(new_floor_friction)
            else:
                geom_friction[floor_id] = new_floor_friction

        return geom_friction, carry

    def _sample_geom_damping_and_stiffness(self,
                                           model: Union[MjModel, Model],
                                           carry: Any,
                                           backend: ModuleType) \
            -> Tuple[Union[np.ndarray, jnp.ndarray], Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples the geometry damping and stiffness parameters.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Union[np.ndarray, jnp.ndarray], Any]: The randomized geometry damping
            and stiffness parameters and carry.

        """

        assert_backend_is_supported(backend)

        damping_min, damping_max = self.rand_conf["geom_damping_range"]
        n_geoms = model.ngeom
        stiffness_min, stiffness_max = self.rand_conf["geom_stiffness_range"]

        if backend == jnp:
            key = carry.key
            key, _k_damp, _k_stiff = jax.random.split(key, 3)
            interpolation_damping = jax.random.uniform(_k_damp, shape=(n_geoms,))
            interpolation_stiff = jax.random.uniform(_k_stiff, shape=(n_geoms,))
            carry = carry.replace(key=key)
        else:
            interpolation_damping = np.random.uniform(size=(n_geoms,))
            interpolation_stiff = np.random.uniform(size=(n_geoms,))

        sampled_damping = (
            damping_min + (damping_max - damping_min) * interpolation_damping
            if self.rand_conf["randomize_geom_damping"]
            else model.geom_solref[:, 1]
        )
        sampled_stiffness = (
            stiffness_min + (stiffness_max - stiffness_min) * interpolation_stiff
            if self.rand_conf["randomize_geom_stiffness"]
            else model.geom_solref[:, 0]
        )

        return sampled_damping, sampled_stiffness, carry
    
    def _sample_joint_friction_loss(self,
                                    model: Union[MjModel, Model],
                                    carry: Any,
                                    backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples the joint friction loss parameters.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The randomized joint friction loss parameters.

        """

        assert_backend_is_supported(backend)

        friction_min, friction_max = self.rand_conf["joint_friction_loss_range"]
        n_dofs = model.nv - 6 #exclude freejoint 6 degrees of freedom

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=(n_dofs,))
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform(size=(n_dofs,))

        sampled_friction_loss = (
            friction_min + (friction_max - friction_min) * interpolation
            if self.rand_conf["randomize_joint_friction_loss"]
            else model.dof_frictionloss[6:]
        )

        return sampled_friction_loss, carry
    
    def _sample_joint_damping(self, model: Union[MjModel, Model],
                              carry: Any,
                              backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples the joint damping parameters.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The randomized joint damping and carry.

        """

        assert_backend_is_supported(backend)

        damping_min, damping_max = self.rand_conf["joint_damping_range"]
        n_dofs = model.nv - 6 #exclude freejoint 6 degrees of freedom

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=(n_dofs,))
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform(size=(n_dofs,))

        sampled_damping = (
            damping_min + (damping_max - damping_min) * interpolation
            if self.rand_conf["randomize_joint_damping"]
            else model.dof_damping[6:]
        )

        return sampled_damping, carry
    
    def _sample_joint_armature(self,
                               model: Union[MjModel, Model],
                               carry: Any,
                               backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples the joint armature parameters.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The randomized joint aramture paramters and carry.

        """

        assert_backend_is_supported(backend)

        armature_min, armature_max = self.rand_conf["joint_armature_range"]
        n_dofs = model.nv - 6 #exclude freejoint 6 degrees of freedom

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=(n_dofs,))
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform(size=(n_dofs,))

        sampled_armature = (
            armature_min + (armature_max - armature_min) * interpolation
            if self.rand_conf["randomize_joint_armature"]
            else model.dof_armature[6:]
        )

        return sampled_armature, carry

    def _sample_gravity(self,
                        model: Union[MjModel, Model],
                        carry: Any, backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples the gravity vector.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The randomized gravity vector and carry.
        """

        assert_backend_is_supported(backend)

        gravity_min, gravity_max = self.rand_conf["gravity_range"]
        randomize_gravity = self.rand_conf.get("randomize_gravity", False)
        randomize_gravity_direction = self.rand_conf.get("randomize_gravity_direction", False)
        alpha_deg = self.rand_conf.get("gravity_cone_angle_deg", 5.0)  
        alpha_rad = backend.deg2rad(alpha_deg)

        if backend == jnp:
            key = carry.key
            key, k1, k2, k3 = jax.random.split(key, 4)

            interp = jax.random.uniform(k1)
            sampled_mag = gravity_min + (gravity_max - gravity_min) * interp if randomize_gravity else -model.opt.gravity[2]

            if randomize_gravity_direction:
                # Sample angle within cone
                cos_alpha = jnp.cos(alpha_rad)
                z = jax.random.uniform(k2, minval=cos_alpha, maxval=1.0)
                theta = jnp.arccos(z)
                phi = jax.random.uniform(k3, minval=0.0, maxval=2 * jnp.pi)

                x = jnp.sin(theta) * jnp.cos(phi)
                y = jnp.sin(theta) * jnp.sin(phi)
                z = -jnp.cos(theta)

                direction = jnp.array([x, y, z])
                gravity = direction * sampled_mag
            else:
                gravity = jnp.array([0.0, 0.0, -sampled_mag])

            carry = carry.replace(key=key)

        else:
            interp = np.random.uniform()
            sampled_mag = gravity_min + (gravity_max - gravity_min) * interp if randomize_gravity else -model.opt.gravity[2]

            if randomize_gravity_direction:
                cos_alpha = np.cos(alpha_rad)
                z = np.random.uniform(cos_alpha, 1.0)
                theta = np.arccos(z)
                phi = np.random.uniform(0.0, 2 * np.pi)

                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = -np.cos(theta)

                direction = np.array([x, y, z])
                gravity = direction * sampled_mag
            else:
                gravity = np.array([0.0, 0.0, -sampled_mag])

        return backend.array(gravity), carry

    def _sample_base_mass(self,
                          model: Union[MjModel, Model],
                          carry: Any, backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
         Samples a base mass to add to the robot.

         Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The randomized gravity vector and carry.

        """

        assert_backend_is_supported(backend)

        base_mass_min, base_mass_max = self.rand_conf["base_mass_to_add_range"]

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k)
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform()

        sampled_base_mass = (
            base_mass_min + (base_mass_max - base_mass_min) * interpolation
            if self.rand_conf["randomize_base_mass"]
            else 0.0)

        return sampled_base_mass, carry

    def _sample_com_displacement(self,
                                 model: Union[MjModel, Model],
                                 carry: Any,
                                 backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples a center-of-mass (COM) displace.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The randomized COM displacement vector and carry.

        """
        assert_backend_is_supported(backend)

        displ_min, displ_max = self.rand_conf["com_displacement_range"]

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=(3,))
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform(size=3)

        sampled_com_displacement = (
            displ_min + (displ_max - displ_min) * interpolation
            if self.rand_conf["randomize_com_displacement"]
            else backend.array([0.0, 0.0, 0.0])
        )

        return sampled_com_displacement, carry
    
    def _sample_link_mass_multipliers(self,
                                      model: Union[MjModel, Model],
                                      carry: Any,
                                      backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples the link mass multipliers.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The randomized link mass multipliers and carry.

        """

        assert_backend_is_supported(backend)

        multiplier_dict = self.rand_conf["link_mass_multiplier_range"]

        mult_base_min, mult_base_max = multiplier_dict["root_body"]
        mult_other_min, mult_other_max = multiplier_dict["other_bodies"]

        # print(mult_base_min.shape)
        # print(mult_other_min.shape)
        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=(model.nbody-1,)) #exclude worldbody 
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform(size=model.nbody-1)

        sampled_base_mass_multiplier = (
            mult_base_min + (mult_base_max - mult_base_min) * interpolation[0].reshape(1)
            if self.rand_conf["randomize_link_mass"]
            else backend.array([1.0])
        )

        # sampled_base_mass_multiplier = jnp.expand_dims(sampled_base_mass_multiplier, axis=0)

        sampled_other_bodies_mass_multipliers = (
            mult_other_min + (mult_other_max - mult_other_min) * interpolation[1:]
            if self.rand_conf["randomize_link_mass"]
            else backend.array([1.0]*(model.nbody-2))
        )

        # sampled_other_bodies_mass_multipliers = jnp.expand_dims(sampled_other_bodies_mass_multipliers, axis=0)

        mass_multipliers = backend.concatenate([
            sampled_base_mass_multiplier,
            sampled_other_bodies_mass_multipliers,
        ])

        return mass_multipliers, carry
    
    def _sample_p_gains_noise(self,
                              env, model: Union[MjModel, Model],
                              carry: Any,
                              backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples p_gains_noise for the PDControl control function.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The p_gains_noise and carry.
        """
        assert_backend_is_supported(backend)

        init_p_gain = env._control_func._init_p_gain

        noise_shape = (len(init_p_gain),) if init_p_gain.size > 1 else (1,)

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=noise_shape, minval=-1.0, maxval=1.0)
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform(size=noise_shape, low=-1.0, high=1.0)

        p_noise_scale = self.rand_conf["p_gains_noise_scale"]

        p_noise = (
            interpolation * (p_noise_scale * init_p_gain) 
            if self.rand_conf["add_p_gains_noise"]
            else backend.array([0.0]*len(init_p_gain))
            )

        return p_noise, carry
    
    def _sample_d_gains_noise(self,
                              env, model: Union[MjModel, Model],
                              carry: Any,
                              backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples d_gains_noise for the PDControl control function..

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The d_gains_noise and carry.

        """

        assert_backend_is_supported(backend)

        init_d_gain = env._control_func._init_d_gain

        noise_shape = (len(init_d_gain),) if init_d_gain.size > 1 else (1,)

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=noise_shape, minval=-1.0, maxval=1.0)
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform(size=noise_shape, low=-1.0, high=1.0)

        d_noise_scale = self.rand_conf["d_gains_noise_scale"]

        d_noise = (
            interpolation * (d_noise_scale * init_d_gain) 
            if self.rand_conf["add_d_gains_noise"]
            else backend.array([0.0]*len(init_d_gain))
            )

        return d_noise, carry