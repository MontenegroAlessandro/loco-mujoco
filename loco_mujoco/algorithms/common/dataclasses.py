import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.training import train_state
from typing import Any, Optional, Tuple, NamedTuple, Dict
from loco_mujoco.environments.base import TrajState
from loco_mujoco.core.wrappers.mjx import Metrics
from dataclasses import dataclass, field


class Transition(NamedTuple):
    done: jnp.ndarray
    absorbing: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    traj_state: TrajState
    metrics: Metrics


class MetricHandlerTransition(NamedTuple):
    env_state: Any
    logged_metrics: Metrics


@struct.dataclass
class AdaptiveLRState:
    learning_rate: jnp.ndarray

class TrainState(train_state.TrainState):
    run_stats: Any
    adaptive_lr_state: Optional[AdaptiveLRState] = None

@struct.dataclass
class TrainStateBuffer:
    train_states: TrainState
    n: int
    size: int   # buffer size

    @classmethod
    def create(cls, train_state: TrainState, size: int):
        return TrainStateBuffer(
            train_states=jax.tree.map(lambda x: jnp.stack([x] * size), train_state),
            n=0,
            size=size
        )

    @classmethod
    def add(cls, train_state_buffer, train_state: TrainState):
        index = train_state_buffer.n
        # Add the new train state at index n
        train_states_updated = jax.tree.map(
            lambda buffer, new: buffer.at[index].set(new),
            train_state_buffer.train_states,
            train_state
        )
        return train_state_buffer.replace(
            train_states=train_states_updated,
            n=index + 1,
        )


@struct.dataclass
class BestTrainStates:
    train_states: TrainState
    metrics: jnp.array
    iterations: jnp.array
    cur_worst_perf: float
    step: int
    n: int
    size: int

    @classmethod
    def create(cls, train_state: TrainState, n: int):
        return BestTrainStates(
            train_states=jax.tree.map(lambda x: jnp.stack([x] * n), train_state),
            metrics=jnp.full((n,), -jnp.inf),
            iterations=jnp.zeros((n,)),
            cur_worst_perf=-jnp.inf,
            n=n,
            size=0
        )
    
@dataclass
class ReplayBuffer:
    """
    [AM] Replay buffer for TD3 agent. Done in numpy for efficiency reasons. 
    """
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray
    ptr: int    # pointer to the current index
    size: int   # size of the replay buffer

@dataclass
class PhasedExplorationSchedule:
    """[AM] Implementing PES noise scheduler."""
    phases: int
    noise_max: float
    noise_min: float
    smooth: float
    linear: bool

    @classmethod
    def create(cls, phases: int, noise_max: float, noise_min: float, linear: bool = True):
        if linear:
            smooth = float((noise_min - noise_max) / phases)
        else:
            smooth = jnp.log(noise_max / noise_min) / jnp.log(phases)
        return cls(
            phases=phases,
            noise_max=noise_max,
            noise_min=noise_min,
            smooth=smooth,
            linear=linear
        )
    
    def update_sigma(self, current_phase: int):
        if self.linear:
            new_sigma = self.noise_max + self.smooth * current_phase
        else:
            new_sigma = self.noise_max * jnp.power(current_phase, -self.smooth)
        return new_sigma
    
@dataclass
class DecoupledReplayBuffer:
    """
    [AM] Replay buffer for Fast TD3 agent, modified to handle parallel environments separately.
    Data is stored in numpy for efficiency.
    """
    total_capacity: int
    num_envs: int
    obs_shape: Tuple[int, ...]
    action_shape: Tuple[int, ...]
    
    # Internal buffers will be initialized in __post_init__
    obs: np.ndarray = field(init=False)
    actions: np.ndarray = field(init=False)
    rewards: np.ndarray = field(init=False)
    next_obs: np.ndarray = field(init=False)
    dones: np.ndarray = field(init=False)
    
    # Pointers and sizes for each environment's buffer
    ptrs: np.ndarray = field(init=False)
    sizes: np.ndarray = field(init=False)
    per_env_capacity: int = field(init=False)

    def __post_init__(self):
        """Initializes the buffer arrays and tracking variables."""
        if self.total_capacity % self.num_envs != 0:
            raise ValueError("total_capacity must be divisible by num_envs")
            
        self.per_env_capacity = self.total_capacity // self.num_envs
        
        # Create buffers with a leading dimension for the number of environments
        self.obs = np.zeros((self.num_envs, self.per_env_capacity, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.num_envs, self.per_env_capacity, *self.action_shape), dtype=np.float32)
        self.rewards = np.zeros((self.num_envs, self.per_env_capacity), dtype=np.float32)
        self.next_obs = np.zeros((self.num_envs, self.per_env_capacity, *self.obs_shape), dtype=np.float32)
        self.dones = np.zeros((self.num_envs, self.per_env_capacity), dtype=np.float32)
        
        self.ptrs = np.zeros(self.num_envs, dtype=np.int32)
        self.sizes = np.zeros(self.num_envs, dtype=np.int32)

    def add(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray, next_obs: np.ndarray, done: np.ndarray):
        """Adds a batch of transitions from parallel environments to their respective buffers."""
        # The input arrays have shape (num_envs, ...), so we can directly assign them.
        indices = self.ptrs
        
        self.obs[np.arange(self.num_envs), indices] = obs
        self.actions[np.arange(self.num_envs), indices] = action
        self.rewards[np.arange(self.num_envs), indices] = reward
        self.next_obs[np.arange(self.num_envs), indices] = next_obs
        self.dones[np.arange(self.num_envs), indices] = done
        
        # Update pointers and sizes for each environment
        self.ptrs = (self.ptrs + 1) % self.per_env_capacity
        self.sizes = np.minimum(self.sizes + 1, self.per_env_capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Samples a batch of transitions uniformly from all available data."""
        # Choose which environments to sample from, proportionally to their size
        total_transitions = np.sum(self.sizes)
        if total_transitions == 0:
            return {} # Return empty dict if buffer is empty
            
        env_probs = self.sizes / total_transitions
        env_indices = np.random.choice(self.num_envs, size=batch_size, p=env_probs)
        
        # For each chosen environment, pick a random transition
        transition_indices = (np.random.rand(batch_size) * self.sizes[env_indices]).astype(int)
        
        batch = {
            "obs": self.obs[env_indices, transition_indices],
            "actions": self.actions[env_indices, transition_indices],
            "rewards": self.rewards[env_indices, transition_indices],
            "next_obs": self.next_obs[env_indices, transition_indices],
            "dones": self.dones[env_indices, transition_indices],
        }
        return batch

@dataclass
class SuperReplayBuffer:
    """
    Decoupled replay buffer allowing for managing n_step returns.
    """
    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        n_steps: int = 1,
        gamma: float = 0.99,
    ):
        self.n_env = n_env
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.n_steps = n_steps
        self.gamma = gamma

        # data
        self.observations = np.zeros((n_env, buffer_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((n_env, buffer_size, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((n_env, buffer_size), dtype=np.float32)
        self.dones = np.zeros((n_env, buffer_size), dtype=np.int64)
        self.truncations = np.zeros((n_env, buffer_size), dtype=np.int64)
        self.next_observations = np.zeros((n_env, buffer_size, *obs_shape), dtype=np.float32)

        self.ptr = 0
        self.size = 0 

    def add(self, obs, action, reward, next_obs, done, truncation):
        """Add a batch of transitions to the buffer."""
        ptr = self.ptr % self.buffer_size

        self.observations[:, ptr] = obs
        self.actions[:, ptr] = action
        self.rewards[:, ptr] = reward
        self.next_observations[:, ptr] = next_obs
        self.dones[:, ptr] = done
        self.truncations[:, ptr] = truncation

        self.ptr += 1
        self.size = min(self.buffer_size * self.n_env, self.size + self.n_env)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch size transitions, coming from the n_envs tapes."""
        if self.n_steps == 1:
            return self._sample_one_step(batch_size)
        else:
            return self._sample_n_steps(batch_size)

    def _sample_one_step(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample just one step"""
        num_total_transitions = min(self.buffer_size, self.ptr)
        per_env_batch_size = max(1, batch_size // self.n_env)

        env_indices = np.arange(self.n_env)[:, np.newaxis]
        indices = np.random.randint(0, num_total_transitions, size=(self.n_env, per_env_batch_size))

        obs = self.observations[env_indices, indices].reshape(-1, *self.obs_shape)
        actions = self.actions[env_indices, indices].reshape(-1, *self.action_shape)
        rewards = self.rewards[env_indices, indices].reshape(-1)
        next_obs = self.next_observations[env_indices, indices].reshape(-1, *self.obs_shape)
        dones = self.dones[env_indices, indices].reshape(-1)
        truncations = self.truncations[env_indices, indices].reshape(-1)

        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "dones": dones.astype(bool),
            "truncs": truncations.astype(bool),
            "n_steps": np.ones_like(dones),
        }

    def _sample_n_steps(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample transitions and compute n_steps returns."""
        per_env_batch_size = max(1, batch_size // self.n_env)
        
        # staring indices
        if self.ptr >= self.buffer_size: 
            max_start_idx = self.buffer_size
            indices = np.random.randint(0, max_start_idx, size=(self.n_env, per_env_batch_size))
        else: 
            max_start_idx = max(1, self.ptr - self.n_steps + 1)
            indices = np.random.randint(0, max_start_idx, size=(self.n_env, per_env_batch_size))

        # starting transitions
        env_indices = np.arange(self.n_env)[:, np.newaxis]
        obs = self.observations[env_indices, indices].reshape(-1, *self.obs_shape)
        actions = self.actions[env_indices, indices].reshape(-1, *self.action_shape)

        # sequences
        seq_offsets = np.arange(self.n_steps)
        all_indices = (indices[..., np.newaxis] + seq_offsets) % self.buffer_size

        all_rewards = self.rewards[env_indices[..., np.newaxis], all_indices]
        all_dones = self.dones[env_indices[..., np.newaxis], all_indices]
        all_truncations = self.truncations[env_indices[..., np.newaxis], all_indices]

        # mask and n-steps return
        zeros_shape = (self.n_env, per_env_batch_size, 1)
        all_dones_shifted = np.concatenate([np.zeros(zeros_shape), all_dones[..., :-1]], axis=-1)
        done_masks = np.cumprod(1.0 - all_dones_shifted, axis=-1)
        
        discounts = np.power(self.gamma, np.arange(self.n_steps))
        n_step_rewards = np.sum(all_rewards * done_masks * discounts, axis=-1)

        # select the final state (done or truncation)
        terminations = np.logical_or(all_dones, all_truncations)
        first_term_idx = np.argmax(terminations, axis=-1)
        no_term_mask = np.all(terminations == 0, axis=-1)
        final_indices_offset = np.where(no_term_mask, self.n_steps - 1, np.minimum(first_term_idx, self.n_steps - 1))
        
        final_seq_indices = (indices + final_indices_offset) % self.buffer_size
        final_next_obs = self.next_observations[env_indices, final_seq_indices]
        final_dones = self.dones[env_indices, final_seq_indices]
        final_truncations = self.truncations[env_indices, final_seq_indices]
        
        # effective steps
        term_masks = np.cumprod(1.0 - np.concatenate([np.zeros(zeros_shape), terminations[..., :-1]], axis=-1), axis=-1)
        effective_n_steps = np.sum(term_masks, axis=-1)

        return {
            "obs": obs,
            "actions": actions,
            "rewards": n_step_rewards.reshape(-1),
            "next_obs": final_next_obs.reshape(-1, *self.obs_shape),
            "dones": final_dones.reshape(-1).astype(bool),
            "truncs": final_truncations.reshape(-1).astype(bool),
            "n_steps": effective_n_steps.reshape(-1).astype(np.int32),
        }


@struct.dataclass
class RunningMeanStdState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray

    @classmethod
    def create(cls, obs_shape):
        """Initializes the running mean-std state."""
        return cls(
            mean=jnp.zeros(obs_shape),
            var=jnp.ones(obs_shape),
            count=jnp.array(1e-6)
        )

    def update(self, batch: jnp.ndarray):
        """
        Updates the running mean and variance with a new batch of data.
        Uses Welford's algorithm for numerical stability.
        """
        batch_mean = jnp.mean(batch, axis=0)
        batch_var = jnp.var(batch, axis=0)
        batch_count = batch.shape[0]

        updated_count = self.count + batch_count

        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / updated_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * self.count * batch_count / updated_count
        new_var = M2 / updated_count

        return self.replace(mean=new_mean, var=new_var, count=updated_count)

    def normalize(self, x: jnp.ndarray):
        """Normalizes the input data using the current running stats."""
        return (x - self.mean) / jnp.sqrt(self.var + 1e-8)