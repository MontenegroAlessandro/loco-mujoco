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
class DecoupledReplayBufferN:
    """
    Replay buffer with parallel envs, n-step, and bootstrap support.
    Versione finale e corretta che gestisce sia dones che truncations.
    """
    total_capacity: int
    num_envs: int
    obs_shape: Tuple[int, ...]
    action_shape: Tuple[int, ...]
    n_step: int
    gamma: float 

    obs: np.ndarray = field(init=False)
    actions: np.ndarray = field(init=False)
    rewards: np.ndarray = field(init=False)
    dones: np.ndarray = field(init=False)
    truncs: np.ndarray = field(init=False)

    ptrs: np.ndarray = field(init=False)
    sizes: np.ndarray = field(init=False)
    per_env_capacity: int = field(init=False)

    def __post_init__(self):
        if self.total_capacity % self.num_envs != 0:
            raise ValueError("total_capacity must be divisible by num_envs")
        self.per_env_capacity = self.total_capacity // self.num_envs

        self.obs = np.zeros((self.num_envs, self.per_env_capacity, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.num_envs, self.per_env_capacity, *self.action_shape), dtype=np.float32)
        self.rewards = np.zeros((self.num_envs, self.per_env_capacity), dtype=np.float32)
        self.dones = np.zeros((self.num_envs, self.per_env_capacity), dtype=np.float32)
        self.truncs = np.zeros((self.num_envs, self.per_env_capacity), dtype=np.float32)
        
        self.ptrs = np.zeros(self.num_envs, dtype=np.int32)
        self.sizes = np.zeros(self.num_envs, dtype=np.int32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        trunc: np.ndarray,
    ):
        idx = self.ptrs
        env_indices = np.arange(self.num_envs)

        self.obs[env_indices, idx] = obs
        self.actions[env_indices, idx] = action
        self.rewards[env_indices, idx] = reward
        self.dones[env_indices, idx] = done
        self.truncs[env_indices, idx] = trunc

        self.ptrs = (self.ptrs + 1) % self.per_env_capacity
        self.sizes = np.minimum(self.sizes + 1, self.per_env_capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        total_transitions = np.sum(self.sizes)
        if total_transitions < batch_size:
            return {}

        # staring indices
        env_probs = self.sizes / total_transitions
        env_indices = np.random.choice(self.num_envs, size=batch_size, p=env_probs)
        max_start_indices = self.sizes[env_indices] - self.n_step
        valid_mask = max_start_indices >= 0
        if not np.all(valid_mask):
            env_indices = env_indices[valid_mask]
            max_start_indices = max_start_indices[valid_mask]
            if len(env_indices) == 0: return {}
            current_batch_size = len(env_indices)
        else:
            current_batch_size = batch_size
        start_indices = (np.random.rand(current_batch_size) * (max_start_indices + 1)).astype(int)

        # sequences
        step_range = np.arange(self.n_step)
        seq_indices = (start_indices[:, np.newaxis] + step_range) % self.per_env_capacity
        batch_rewards = self.rewards[env_indices[:, np.newaxis], seq_indices]
        batch_dones = self.dones[env_indices[:, np.newaxis], seq_indices]
        batch_truncs = self.truncs[env_indices[:, np.newaxis], seq_indices]
        
        # mask
        terminations = np.logical_or(batch_dones, batch_truncs)
        first_term_idx = np.argmax(terminations, axis=1)
        no_term_mask = np.all(terminations == 0, axis=1)
        effective_n_steps = np.where(no_term_mask, self.n_step, first_term_idx + 1).astype(np.int32)
        done_mask = np.arange(self.n_step) < effective_n_steps[:, np.newaxis]

        # return
        discounts = np.power(self.gamma, step_range)
        n_step_rewards = np.sum(batch_rewards * discounts * done_mask, axis=1)
        
        # final state
        final_step_indices = (start_indices + effective_n_steps) % self.per_env_capacity

        return {
            "obs": self.obs[env_indices, start_indices],
            "actions": self.actions[env_indices, start_indices],
            "rewards": n_step_rewards,
            "next_obs": self.obs[env_indices, final_step_indices],
            "dones": self.dones[env_indices, final_step_indices],
            "truncs": self.truncs[env_indices, final_step_indices],
            "n_steps": effective_n_steps,
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