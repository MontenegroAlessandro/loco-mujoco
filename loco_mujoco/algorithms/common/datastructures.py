import jax.numpy as jnp
from flax import struct

@struct.dataclass
class ReplayBuffer:
    """
    [AM] Replay buffer for TD3 agent.
    """
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_obs: jnp.ndarray
    dones: jnp.ndarray
    ptr: int    # pointer to the current index
    size: int   # size of the replay buffer