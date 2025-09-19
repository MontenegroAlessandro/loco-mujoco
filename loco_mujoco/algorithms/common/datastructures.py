import numpy as np
from dataclasses import dataclass

@dataclass
class ReplayBuffer:
    """
    [AM] Replay buffer for TD3 agent.
    """
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray
    ptr: int    # pointer to the current index
    size: int   # size of the replay buffer