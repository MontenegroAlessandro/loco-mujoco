import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
import distrax


def get_activation_fn(name: str):
    """ Get activation function by name from the flax.linen module."""
    try:
        # Use getattr to dynamically retrieve the activation function from jax.nn
        return getattr(nn, name)
    except AttributeError:
        raise ValueError(f"Activation function '{name}' not found. Name must be the same as in flax.linen!")


class FullyConnectedNet(nn.Module):

    hidden_layer_dims: Sequence[int]
    output_dim: int
    activation: str = "tanh"
    output_activation: str = None    # none means linear activation
    use_running_mean_stand: bool = True
    squeeze_output: bool = True

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)
        self.output_activation_fn = get_activation_fn(self.output_activation) \
            if self.output_activation is not None else lambda x: x

    @nn.compact
    def __call__(self, x):

        if self.use_running_mean_stand:
            x = RunningMeanStd()(x)

        # build network
        for i, dim_layer in enumerate(self.hidden_layer_dims):
            x = nn.Dense(dim_layer, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = self.activation_fn(x)

        # add last layer
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        x = self.output_activation_fn(x)

        return jnp.squeeze(x) if self.squeeze_output else x

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    init_std: float = 1.0
    learnable_std: bool = True
    hidden_layer_dims: Sequence[int] = (1024, 512)
    actor_obs_ind: jnp.ndarray = None
    critic_obs_ind: jnp.ndarray = None

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)

    @nn.compact
    def __call__(self, x):

        x = RunningMeanStd()(x)

        # build actor
        actor_x = x if self.actor_obs_ind is None else x[..., self.actor_obs_ind]
        actor_mean = FullyConnectedNet(self.hidden_layer_dims, self.action_dim, self.activation,
                                       None, False, False)(actor_x)
        actor_logtstd = self.param("log_std", nn.initializers.constant(jnp.log(self.init_std)),
                                   (self.action_dim,))
        if not self.learnable_std:
            actor_logtstd = jax.lax.stop_gradient(actor_logtstd)

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        # build critic
        critic_x = x if self.critic_obs_ind is None else x[..., self.critic_obs_ind]
        critic = FullyConnectedNet(self.hidden_layer_dims, 1, self.activation, None, False, False)(critic_x)

        return pi, jnp.squeeze(critic, axis=-1)

class TD3Actor(nn.Module):
    """"
    [AM] Actor network for TD3 (i.e., deterministic policy).
    """
    action_dim: int
    hidden_layer_dims: Sequence[int] = (256,256)
    activation: str = "tanh"

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)

    @nn.compact
    def __call__(self, obs):
        x = obs
        x = RunningMeanStd()(x)

        action = FullyConnectedNet(
            hidden_layer_dims=self.hidden_layer_dims,
            output_dim=self.action_dim,
            activation=self.activation,
            output_activation=None,
            use_running_mean_stand=False,
            squeeze_output=False
        )(x)
        
        return action
    
class TD3Critic(nn.Module):
    """
    [AM] Critic twin networks for TD3 (i.e., Q-function).
    """
    hidden_layer_dims: Sequence[int] = (256,256)
    activation: str = "tanh"

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)

    @nn.compact
    def __call__(self, obs, action):
        # concatenate observation and action
        x = jnp.concatenate([obs, action], axis=-1)
        x = RunningMeanStd()(x)

        # get first critic result
        q1 = FullyConnectedNet(
            hidden_layer_dims=self.hidden_layer_dims,
            output_dim=1,
            activation=self.activation,
            output_activation=None,
            use_running_mean_stand=False,
            squeeze_output=False
        )(x)
        q1 = jnp.squeeze(q1, axis=-1)

        # get second critic result
        q2 = FullyConnectedNet(
            hidden_layer_dims=self.hidden_layer_dims,
            output_dim=1,
            activation=self.activation,
            output_activation=None,
            use_running_mean_stand=False,
            squeeze_output=False
        )(x)
        q2 = jnp.squeeze(q2, axis=-1)
        
        return q1, q2


class DistributionalQNetwork(nn.Module):
    num_atoms: int
    hidden_layer_dims: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs, action, update_stats: bool = True):
        x = jnp.concatenate([obs, action], axis=-1)
        x = RunningMeanStd()(x) 

        logits = FullyConnectedNet(
            hidden_layer_dims=self.hidden_layer_dims,
            output_dim=self.num_atoms,
            activation=self.activation,
            output_activation=None,
            use_running_mean_stand=False, 
            squeeze_output=False
        )(x)
        
        return logits

class FastTD3Critic(nn.Module):
    num_atoms: int
    v_min: float
    v_max: float
    hidden_layer_dims: Sequence[int] = (256, 256)
    activation: str = "relu"

    def setup(self):
        self.q_support = jnp.linspace(self.v_min, self.v_max, self.num_atoms)
        
        self.qnet1 = DistributionalQNetwork(
            num_atoms=self.num_atoms,
            hidden_layer_dims=self.hidden_layer_dims,
            activation=self.activation
        )
        self.qnet2 = DistributionalQNetwork(
            num_atoms=self.num_atoms,
            hidden_layer_dims=self.hidden_layer_dims,
            activation=self.activation
        )
    
    def __call__(self, obs, action, update_stats: bool = True):
        logits1 = self.qnet1(obs, action, update_stats=update_stats)
        logits2 = self.qnet2(obs, action, update_stats=update_stats)
        return logits1, logits2
    
    def _project_single(self, logits, rewards, bootstrap, discount):
        """Helper function to perform projection on a single distribution."""
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

        target_z = rewards[:, None] + bootstrap[:, None] * discount * self.q_support
        target_z = jnp.clip(target_z, self.v_min, self.v_max)
        
        b = (target_z - self.v_min) / delta_z
        l = jnp.floor(b).astype(jnp.int32)
        u = jnp.ceil(b).astype(jnp.int32)

        l = jnp.where(l == u, l - 1, l)
        u = jnp.where(u > l, u, u + 1)
        l = jnp.clip(l, 0, self.num_atoms - 1)
        u = jnp.clip(u, 0, self.num_atoms - 1)

        next_dist = nn.softmax(logits, axis=1)
        proj_dist = jnp.zeros_like(next_dist)
        
        proj_dist = proj_dist.at[jnp.arange(batch_size)[:, None], l].add(next_dist * (u - b))
        proj_dist = proj_dist.at[jnp.arange(batch_size)[:, None], u].add(next_dist * (b - l))

        return proj_dist

    def projection(self, logits1, logits2, rewards, bootstrap, discount):
        proj1 = self._project_single(logits1, rewards, bootstrap, discount)
        proj2 = self._project_single(logits2, rewards, bootstrap, discount)
        return proj1, proj2
    
    def get_value(self, probs):
        """Calculates the expected Q-value from a probability distribution."""
        return jnp.sum(probs * self.q_support, axis=-1)


class RunningMeanStd(nn.Module):
    """Layer that maintains running mean and variance for input normalization."""

    @nn.compact
    def __call__(self, x):

        x = jnp.atleast_2d(x)

        # Initialize running mean, variance, and count
        mean = self.variable('run_stats', 'mean', lambda: jnp.zeros(x.shape[-1]))
        var = self.variable('run_stats', 'var', lambda: jnp.ones(x.shape[-1]))
        count = self.variable('run_stats', 'count', lambda: jnp.array(1e-6))

        # Compute batch mean and variance
        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0) + 1e-6  # Add epsilon for numerical stability
        batch_count = x.shape[0]

        # Update counts
        updated_count = count.value + batch_count

        # Numerically stable mean and variance update
        delta = batch_mean - mean.value
        new_mean = mean.value + delta * batch_count / updated_count

        # Compute the new variance using Welford's method
        m_a = var.value * count.value
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * count.value * batch_count / updated_count
        new_var = M2 / updated_count

        # Normalize input
        normalized_x = (x - new_mean) / jnp.sqrt(new_var + 1e-8)

        # Update state variables
        mean.value = new_mean
        var.value = new_var
        count.value = updated_count

        return jnp.squeeze(normalized_x)
