import jax
import jax.numpy as jnp
import optax
import flax
from dataclasses import dataclass
from loco_mujoco.algorithms import AgentConfBase, AgentStateBase, TD3Actor, TD3Critic, JaxRLAlgorithmBase
from omegaconf import DictConfig, OmegaConf
from typing import Any
from flax import struct
from flax.training.train_state import TrainState
from flax.core import FrozenDict
from loco_mujoco.utils import MetricsHandler

@dataclass(frozen=True)
class TD3AgentConf(AgentConfBase):
    """
    [AM] Static configuration for TD3 agent.
    """
    config: DictConfig
    actor_module: TD3Actor 
    critic_module: TD3Critic
    actor_tx: Any
    critic_tx: Any
    
    def serialize(self):
        conf_dict = OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True)
        serialized_actor = flax.serialization.to_state_dict(self.actor_module)
        serialized_critic = flax.serialization.to_state_dict(self.critic_module)
        return {"config": conf_dict, "actor_module": serialized_actor, "critic_module": serialized_critic}

    @classmethod
    def from_dict(cls, d):
        config = OmegaConf.create(d["config"])
        actor_module = flax.serialization.from_state_dict(TD3Actor, d["actor_module"])
        critic_module = flax.serialization.from_state_dict(TD3Critic, d["critic_module"])
        actor_tx = optax.adamw(learning_rate=config.experiment.actor_lr)
        critic_tx = optax.adamw(learning_rate=config.experiment.critic_lr)
        return cls(config=config, actor_module=actor_module, critic_module=critic_module,actor_tx=actor_tx, critic_tx=critic_tx)

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

@struct.dataclass
class TD3AgentState(AgentStateBase):
    """
    [AM] Agent state for TD3 agent.
    """
    actor_train_state: TrainState
    critic_train_state: TrainState
    target_actor_params: FrozenDict
    target_critic_params: FrozenDict
    replay_buffer: ReplayBuffer

    def serialize(self):
        serialized_state = {
            "actor_train_state": flax.serialization.to_state_dict(self.actor_train_state),
            "critic_train_state": flax.serialization.to_state_dict(self.critic_train_state),
            "target_actor_params": self.target_actor_params,
            "target_critic_params": self.target_critic_params,
            "replay_buffer": flax.serialization.to_state_dict(self.replay_buffer)
        }
        return serialized_state

    @classmethod
    def from_dict(cls, d, agent_conf):
        actor_ts = TrainState.create(apply_fn=agent_conf.actor_module.apply, params={}, tx=agent_conf.actor_tx)
        critic_ts = TrainState.create(apply_fn=agent_conf.critic_module.apply, params={}, tx=agent_conf.critic_tx)
        
        return cls(
            actor_train_state=flax.serialization.from_state_dict(actor_ts, d["actor_train_state"]),
            critic_train_state=flax.serialization.from_state_dict(critic_ts, d["critic_train_state"]),
            target_actor_params=d["target_actor_params"],
            target_critic_params=d["target_critic_params"],
            replay_buffer=flax.serialization.from_state_dict(ReplayBuffer, d["replay_buffer"])
        )

class TD3Jax(JaxRLAlgorithmBase):
    """
    [AM] TD3 algorithm implementation in JAX.
    """
    _agent_conf: TD3AgentConf
    _agent_state: TD3AgentState

    @classmethod
    def init_agent_conf(cls, env, config: DictConfig) -> TD3AgentConf:
        """Initializes the static agent configuration."""
        # Instantiate the network modules
        actor_module = TD3Actor(
            action_dim=env.info.action_space.shape[0],
            hidden_layer_dims=config.experiment.actor_hidden_dims
        )
        
        critic_module = TD3Critic(
            hidden_layer_dims=config.experiment.critic_hidden_dims
        )

        # Define the optimizers
        actor_tx = optax.adamw(learning_rate=config.experiment.actor_lr)
        critic_tx = optax.adamw(learning_rate=config.experiment.critic_lr)

        # Return the populated agent configuration
        return TD3AgentConf(
            config=config,
            actor_module=actor_module,
            critic_module=critic_module,
            actor_tx=actor_tx,
            critic_tx=critic_tx,
        )

    @staticmethod
    def _create_initial_agent_state(rng, env, agent_conf: TD3AgentConf) -> TD3AgentState:
        """Creates the initial dynamic state of the agent."""
        config = agent_conf.config.experiment
        
        # Create PRNG keys for network initialization
        rng, actor_key, critic_key = jax.random.split(rng, 3)
        
        # Prepare dummy data for initialization
        obs_shape = env.info.observation_space.shape
        action_shape = env.info.action_space.shape
        dummy_obs = jnp.zeros((1, *obs_shape))
        dummy_action = jnp.zeros((1, *action_shape))
        
        # Initialize network parameters
        actor_params = agent_conf.actor_module.init(actor_key, dummy_obs)['params']
        critic_params = agent_conf.critic_module.init(critic_key, dummy_obs, dummy_action)['params']
        
        # Create TrainStates for actor and critic
        actor_train_state = TrainState.create(
            apply_fn=agent_conf.actor_module.apply,
            params=actor_params,
            tx=agent_conf.actor_tx
        )
        critic_train_state = TrainState.create(
            apply_fn=agent_conf.critic_module.apply,
            params=critic_params,
            tx=agent_conf.critic_tx
        )
        
        # Initialize target networks as copies of the main networks
        target_actor_params = actor_params
        target_critic_params = critic_params
        
        # Initialize an empty replay buffer
        buffer_size = config.buffer_size
        replay_buffer = ReplayBuffer(
            obs=jnp.zeros((buffer_size, env.info.observation_space.shape[0])),
            actions=jnp.zeros((buffer_size, env.info.action_space.shape[0])),
            rewards=jnp.zeros(buffer_size),
            next_obs=jnp.zeros((buffer_size, env.info.observation_space.shape[0])),
            dones=jnp.zeros(buffer_size, dtype=jnp.int32),
            ptr=0,
            size=0
        )

        # Assemble and return the complete initial agent state
        return TD3AgentState(
            actor_train_state=actor_train_state,
            critic_train_state=critic_train_state,
            target_actor_params=target_actor_params,
            target_critic_params=target_critic_params,
            replay_buffer=replay_buffer
        )
    
    @classmethod
    def _train_fn(
        cls,
        rng,
        env,
        agent_conf:TD3AgentConf,
        mh: MetricsHandler = None
    ):
        # extract the experiment config
        config = agent_conf.config.experiment
        action_limit = env.info.action_space.high[0]

        # create initial agent state and env state
        initial_agent_state = cls._create_initial_agent_state(rng, env, agent_conf)
        reset_rng = jax.random.split(rng, config.num_envs)
        obsv, env_state = env.reset(reset_rng)

        # training step
        def _update_step(runner_state, _):
            # unroll info
            agent_state, env_state, last_obs, rng = runner_state

            # ENV INTERACTION
            # take randomness seeds
            rng, action_rng, step_rng = jax.random.split(rng, 3)

            # action selection with exploration noise
            action = agent_state.actor_train_state.apply_fn(
                {'params': agent_state.actor_train_state.params}, last_obs
            ) # select the action from the deterministic policy
            noise = jax.random.normal(action_rng, shape=action.shape) * config.exploration_noise # sample gaussian noise
            action = jnp.clip(action + noise, -action_limit, action_limit) # clip the action + noise

            # apply the action
            obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)

            # add the transition to the replay buffer
            replay_buffer = agent_state.replay_buffer
            ptr = replay_buffer.ptr
            indices = (ptr + jnp.arange(config.num_envs)) % config.buffer_size
            
            new_replay_buffer = replay_buffer.replace(
                obs=replay_buffer.obs.at[indices].set(last_obs),
                actions=replay_buffer.actions.at[indices].set(action),
                rewards=replay_buffer.rewards.at[indices].set(reward),
                next_obs=replay_buffer.next_obs.at[indices].set(obsv),
                dones=replay_buffer.dones.at[indices].set(done),
                ptr=(ptr + config.num_envs) % config.buffer_size,
                size=jnp.minimum(replay_buffer.size + config.num_envs, config.buffer_size)
            )
            agent_state = agent_state.replace(replay_buffer=new_replay_buffer)

            # LEARN STEP
            def _learning_step(agent_state, rng):
                # sample a batch of transitions from the replay buffer
                rng, sample_rng = jax.random.split(rng)
                batch_indices = jax.random.randint(sample_rng, (config.batch_size,), 0, agent_state.replay_buffer.size)
                batch = ReplayBuffer(
                    obs=agent_state.replay_buffer.obs[batch_indices],
                    actions=agent_state.replay_buffer.actions[batch_indices],
                    rewards=agent_state.replay_buffer.rewards[batch_indices],
                    next_obs=agent_state.replay_buffer.next_obs[batch_indices],
                    dones=agent_state.replay_buffer.dones[batch_indices],
                    ptr=0, 
                    size=0 
                )

                # update the critic
                rng, noise_rng = jax.random.split(rng)
                noise = (jax.random.normal(noise_rng, batch.actions.shape) * config.policy_noise).clip(-config.noise_clip, config.noise_clip)
                next_action = (
                agent_state.actor_train_state.apply_fn({'params': agent_state.target_actor_params}, batch.next_obs) + noise).clip(-action_limit, action_limit)

                q1_next, q2_next = agent_conf.critic_module.apply({'params': agent_state.target_critic_params}, batch.next_obs, next_action)
                min_q_next = jnp.minimum(q1_next, q2_next)
                target_q = batch.rewards + (1-batch.dones) * config.gamma * min_q_next

                def _critic_loss_fn(critic_params):
                    q1, q2 = agent_conf.critic_module.apply({'params': critic_params}, batch.obs, batch.actions)
                    critic_loss = ((q1 - target_q)**2).mean() + ((q2 - target_q)**2).mean()
                    return critic_loss
                
                # compute gradients and update critic
                critic_loss, critic_grads = jax.value_and_grad(_critic_loss_fn)(agent_state.critic_train_state.params)
                critic_train_state = agent_state.critic_train_state.apply_gradients(grads=critic_grads)

                # delayed policy updates
                def _actor_and_target_update(actor_ts, critic_ts, target_actor_p, target_critic_p):
                    # actor update
                    def _actor_loss_fn(actor_params):
                        actions = agent_conf.actor_module.apply({'params': actor_params}, batch.obs)
                        q_val, _ = agent_conf.critic_module.apply({'params': critic_ts.params}, batch.obs, actions)
                        return -q_val.mean()
                    
                    actor_loss, actor_grads = jax.value_and_grad(_actor_loss_fn)(actor_ts.params)
                    new_actor_ts = actor_ts.apply_gradients(grads=actor_grads)

                    # Target network soft update
                    new_target_actor_p = jax.tree.map(
                        lambda x, y: x * (1.0 - config.tau) + y * config.tau, target_actor_p, new_actor_ts.params
                    )
                    new_target_critic_p = jax.tree.map(
                        lambda x, y: x * (1.0 - config.tau) + y * config.tau, target_critic_p, critic_ts.params
                    )
                    return new_actor_ts, new_target_actor_p, new_target_critic_p, actor_loss
                
                # run dealyed update just when needed
                actor_ts, target_actor_p, target_critic_p, actor_loss = jax.lax.cond(
                    critic_train_state.step % config.policy_frequency == 0,
                    lambda: _actor_and_target_update(agent_state.actor_train_state, critic_train_state, agent_state.target_actor_params, agent_state.target_critic_params),
                    lambda: (agent_state.actor_train_state, agent_state.target_actor_params, agent_state.target_critic_params, 0.0)
                )
                metrics = {"critic_loss": critic_loss, "actor_loss": actor_loss}

                return agent_state.replace(
                    actor_train_state=actor_ts,
                    critic_train_state=critic_train_state,
                    target_actor_params=target_actor_p,
                    target_critic_params=target_critic_p
                ), metrics
            
            # Only run the learning step if the buffer is large enough
            agent_state, metrics = jax.lax.cond(
                agent_state.replay_buffer.size > config.learning_starts,
                lambda: _learning_step(agent_state, rng),
                lambda: (agent_state, {"critic_loss": 0.0, "actor_loss": 0.0})
            )

            runner_state = (agent_state, env_state, obsv, rng)

            return runner_state, metrics
        
        # --- Run the main loop ---
        runner_state_initial = (initial_agent_state, env_state, obsv, rng)
        final_runner_state, collected_metrics = jax.lax.scan(
            _update_step, runner_state_initial, None, length=config.total_timesteps
        )
        
        return {"agent_state": final_runner_state[0], "metrics": collected_metrics}
    
    @classmethod
    def play_policy(cls, env, agent_conf: TD3AgentConf, agent_state: TD3AgentState, n_envs: int, 
                    n_steps=None, render=True, record=False, rng=None, deterministic=True, **kwargs):
        
        @jax.jit
        def sample_action(params, obs):
            action = agent_conf.actor_module.apply({'params': params}, obs)
            return action

        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        keys = jax.random.split(rng, n_envs + 1)
        rng, env_keys = keys[0], keys[1:]

        obs, env_state = env.reset(env_keys)

        if n_steps is None:
            n_steps = float('inf')

        i = 0
        while i < n_steps:
            rng, _rng = jax.random.split(rng)
            action = sample_action(agent_state.actor_train_state.params, obs)
            
            # Add exploration noise if not deterministic
            if not deterministic:
                noise = jax.random.normal(_rng, action.shape) * agent_conf.config.experiment.exploration_noise
                action_limit = env.info.action_space.high[0]
                action = jnp.clip(action + noise, -action_limit, action_limit)

            obs, reward, absorbing, done, info, env_state = env.step(env_state, action)
            
            if render:
                env.mjx_render(env_state, record=record)
            
            i += 1
        
        env.stop()
