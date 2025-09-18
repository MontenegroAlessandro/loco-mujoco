import jax
import jax.numpy as jnp
import optax
import flax
import flax.linen as nn
from dataclasses import dataclass
from loco_mujoco.algorithms import AgentConfBase, AgentStateBase, TD3Actor, TD3Critic, JaxRLAlgorithmBase, FastTD3Critic, ReplayBuffer, TrainState
from omegaconf import DictConfig, OmegaConf
from typing import Any
from flax import struct
from flax.core import FrozenDict
from loco_mujoco.utils import MetricsHandler

@dataclass(frozen=True)
class FastTD3AgentConf(AgentConfBase):
    """Static configuration for the FastTD3 agent."""
    config: DictConfig
    actor_module: TD3Actor 
    critic_module: FastTD3Critic # Use the new distributional critic
    actor_tx: optax.GradientTransformation
    critic_tx: optax.GradientTransformation

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
class FastTD3AgentState(AgentStateBase):
    """Dynamic state for the FastTD3 agent."""
    actor_train_state: TrainState
    critic_train_state: TrainState
    target_actor_params: flax.core.FrozenDict
    target_critic_params: flax.core.FrozenDict
    replay_buffer: ReplayBuffer
    noise_scales: jnp.ndarray
    
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

class FastTD3Jax(JaxRLAlgorithmBase):
    _agent_conf = FastTD3AgentConf
    _agent_state = FastTD3AgentState

    @classmethod
    def init_agent_conf(cls, env, config: DictConfig) -> FastTD3AgentConf:
        """Initializes the static agent configuration for FastTD3."""
        
        # Actor is the same as in standard TD3
        actor_module = TD3Actor(
            action_dim=env.info.action_space.shape[0],
            hidden_layer_dims=config.experiment.actor_hidden_dims
        )
        
        # Critic now uses the distributional network and its specific hyperparameters
        critic_module = FastTD3Critic(
            hidden_layer_dims=config.experiment.critic_hidden_dims,
            num_atoms=config.experiment.num_atoms,
            v_min=config.experiment.v_min,
            v_max=config.experiment.v_max
        )

        # Optimizers
        actor_tx = optax.adamw(learning_rate=config.experiment.actor_lr)
        critic_tx = optax.adamw(learning_rate=config.experiment.critic_lr)

        return FastTD3AgentConf(
            config=config,
            actor_module=actor_module,
            critic_module=critic_module,
            actor_tx=actor_tx,
            critic_tx=critic_tx,
        )

    @staticmethod
    def _create_initial_agent_state(rng, env, agent_conf: FastTD3AgentConf) -> FastTD3AgentState:
        """Creates the initial dynamic state of the agent."""
        config = agent_conf.config.experiment
        
        # Create PRNG keys for network initialization
        rng, actor_key, critic_key, noise_key = jax.random.split(rng, 4)
        
        # Prepare dummy data for initialization
        obs_shape = env.info.observation_space.shape
        action_shape = env.info.action_space.shape
        dummy_obs = jnp.zeros((1, *obs_shape))
        dummy_action = jnp.zeros((1, *action_shape))
        
        # Initialize network parameters
        actor_variables = agent_conf.actor_module.init(actor_key, dummy_obs)
        critic_variables = agent_conf.critic_module.init(critic_key, dummy_obs, dummy_action)
        
        # Create TrainStates for actor and critic
        actor_train_state = TrainState.create(
            apply_fn=agent_conf.actor_module.apply,
            params=actor_variables['params'],
            tx=agent_conf.actor_tx,
            run_stats=actor_variables['run_stats'],
        )
        critic_train_state = TrainState.create(
            apply_fn=agent_conf.critic_module.apply,
            params=critic_variables['params'],
            tx=agent_conf.critic_tx,
            run_stats=critic_variables['run_stats'],
        )
        
        # Initialize target networks as copies of the main networks
        target_actor_params = actor_variables['params']
        target_critic_params = critic_variables['params']
        
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

        # initialize noise scales
        noise_scales = jax.random.uniform(
            noise_key,
            shape=(config.num_envs,1),
            minval=config.std_min,
            maxval=config.std_max
        )

        # Assemble and return the complete initial agent state
        return FastTD3AgentState(
            actor_train_state=actor_train_state,
            critic_train_state=critic_train_state,
            target_actor_params=target_actor_params,
            target_critic_params=target_critic_params,
            replay_buffer=replay_buffer,
            noise_scales=noise_scales
        )
    
    @classmethod
    def _train_fn(cls, rng, env, agent_conf: FastTD3AgentConf, mh: MetricsHandler = None):
        # extract the experiment config
        config = agent_conf.config.experiment
        action_limit = env.info.action_space.high[0]
        
        # create initial agent state and env state
        initial_agent_state = cls._create_initial_agent_state(rng, env, agent_conf)
        reset_rng = jax.random.split(rng, config.num_envs)
        obsv, env_state = env.reset(reset_rng)

        def _update_step(runner_state, _):
            # unroll info
            agent_state, env_state, last_obs, rng = runner_state
            
            # ENV INTERACTION
            # take randomness seeds
            rng, action_rng, noise_resample_rng = jax.random.split(rng, 3)

            actor_vars = {
                'params': agent_state.actor_train_state.params, 
                'run_stats': agent_state.actor_train_state.run_stats
            }

            # action selection with exploration noise
            action, updates = agent_state.actor_train_state.apply_fn(
                actor_vars, last_obs, mutable=['run_stats']
            ) # select the action from the deterministic policy

            new_actor_ts = agent_state.actor_train_state.replace(run_stats=updates['run_stats'])
            agent_state = agent_state.replace(actor_train_state=new_actor_ts)

            noise = jax.random.normal(action_rng, shape=action.shape) * agent_state.noise_scales # sample gaussian noise
            action = jnp.clip(action + noise, -action_limit, action_limit) # clip the action + noise

            # apply the action
            obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)

            # resample the noise
            new_scales = jax.random.uniform(
                noise_resample_rng,
                shape=(config.num_envs, 1),
                minval=config.std_min,
                maxval=config.std_max
            )
            updated_noise_scales = jnp.where(
                done[:, None], new_scales, agent_state.noise_scales
            )

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
            agent_state = agent_state.replace(replay_buffer=new_replay_buffer, noise_scales=updated_noise_scales)
            
            # LEARNING
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

                actor_vars = {
                    'params': agent_state.target_actor_params, 
                    'run_stats': agent_state.actor_train_state.run_stats
                }
                next_action_dist, _ = agent_state.actor_train_state.apply_fn(actor_vars, batch.next_obs, mutable=['run_stats'])
                next_action = (next_action_dist + noise).clip(-action_limit, action_limit)
                
                critic_vars = {
                    'params': agent_state.target_critic_params, 
                    'run_stats': agent_state.critic_train_state.run_stats
                }
                (target_logits1, target_logits2), _ = agent_conf.critic_module.apply(critic_vars, batch.next_obs, next_action, mutable=['run_stats'])
                
                # Project the target distributions
                proj1, proj2 = agent_conf.critic_module.apply(
                    critic_vars, target_logits1, target_logits2, batch.rewards,
                    (1.0 - batch.dones), config.gamma, method=agent_conf.critic_module.projection
                )

                # Clipped Double Q-Learning: select the distribution corresponding to the smaller Q-value
                q1_val = agent_conf.critic_module.apply(
                    critic_vars, nn.softmax(target_logits1), method=agent_conf.critic_module.get_value
                )
                q2_val = agent_conf.critic_module.apply(
                    critic_vars, nn.softmax(target_logits2), method=agent_conf.critic_module.get_value
                )
                
                # Select the target distribution from the critic with the lower Q-value
                target_dist = jnp.where(q1_val[:, None] < q2_val[:, None], proj1, proj2)

                def _critic_loss_fn(critic_params):
                    critic_vars_loss = {
                        'params': critic_params, 
                        'run_stats': agent_state.critic_train_state.run_stats
                    }
                    (logits1, logits2), _ = agent_conf.critic_module.apply(critic_vars_loss, batch.obs, batch.actions, mutable=['run_stats'])
                    loss1 = -jnp.sum(target_dist * nn.log_softmax(logits1), axis=1).mean()
                    loss2 = -jnp.sum(target_dist * nn.log_softmax(logits2), axis=1).mean()
                    return loss1 + loss2

                critic_loss, critic_grads = jax.value_and_grad(_critic_loss_fn)(agent_state.critic_train_state.params)
                critic_train_state = agent_state.critic_train_state.apply_gradients(grads=critic_grads)
                
                # delayed policy updates
                def _actor_and_target_update(actor_ts, critic_ts, target_actor_p, target_critic_p):
                    def _actor_loss_fn(actor_params):
                        actor_vars_loss = {'params': actor_params, 'run_stats': actor_ts.run_stats}
                        actions, _ = agent_conf.actor_module.apply(actor_vars_loss, batch.obs, mutable=['run_stats'])
                        
                        critic_vars_loss_actor = {'params': critic_ts.params, 'run_stats': critic_ts.run_stats}
                        (q_logits, _), _ = agent_conf.critic_module.apply(critic_vars_loss_actor, batch.obs, actions, mutable=['run_stats'])
                        
                        q_probs = nn.softmax(q_logits)
                        
                        q_probs = nn.softmax(q_logits)
                        q_value = agent_conf.critic_module.apply(
                            critic_vars_loss_actor, q_probs, method=agent_conf.critic_module.get_value
                        )
                        return -q_value.mean()
                    
                    actor_loss, actor_grads = jax.value_and_grad(_actor_loss_fn)(actor_ts.params)
                    new_actor_ts = actor_ts.apply_gradients(grads=actor_grads)
                    
                    # Target network soft update 
                    new_target_actor_p = jax.tree.map(lambda x, y: x * (1.0 - config.tau) + y * config.tau, target_actor_p, new_actor_ts.params)
                    new_target_critic_p = jax.tree.map(lambda x, y: x * (1.0 - config.tau) + y * config.tau, target_critic_p, critic_ts.params)
                    
                    return new_actor_ts, new_target_actor_p, new_target_critic_p, actor_loss

                # Delayed update
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

            # UTD updates
            rng, learning_rng = jax.random.split(rng)

            def _perform_learning_updates(agent_s, rng_key):
                """A function that runs the learning step `utd_ratio` times."""

                # This is the body of our for-loop. It performs one learning step.
                def _single_update(i, loop_carry):
                    agent_s_loop, metrics_accumulator = loop_carry
                    
                    # Create a unique RNG key for each gradient update
                    update_rng = jax.random.fold_in(rng_key, i)
                    
                    # Perform one learning step
                    updated_agent_s, metrics = _learning_step(agent_s_loop, update_rng)
                    
                    # Accumulate the metrics from this step
                    metrics_accumulator = jax.tree.map(lambda acc, new: acc + new, metrics_accumulator, metrics)
                    
                    return updated_agent_s, metrics_accumulator

                # Initial values for the loop carry
                initial_metrics = {"critic_loss": 0.0, "actor_loss": 0.0}
                
                # Use fori_loop to run the update `utd_ratio` times
                final_agent_s, total_metrics = jax.lax.fori_loop(
                    0,
                    config.utd_ratio,
                    _single_update,
                    (agent_s, initial_metrics)
                )
                
                # Average the metrics over the number of updates
                avg_metrics = jax.tree.map(lambda x: x / config.utd_ratio, total_metrics)
                
                return final_agent_s, avg_metrics

            # Use cond to call the learning loop only after `learning_starts`
            agent_state, metrics = jax.lax.cond(
                agent_state.replay_buffer.size > config.learning_starts,
                lambda: _perform_learning_updates(agent_state, learning_rng),
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
    def play_policy(cls, env, agent_conf: FastTD3AgentConf, agent_state: FastTD3AgentState, n_envs: int, 
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