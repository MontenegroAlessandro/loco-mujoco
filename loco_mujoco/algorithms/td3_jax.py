import jax
import jax.numpy as jnp
import optax
import flax
import numpy as np
from dataclasses import dataclass
from loco_mujoco.algorithms import AgentConfBase, AgentStateBase, TD3Actor, TD3Critic, JaxRLAlgorithmBase, ReplayBuffer, TrainState
from omegaconf import DictConfig, OmegaConf
from typing import Any
from flax import struct
from flax.core import FrozenDict
from loco_mujoco.utils import MetricsHandler
from tqdm import tqdm
from functools import partial

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
class TD3AgentState(AgentStateBase):
    """
    [AM] Agent state for TD3 agent.
    """
    actor_train_state: TrainState
    critic_train_state: TrainState
    target_actor_params: FrozenDict
    target_critic_params: FrozenDict
    # replay_buffer: ReplayBuffer
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
            # replay_buffer=flax.serialization.from_state_dict(ReplayBuffer, d["replay_buffer"])
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
            hidden_layer_dims=config.experiment.actor_hidden_dims,
            activation=config.experiment.activation
        )
        
        critic_module = TD3Critic(
            hidden_layer_dims=config.experiment.critic_hidden_dims,
            activation=config.experiment.activation
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

        # Noise scales 
        noise_scales = jax.random.uniform(
            noise_key, shape=(config.num_envs, 1),
            minval=config.std_min, maxval=config.std_max
        )
        
        # Initialize an empty replay buffer
        # buffer_size = config.buffer_size
        # replay_buffer = ReplayBuffer(
        #     obs=jnp.zeros((buffer_size, env.info.observation_space.shape[0])),
        #     actions=jnp.zeros((buffer_size, env.info.action_space.shape[0])),
        #     rewards=jnp.zeros(buffer_size),
        #     next_obs=jnp.zeros((buffer_size, env.info.observation_space.shape[0])),
        #     dones=jnp.zeros(buffer_size, dtype=jnp.int32),
        #     ptr=0,
        #     size=0
        # )

        # Assemble and return the complete initial agent state
        return TD3AgentState(
            actor_train_state=actor_train_state,
            critic_train_state=critic_train_state,
            target_actor_params=target_actor_params,
            target_critic_params=target_critic_params,
            # replay_buffer=replay_buffer,
            noise_scales=noise_scales
        )
    
    @classmethod
    def _train_fn(
        cls,
        rng,
        env,
        agent_conf:TD3AgentConf,
        mh: MetricsHandler = None,
        eval_env = None,
        wandb_run = None
    ):
        # extract the experiment config
        config = agent_conf.config.experiment
        action_limit = env.info.action_space.high[0]

        @jax.jit
        def _learning_step(agent_state, batch):
            rng, noise_rng = jax.random.split(jax.random.PRNGKey(agent_state.actor_train_state.step))
            
            # update critic
            noise = (jax.random.normal(noise_rng, batch["actions"].shape) * config.policy_noise).clip(-config.noise_clip, config.noise_clip)
            
            actor_vars_target = {'params': agent_state.target_actor_params, 'run_stats': agent_state.actor_train_state.run_stats}
            next_action_dist, _ = agent_state.actor_train_state.apply_fn(actor_vars_target, batch["next_obs"], mutable=['run_stats'])
            next_action = (next_action_dist + noise).clip(-action_limit, action_limit)

            critic_vars_target = {'params': agent_state.target_critic_params, 'run_stats': agent_state.critic_train_state.run_stats}
            (q1_next, q2_next), _ = agent_conf.critic_module.apply(critic_vars_target, batch["next_obs"], next_action, mutable=['run_stats'])
            min_q_next = jnp.minimum(q1_next, q2_next)
            target_q = batch["rewards"] + (1.0 - batch["dones"]) * config.gamma * min_q_next

            def _critic_loss_fn(critic_params):
                critic_vars_loss = {'params': critic_params, 'run_stats': agent_state.critic_train_state.run_stats}
                (q1, q2), _ = agent_conf.critic_module.apply(critic_vars_loss, batch["obs"], batch["actions"], mutable=['run_stats'])
                critic_loss = ((q1 - target_q)**2).mean() + ((q2 - target_q)**2).mean()
                return critic_loss
            
            critic_loss, critic_grads = jax.value_and_grad(_critic_loss_fn)(agent_state.critic_train_state.params)
            critic_train_state = agent_state.critic_train_state.apply_gradients(grads=critic_grads)

            # update actor just using Q1 (as the paper says)
            def _actor_and_target_update(actor_ts, critic_ts, target_actor_p, target_critic_p):
                def _actor_loss_fn(actor_params):
                    actor_vars_loss = {'params': actor_params, 'run_stats': actor_ts.run_stats}
                    actions, _ = agent_conf.actor_module.apply(actor_vars_loss, batch["obs"], mutable=['run_stats'])
                    
                    critic_vars_loss_actor = {'params': critic_ts.params, 'run_stats': critic_ts.run_stats}
                    (q_val, _), _ = agent_conf.critic_module.apply(critic_vars_loss_actor, batch["obs"], actions, mutable=['run_stats'])
                    return -q_val.mean()
                
                actor_loss, actor_grads = jax.value_and_grad(_actor_loss_fn)(actor_ts.params)
                new_actor_ts = actor_ts.apply_gradients(grads=actor_grads)
                
                new_target_actor_p = jax.tree.map(lambda x, y: x * (1.0 - config.tau) + y * config.tau, target_actor_p, new_actor_ts.params)
                new_target_critic_p = jax.tree.map(lambda x, y: x * (1.0 - config.tau) + y * config.tau, target_critic_p, critic_ts.params)
                return new_actor_ts, new_target_actor_p, new_target_critic_p, actor_loss
            
            # Delayed update
            # actor_ts, target_actor_p, target_critic_p, actor_loss = jax.lax.cond(
            #     critic_train_state.step % config.policy_frequency == 0,
            #     lambda: _actor_and_target_update(agent_state.actor_train_state, critic_train_state, agent_state.target_actor_params, agent_state.target_critic_params),
            #     lambda: (agent_state.actor_train_state, agent_state.target_actor_params, agent_state.target_critic_params, 0.0)
            # )
            actor_ts, target_actor_p, target_critic_p, actor_loss = _actor_and_target_update(agent_state.actor_train_state, critic_train_state, agent_state.target_actor_params, agent_state.target_critic_params)
            
            metrics = {"critic_loss": critic_loss, "actor_loss": actor_loss}
            
            return agent_state.replace(
                actor_train_state=actor_ts,
                critic_train_state=critic_train_state,
                target_actor_params=target_actor_p,
                target_critic_params=target_critic_p
            ), metrics

        # [2] initialize the agent state and the replay buffer
        agent_state = cls._create_initial_agent_state(rng, env, agent_conf)
        replay_buffer = ReplayBuffer(
            obs=np.zeros((int(config.buffer_size), *env.info.observation_space.shape), dtype=np.float32),
            actions=np.zeros((int(config.buffer_size), *env.info.action_space.shape), dtype=np.float32),
            rewards=np.zeros(int(config.buffer_size), dtype=np.float32),
            next_obs=np.zeros((int(config.buffer_size), *env.info.observation_space.shape), dtype=np.float32),
            dones=np.zeros(int(config.buffer_size), dtype=np.float32),
            ptr=0, size=0
        )
        
        reset_rng = jax.random.split(rng, config.num_envs)
        obsv, env_state = env.reset(reset_rng)

        # metrics storing
        critic_losses = []
        actor_losses = []
        
        # [3] training loop
        num_updates = int(config.total_timesteps // config.num_envs)
        log_interval = config.get("log_interval", 100)
        log_interval = log_interval if log_interval < config.num_envs else int(log_interval // config.num_envs)
        
        for i in tqdm(range(num_updates)):
            # [3.1] environment interaction and replay buffer update
            rng, action_rng, noise_resample_rng = jax.random.split(rng, 3)
            actor_vars = {'params': agent_state.actor_train_state.params, 'run_stats': agent_state.actor_train_state.run_stats}
            action, updates = agent_state.actor_train_state.apply_fn(actor_vars, obsv, mutable=['run_stats'])
            new_actor_ts = agent_state.actor_train_state.replace(run_stats=updates['run_stats'])
            
            noise = jax.random.normal(action_rng, shape=action.shape) * agent_state.noise_scales
            action = jnp.clip(action + noise, -action_limit, action_limit)
            
            # next_obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)
            next_obsv, reward, absorbing, done, info, env_state = cls._wrap_step(env, env_state, action)

            # update the noise for the next interaction 
            new_scales = jax.random.uniform(noise_resample_rng, shape=(config.num_envs, 1), minval=config.std_min, maxval=config.std_max)
            updated_noise_scales = jnp.where(done[:, None], new_scales, agent_state.noise_scales)
            
            # replay buffer update (circular array)
            ptr = replay_buffer.ptr
            indices = (ptr + np.arange(config.num_envs)) % config.buffer_size
            replay_buffer.obs[indices] = np.asarray(obsv)
            replay_buffer.actions[indices] = np.asarray(action)
            replay_buffer.rewards[indices] = np.asarray(reward)
            replay_buffer.next_obs[indices] = np.asarray(next_obsv)
            replay_buffer.dones[indices] = np.asarray(done)
            replay_buffer.ptr = (ptr + config.num_envs) % config.buffer_size
            replay_buffer.size = min(replay_buffer.size + config.num_envs, config.buffer_size)
            
            obsv = next_obsv
            agent_state = agent_state.replace(actor_train_state=new_actor_ts, noise_scales=updated_noise_scales)
            
            # learn (just after the warm up)
            if i * config.num_envs > config.learning_starts:
                # learn for utd_ratio times
                for _ in range(config.utd_ratio):
                    # sample batch
                    batch_indices = np.random.randint(0, replay_buffer.size, size=config.batch_size)
                    batch = {
                        "obs": replay_buffer.obs[batch_indices],
                        "actions": replay_buffer.actions[batch_indices],
                        "rewards": replay_buffer.rewards[batch_indices],
                        "next_obs": replay_buffer.next_obs[batch_indices],
                        "dones": replay_buffer.dones[batch_indices],
                    }
                    # learn step
                    agent_state, metrics = _learning_step(agent_state, batch)
                    # metrics update
                    critic_losses.append(jax.device_get(metrics["critic_loss"]))
                    actor_losses.append(jax.device_get(metrics["actor_loss"]))
            
            # log stuff
            if i % log_interval == 0 and wandb_run is not None:
                log_data = {}

                # Add learning metrics if training has started
                if i * config.num_envs > config.learning_starts:
                    log_data["Loss/Critic Loss"] = jax.device_get(metrics["critic_loss"])
                    log_data["Loss/Actor Loss"] = jax.device_get(metrics["actor_loss"])

                eval_rng, rng = jax.random.split(rng)
                eval_return, eval_length = cls.run_evaluation(agent_conf, agent_state, eval_env, eval_rng)

                # Add evaluation metrics to the log data
                if not np.isnan(eval_return):
                    log_data["Evaluation/Mean Return"] = eval_return
                    log_data["Evaluation/Mean Length"] = eval_length

                if log_data:
                    wandb_run.log(log_data, step=i * config.num_envs)
        
        return {"agent_state": agent_state, "metrics": {"critic_loss": np.array(critic_losses), "actor_loss": np.array(actor_losses)}}
    
    @classmethod
    def run_evaluation(cls, agent_conf, agent_state, eval_env, rng):
        """Runs a deterministic evaluation and manually computes metrics."""
        config = agent_conf.config.experiment
        action_limit = eval_env.info.action_space.high[0]

        @jax.jit
        def _eval_step(carry, _):
            # Unpack the carry state
            agent_state, obsv, env_state, rng, episode_returns, episode_lengths = carry
            
            # Select action deterministically
            actor_vars = {'params': agent_state.actor_train_state.params, 
                          'run_stats': agent_state.actor_train_state.run_stats}
            action, _ = agent_state.actor_train_state.apply_fn(actor_vars, obsv, mutable=['run_stats'])
            action = jnp.clip(action, -action_limit, action_limit)

            # Step the environment
            next_obsv, reward, absorbing, done, info, env_state = eval_env.step(env_state, action)

            # Update current episode stats
            new_returns = episode_returns + reward
            new_lengths = episode_lengths + 1
            
            # Store the final return and length if an episode is done, otherwise store NaN
            finished_returns = jnp.where(done, new_returns, jnp.nan)
            finished_lengths = jnp.where(done, new_lengths, jnp.nan)
            
            # Reset stats for environments that are done
            next_episode_returns = jnp.where(done, 0.0, new_returns)
            next_episode_lengths = jnp.where(done, 0, new_lengths)
            
            # Pack the next carry state and the output for this step
            next_carry = (agent_state, next_obsv, env_state, rng, next_episode_returns, next_episode_lengths)
            output = (finished_returns, finished_lengths)

            return next_carry, output

        # scan loop
        num_eval_envs = config.validation.num_envs
        reset_rng = jax.random.split(rng, num_eval_envs)
        obsv, env_state = eval_env.reset(reset_rng)

        # Initial carry now includes arrays to track returns and lengths
        initial_carry = (
            agent_state, obsv, env_state, rng,
            jnp.zeros(num_eval_envs), jnp.zeros(num_eval_envs)
        )
        
        # Run the evaluation loop
        _, (all_returns, all_lengths) = jax.lax.scan(
            _eval_step, initial_carry, None, length=config.validation.num_steps
        )

        # Calculate the mean over all completed episodes, ignoring the NaNs
        mean_return = jnp.nanmean(all_returns)
        mean_length = jnp.nanmean(all_lengths)
        
        return mean_return, mean_length
    
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

    @staticmethod
    @partial(jax.jit, static_argnames=['env'])
    def _wrap_step(env, env_state, action):
        return env.step(env_state, action)
