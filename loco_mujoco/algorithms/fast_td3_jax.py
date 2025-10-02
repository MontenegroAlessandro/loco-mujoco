import jax
import jax.numpy as jnp
import optax
import flax
import numpy as np
from dataclasses import dataclass
from loco_mujoco.algorithms import (
    AgentConfBase, AgentStateBase, TD3Actor, FastTD3Critic, JaxRLAlgorithmBase, 
    DecoupledReplayBuffer, TrainState, PhasedExplorationSchedule, RunningMeanStdState
)
from omegaconf import DictConfig, OmegaConf
from typing import Any
from flax import struct
from flax.core import FrozenDict
from loco_mujoco.utils import MetricsHandler
from tqdm import tqdm
from functools import partial

@dataclass(frozen=True)
class FastTD3AgentConf(AgentConfBase):
    """
    [AM] Static configuration for TD3 agent.
    """
    config: DictConfig
    actor_module: TD3Actor 
    critic_module: FastTD3Critic
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
        critic_module = flax.serialization.from_state_dict(FastTD3Critic, d["critic_module"])
        actor_tx = optax.adamw(learning_rate=config.experiment.actor_lr)
        critic_tx = optax.adamw(learning_rate=config.experiment.critic_lr)
        return cls(config=config, actor_module=actor_module, critic_module=critic_module,actor_tx=actor_tx, critic_tx=critic_tx)

@struct.dataclass
class FastTD3AgentState(AgentStateBase):
    """
    [AM] Agent state for TD3 agent.
    """
    actor_train_state: TrainState
    critic_train_state: TrainState
    target_actor_params: FrozenDict
    target_critic_params: FrozenDict
    noise_scales: jnp.ndarray
    obs_normalizer_state: RunningMeanStdState

    def serialize(self):
        serialized_state = {
            "actor_train_state": flax.serialization.to_state_dict(self.actor_train_state),
            "critic_train_state": flax.serialization.to_state_dict(self.critic_train_state),
            "target_actor_params": self.target_actor_params,
            "target_critic_params": self.target_critic_params,
            "noise_scales": self.noise_scales,
            "obs_normalizer_state": flax.serialization.to_state_dict(self.obs_normalizer_state) 
        }
        return serialized_state

    @classmethod
    def from_dict(cls, d, agent_conf):
        actor_ts = TrainState.create(apply_fn=agent_conf.actor_module.apply, params={}, tx=agent_conf.actor_tx)
        critic_ts = TrainState.create(apply_fn=agent_conf.critic_module.apply, params={}, tx=agent_conf.critic_tx)

        obs_shape = np.array(d["obs_normalizer_state"]["mean"]).shape
        normalizer_state = RunningMeanStdState.create(obs_shape)
        
        return cls(
            actor_train_state=flax.serialization.from_state_dict(actor_ts, d["actor_train_state"]),
            critic_train_state=flax.serialization.from_state_dict(critic_ts, d["critic_train_state"]),
            target_actor_params=d["target_actor_params"],
            target_critic_params=d["target_critic_params"],
            noise_scales=d["noise_scales"],
            obs_normalizer_state=flax.serialization.from_state_dict(normalizer_state, d["obs_normalizer_state"])
        )

@struct.dataclass
class EvalState:
    """State for the episodic evaluation loop."""
    env_state: Any
    obs: jnp.ndarray
    rng: jax.random.PRNGKey
    episode_returns: jnp.ndarray
    episode_cumulative_rewards: jnp.ndarray
    episode_lengths: jnp.ndarray
    dones: jnp.ndarray 
    discounts: jnp.ndarray

class FastTD3Jax(JaxRLAlgorithmBase):
    """
    [AM] Fast TD3 algorithm implementation in JAX.
    """
    _agent_conf: FastTD3AgentConf
    _agent_state: FastTD3AgentState

    @classmethod
    def init_agent_conf(cls, env, config: DictConfig) -> FastTD3AgentConf:
        """Initializes the static agent configuration."""
        # Instantiate the network modules
        actor_module = TD3Actor(
            action_dim=env.info.action_space.shape[0],
            hidden_layer_dims=config.experiment.actor_hidden_dims,
            activation=config.experiment.activation,
            max_action=env.info.action_space.high[0]
        )
        
        critic_module = FastTD3Critic(
            hidden_layer_dims=config.experiment.critic_hidden_dims,
            activation=config.experiment.activation,
            num_atoms=config.experiment.num_atoms,  
            v_min=config.experiment.v_min,          
            v_max=config.experiment.v_max
        )

        # Define the optimizers
        actor_tx = optax.adamw(learning_rate=config.experiment.actor_lr)
        critic_tx = optax.adamw(learning_rate=config.experiment.critic_lr)

        # Return the populated agent configuration
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
            run_stats=None,
        )
        critic_train_state = TrainState.create(
            apply_fn=agent_conf.critic_module.apply,
            params=critic_variables['params'],
            tx=agent_conf.critic_tx,
            run_stats=None
        )
        
        # Initialize target networks as copies of the main networks
        target_actor_params = actor_variables['params']
        target_critic_params = critic_variables['params']

        # Noise scales 
        noise_scales = jax.random.uniform(
            noise_key, shape=(config.num_envs, 1),
            minval=config.policy_exploration.min_exploration, maxval=config.policy_exploration.max_exploration
        )

        # initialize the normalizer
        obs_shape = env.info.observation_space.shape
        obs_normalizer_state = RunningMeanStdState.create(obs_shape)

        # Assemble and return the complete initial agent state
        return FastTD3AgentState(
            actor_train_state=actor_train_state,
            critic_train_state=critic_train_state,
            target_actor_params=target_actor_params,
            target_critic_params=target_critic_params,
            noise_scales=noise_scales,
            obs_normalizer_state=obs_normalizer_state
        )
    
    @classmethod
    def _train_fn(
        cls,
        rng,
        env,
        agent_conf: FastTD3AgentConf,
        mh: MetricsHandler = None,
        eval_env = None,
        wandb_run = None
    ):
        # extract the experiment config
        config = agent_conf.config.experiment
        action_limit = float(env.info.action_space.high[0])

        @jax.jit
        def _learning_step(agent_state, batch, rng):
            rng, noise_rng = jax.random.split(rng)
            
            # update critic
            normalized_obs = agent_state.obs_normalizer_state.normalize(batch["obs"])
            normalized_next_obs = agent_state.obs_normalizer_state.normalize(batch["next_obs"])

            # actions computation
            actor_vars_target = {'params': agent_state.target_actor_params}
            # actor_vars_target = {'params': agent_state.actor_train_state.params} # FIXME
            next_pi = agent_state.actor_train_state.apply_fn(actor_vars_target, normalized_next_obs)
            noise = jnp.clip(jax.random.normal(noise_rng, next_pi.shape) * config.target_noise, -config.target_noise_clip, config.target_noise_clip)
            next_actions = jnp.clip(next_pi + noise, -action_limit, action_limit)

            # critic values
            critic_vars_target = {'params': agent_state.target_critic_params}
            next_logits1, next_logits2 = agent_conf.critic_module.apply(critic_vars_target, normalized_next_obs, next_actions)
            next_dist1 = jax.nn.softmax(next_logits1)
            next_dist2 = jax.nn.softmax(next_logits2)

            support = jnp.linspace(config.v_min, config.v_max, config.num_atoms)
            next_q1 = jnp.sum(next_dist1 * support, axis=-1)
            next_q2 = jnp.sum(next_dist2 * support, axis=-1)

            use_dist1 = (next_q1 < next_q2)[:, None]
            next_dist = jnp.where(use_dist1, next_dist1, next_dist2)

            # take data handling the n-step bootstrap
            # rewards = batch["rewards"]
            # dones = batch["dones"].astype(bool)
            # truncs = batch.get("truncs", jnp.zeros_like(dones)).astype(bool)
            # n_steps = batch.get("n_steps", jnp.ones_like(dones, dtype=np.int32))

            # bootstrap = jnp.logical_or(truncs, ~dones).astype(jnp.float32)
            # gamma_n = config.gamma ** n_steps

            target_dist = FastTD3Critic.project_distribution(
                next_dist, batch["rewards"], batch["dones"], config.gamma,
                support, config.v_min, config.v_max
            )
            # target_dist = FastTD3Critic.project_distribution(
            #     next_dist,
            #     rewards=rewards,
            #     bootstrap=bootstrap,
            #     gamma_n=gamma_n,
            #     support=support,
            #     v_min=config.v_min,
            #     v_max=config.v_max
            # )
            target_dist = jax.lax.stop_gradient(target_dist)

            def _critic_loss_fn(critic_params):
                critic_vars_loss = {'params': critic_params}
                logits1, logits2 = agent_conf.critic_module.apply(critic_vars_loss, normalized_obs, batch["actions"])
                
                log_probs1 = jax.nn.log_softmax(logits1)
                log_probs2 = jax.nn.log_softmax(logits2)
                
                loss1 = -jnp.sum(target_dist * log_probs1, axis=-1).mean()
                loss2 = -jnp.sum(target_dist * log_probs2, axis=-1).mean()
                
                critic_loss = loss1 + loss2
                return critic_loss
            
            critic_loss, critic_grads = jax.value_and_grad(_critic_loss_fn)(agent_state.critic_train_state.params)
            critic_train_state = agent_state.critic_train_state.apply_gradients(grads=critic_grads)

            # update actor
            def _actor_and_target_update(actor_ts, critic_ts, target_actor_p, target_critic_p):
                def _actor_loss_fn(actor_params):
                    actor_vars_loss = {'params': actor_params}
                    actions = agent_conf.actor_module.apply(actor_vars_loss, normalized_obs)
                    
                    # FIXME: cdq
                    critic_vars_loss_actor = {'params': critic_ts.params}
                    logits1, logits2 = agent_conf.critic_module.apply(critic_vars_loss_actor, normalized_obs, actions)
                    q1_val = jnp.sum(jax.nn.softmax(logits1) * support, axis=-1)
                    q2_val = jnp.sum(jax.nn.softmax(logits2) * support, axis=-1)
                    q_val = jnp.minimum(q1_val, q2_val)
                    
                    return -jnp.mean(q_val)
                
                actor_loss, actor_grads = jax.value_and_grad(_actor_loss_fn)(actor_ts.params)
                new_actor_ts = actor_ts.apply_gradients(grads=actor_grads)
                
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

        # [2] initialize the agent state and the replay buffer
        agent_state = cls._create_initial_agent_state(rng, env, agent_conf)
        # replay_buffer = SuperReplayBuffer(
        #     total_capacity=int(config.buffer_size), # FIXME
        #     n_env=config.num_envs,
        #     obs_shape=env.info.observation_space.shape,
        #     action_shape=env.info.action_space.shape,
        #     n_steps=config.n_step,
        #     gamma=config.gamma
        # )
        replay_buffer = DecoupledReplayBuffer(
            total_capacity=int(config.buffer_size), # FIXME
            num_envs=config.num_envs,
            obs_shape=env.info.observation_space.shape,
            action_shape=env.info.action_space.shape,
            # n_steps=config.n_step,
            # gamma=config.gamma
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
        start_learning = int(config.learning_starts // config.num_envs)
        # utd = int(config.utd_ratio) if config.update_after == 0 else int(config.update_after)
        learning_started = False
        utd = int(config.utd_ratio)

        # if needed, initialize the exploration scheduler
        noise_scheduler = None
        if config.policy_exploration.active_schedule and config.policy_exploration.max_exploration > config.policy_exploration.min_exploration:
            lin_schedule = (config.policy_exploration.schedule_type == "linear")
            noise_scheduler = PhasedExplorationSchedule.create(
                phases=num_updates, noise_max=config.policy_exploration.max_exploration, noise_min=config.policy_exploration.min_exploration, linear=lin_schedule
            )
        
        print(f"Action Limits ({-action_limit},{action_limit})")
        for i in tqdm(range(num_updates)):
            # sample new scales
            if noise_scheduler is not None:
                new_scales = noise_scheduler.update_sigma(i+1) * jnp.ones((config.num_envs,1))
                agent_state = agent_state.replace(noise_scales=new_scales)

            # update the normalizer
            new_obs_normalizer_state = agent_state.obs_normalizer_state.update(obsv)

            # normalize observations
            normalized_obsv = new_obs_normalizer_state.normalize(obsv)

            # [3.1] environment interaction and replay buffer update
            rng, action_rng, noise_resample_rng = jax.random.split(rng, 3)
            actor_vars = {'params': agent_state.actor_train_state.params}
            action = agent_state.actor_train_state.apply_fn(actor_vars, normalized_obsv)
            
            # compute noise, noise the action, clip the noised action
            noise = jax.random.normal(action_rng, shape=action.shape) * agent_state.noise_scales
            noised_action = action + noise
            clipped_action = jnp.clip(noised_action, -action_limit, action_limit)
            
            next_obsv, reward, absorbing, done, info, env_state = cls._wrap_step(env, env_state, clipped_action)

            # update the noise for the next interaction 
            if noise_scheduler is None:
                new_scales = jax.random.uniform(
                    noise_resample_rng, shape=(config.num_envs, 1), 
                    minval=config.policy_exploration.min_exploration, 
                    maxval=config.policy_exploration.max_exploration
                )
                updated_noise_scales = jnp.where(done[:, None], new_scales, agent_state.noise_scales)
            else:
                updated_noise_scales = agent_state.noise_scales
            
            # replay buffer update (circular array)
            replay_buffer.add(
                obs=np.asarray(obsv), 
                action=np.asarray(noised_action), 
                reward=np.asarray(reward), 
                done=np.asarray(done),
                next_obs=np.asarray(next_obsv),
                # truncation=np.asarray(absorbing)
            )
            
            obsv = next_obsv
            agent_state = agent_state.replace(
                noise_scales=updated_noise_scales,
                obs_normalizer_state=new_obs_normalizer_state 
            )
            
            # learn (just after the warm up)
            if i > start_learning and (i % config.update_after == 0 or i == num_updates - 1):
                # learn for utd_ratio times
                keys = jax.random.split(rng, utd + 1)
                rng, update_keys = keys[0], keys[1:]
                
                # say that we can log losses
                if not learning_started:
                    learning_started = True

                for j in range(utd):
                    # sample batch
                    batch = replay_buffer.sample(config.batch_size)
                    # learn step
                    agent_state, metrics = _learning_step(agent_state, batch, update_keys[j])
                    # metrics update
                    critic_losses.append(jax.device_get(metrics["critic_loss"]))
                    actor_losses.append(jax.device_get(metrics["actor_loss"]))
            
            # log stuff
            if i % log_interval == 0 and wandb_run is not None:
                log_data = {}

                # Add learning metrics if training has started
                if learning_started:
                    log_data["Loss/Critic Loss"] = jax.device_get(metrics["critic_loss"])
                    log_data["Loss/Actor Loss"] = jax.device_get(metrics["actor_loss"])

                rng, eval_rng = jax.random.split(rng)
                eval_return, eval_cum_rew, eval_length = cls.run_episodic_evaluation(agent_conf, agent_state, eval_env, eval_rng)

                # Add evaluation metrics to the log data
                if not np.isnan(eval_return):
                    log_data["Evaluation/Mean Return (Discounted)"] = eval_return
                    log_data["Evaluation/Mean Return (UNDiscounted)"] = eval_cum_rew
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

            normalized_obs = agent_state.obs_normalizer_state.normalize(obsv)
            
            # Select action deterministically
            actor_vars = {
                'params': agent_state.actor_train_state.params, 
            }
            action = agent_state.actor_train_state.apply_fn(actor_vars, normalized_obs)
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
    def run_episodic_evaluation(cls, agent_conf, agent_state, eval_env, rng):
        """Runs a true episodic evaluation."""
        config = agent_conf.config.experiment
        action_limit = eval_env.info.action_space.high[0]
        num_eval_envs = config.validation.num_envs
        gamma = agent_conf.config.experiment.gamma
        
        # Reset environments
        reset_rngs = jax.random.split(rng, num_eval_envs)
        obsv, env_state = eval_env.reset(reset_rngs)

        # Define the evaluation state for the while_loop
        initial_eval_state = EvalState(
            env_state=env_state,
            obs=obsv,
            rng=rng,
            episode_returns=jnp.zeros(num_eval_envs),
            episode_cumulative_rewards=jnp.zeros(num_eval_envs),
            episode_lengths=jnp.zeros(num_eval_envs),
            dones=jnp.zeros(num_eval_envs, dtype=jnp.bool_),
            discounts=jnp.ones(num_eval_envs)
        )

        # Save the initial values for the quality nets
        # get states 
        actor_vars = {
            'params': agent_state.actor_train_state.params,
        }
        actions = agent_state.actor_train_state.apply_fn(actor_vars, obsv)
        # no clip the action before passing to the critic
        critic_vars = {
            'params': agent_state.critic_train_state.params,
        }

        def cond_fun(state: EvalState):
            """Loop continues as long as any environment is not done."""
            return jnp.any(~state.dones)

        def body_fun(state: EvalState):
            """Performs one step of the evaluation."""
            normalized_obs = agent_state.obs_normalizer_state.normalize(state.obs)

            # Select action deterministically
            actor_vars = {
                'params': agent_state.actor_train_state.params, 
            }
            action = agent_state.actor_train_state.apply_fn(actor_vars, normalized_obs)
            action = jnp.clip(action, -action_limit, action_limit)

            # Step the environment
            next_obsv, reward, _, done, _, next_env_state = eval_env.step(state.env_state, action)
            
            # Update returns and lengths only for environments that are still active
            new_cum_rew = jnp.where(
                state.dones, state.episode_cumulative_rewards, state.episode_cumulative_rewards + reward
            )
            reward = state.discounts * reward # discount the reward
            new_returns = jnp.where(
                state.dones, state.episode_returns, state.episode_returns + reward
            )
            new_discounts = jnp.where(
                state.dones, state.discounts, state.discounts * gamma
            )
            new_lengths = jnp.where(
                state.dones, state.episode_lengths, state.episode_lengths + 1
            )
            
            # Update the done status
            new_dones = jnp.logical_or(state.dones, done)
            
            return state.replace(
                env_state=next_env_state,
                obs=next_obsv,
                episode_returns=new_returns,
                episode_cumulative_rewards=new_cum_rew,
                episode_lengths=new_lengths,
                dones=new_dones,
                discounts=new_discounts
            )

        # JIT and run the while loop
        final_state = jax.lax.while_loop(cond_fun, body_fun, initial_eval_state)
        
        # The final returns and lengths are now stored in the state
        mean_return = jnp.mean(final_state.episode_returns)
        mean_length = jnp.mean(final_state.episode_lengths)
        mean_cum_reward = jnp.mean(final_state.episode_cumulative_rewards)
        print(f"{33 * '='}")
        print(f"Ret (disc) = {mean_return}")
        print(f"Ret (undisc) = {mean_cum_reward}")
        print(f"Len = {mean_length}")
        print(f"{33 * '='}")

        return mean_return, mean_cum_reward, mean_length

    @classmethod
    def play_policy(cls, env, agent_conf: FastTD3AgentConf, agent_state: FastTD3AgentState, n_envs: int, 
                    n_steps=None, render=True, record=False, rng=None, deterministic=True, **kwargs):
        
        action_limit = env.info.action_space.high[0]
        
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
            normalized_obs = agent_state.obs_normalizer_state.normalize(obs)

            rng, _rng = jax.random.split(rng)
            action = sample_action(agent_state.actor_train_state.params, normalized_obs)
            
            # Add exploration noise if not deterministic
            if not deterministic:
                noise = jax.random.normal(_rng, action.shape) * agent_conf.config.experiment.exploration_noise
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
