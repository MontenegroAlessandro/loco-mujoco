# td3_experiment.py

import os
import sys
import jax
import jax.numpy as jnp
import wandb
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import TD3Jax  
from loco_mujoco.core.wrappers import LogWrapper, VecEnv, NormalizeVecReward 

import hydra
from omegaconf import DictConfig, OmegaConf
import traceback


@hydra.main(version_base=None, config_path="./", config_name="conf.yaml")
def experiment(config: DictConfig):
    try:
        os.environ['XLA_FLAGS'] = ('--xla_gpu_triton_gemm_any=True ')
        result_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        # Setup wandb
        wandb.login()
        config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        run = wandb.init(project=config.wandb.project, config=config_dict)

        # Create env
        factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
        env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
        
        # --- TD3 SPECIFIC CHANGES START HERE ---

        # Wrap env to log episode stats
        env = LogWrapper(env)
        env = VecEnv(env)
        if config.experiment.normalize_env:
            env = NormalizeVecReward(env, config.experiment.gamma)
        
        # Get initial agent configuration
        agent_conf = TD3Jax.init_agent_conf(env, config)

        # Build training function
        train_fn = TD3Jax.build_train_fn(env, agent_conf)

        # JIT and vmap training function
        train_fn = jax.vmap(train_fn) if config.experiment.n_seeds > 1 else train_fn

        # Get rng keys and run training
        rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds + 1)]
        rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
        out = train_fn(_rng)

        # Save agent state
        agent_state = out["agent_state"]
        save_path = TD3Jax.save_agent(result_dir, agent_conf, agent_state)
        run.config.update({"agent_save_path": save_path})

        # --- METRICS LOGGING ---
        if not config.experiment.debug:
            metrics = out["metrics"]
            # To get episode returns, you must ensure your TD3 _train_fn also returns them.
            # Assuming the LogWrapper provides them in `info` and they are passed up.
            episode_metrics = out["episode_metrics"] 

            # Calculate mean across seeds
            metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), metrics)
            episode_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), episode_metrics)
            
            # Log metrics
            for i in range(len(metrics.critic_loss)):
                step = int(i * config.experiment.num_envs)
                log_data = {
                    "Loss/Critic Loss": metrics.critic_loss[i],
                    "Loss/Actor Loss": metrics.actor_loss[i],
                    "Episode/Mean Return": jnp.mean(episode_metrics.returned_episode_returns[i]),
                    "Episode/Mean Length": jnp.mean(episode_metrics.returned_episode_lengths[i])
                }
                run.log(log_data, step=step)

        # Run the environment with the trained agent to record video
        TD3Jax.play_policy(env, agent_conf, agent_state, n_envs=1, n_steps=1000, record=True, deterministic=True)
        video_file = env.video_file_path
        run.log({"Agent Video": wandb.Video(video_file)})

        wandb.finish()

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    experiment()