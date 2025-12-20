import os
import sys
import wandb
from dataclasses import fields
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import traceback

import time

job_timestamp_str = time.strftime("%Y-%m-%d/%H-%M-%S", time.localtime())

@hydra.main(version_base=None, config_path="./", config_name="conf_t1")
def experiment(config: DictConfig):
    try:
        import jax
        import jax.numpy as jnp
        from loco_mujoco import TaskFactory
        from loco_mujoco.algorithms import PPOJax
        from loco_mujoco.utils.metrics import QuantityContainer
        from loco_mujoco.utils import MetricsHandler


        os.environ['XLA_FLAGS'] = (
            '--xla_gpu_triton_gemm_any=True '
            # '--xla_gpu_enable_cub_radix_sort=false '
            # '--xla_gpu_deterministic_ops=true '
            '--xla_gpu_autotune_level=1 '
            '--xla_gpu_force_compilation_parallelism=1 '
            # '--xla_gpu_disable_gpuasm_optimizations=true'

            )
        # Reduce JAX logging verbosity
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['JAX_LOG_LEVEL'] = 'WARNING'

        # Accessing the current sweep number
        result_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        # setup wandb
        # Accessing the current sweep number
        hydra_config = hydra.core.hydra_config.HydraConfig.get()
        result_dir = hydra_config.runtime.output_dir

        # get overrides
        overrides = hydra_config.overrides.task
        str_overrides = []
        if len(overrides) > 0:
            for override in overrides:
                value = override.split("=")[1]
                key = override.split("=")[0].split(".")[-1]
                str_overrides.append(f"{key}={value}")
        
        # setup wandb
        api_key = os.getenv("WANDB_KEY_ALE")
        if api_key is None:
            raise RuntimeError("WANDB_KEY_ALE is not set!")
        wandb.login(key=api_key)
        config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

        # name the run as time stamp + overrides
        wandb_run_name = f"{';'.join(str_overrides + [job_timestamp_str])}"

        run = wandb.init(
            entity="",
            project=config.wandb.project,
            name=wandb_run_name,
            config=config_dict)

        # get task factory
        factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

        # create env
        env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

        mh = MetricsHandler(config, env) if config.experiment.validation.active else None

        # get initial agent configuration
        agent_conf = PPOJax.init_agent_conf(env, config)
        agent_state = None
        resume_just_params = config.experiment.resume_just_params
        if "resume_from_path" in config.experiment:
            if resume_just_params:
                _, agent_state = PPOJax.load_agent(config.experiment.resume_from_path)
            else:
                agent_conf, agent_state = PPOJax.load_agent(config.experiment.resume_from_path)

        # build training function
        # train_fn = PPOJax.build_train_fn(env, agent_conf)

        train_fn = PPOJax.build_train_fn(env, agent_conf, agent_state=agent_state, mh=mh, wandb_run=run)

        # jit and vmap training function
        train_fn = jax.jit(jax.vmap(train_fn)) if config.experiment.n_seeds > 1 else jax.jit(train_fn)

        # get rng keys and run training
        rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds+1)]  # create rngs from seed
        rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
        out = train_fn(_rng)

        # save agent state
        agent_state = out["agent_state"]
        save_path = PPOJax.save_agent(result_dir, agent_conf, agent_state)
        run.config.update({"agent_save_path": save_path})

        # import time
        # t_start = time.time()
        # # get the metrics and log them
        # if not config.experiment.debug:
        #     training_metrics = out["training_metrics"]
        #     validation_metrics = out["validation_metrics"]

        #     # calculate mean across seeds
        #     training_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), training_metrics)
        #     validation_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), validation_metrics)

        #     for i in range(len(training_metrics.mean_episode_return)):
        #         run.log({"Mean Episode Return": training_metrics.mean_episode_return[i],
        #                  "Mean Episode Length": training_metrics.mean_episode_length[i]},
        #                 step=int(training_metrics.max_timestep[i]))

        #         if (i+1) % config.experiment.validation_interval == 0 and config.experiment.validation.active:
        #             run.log({"Validation Info/Mean Episode Return": validation_metrics.mean_episode_return[i],
        #                      "Validation Info/Mean Episode Length": validation_metrics.mean_episode_length[i]},
        #                     step=int(training_metrics.max_timestep[i]))

        #             # log all measures
        #             metrics_to_log = {}
        #             for field in fields(validation_metrics):
        #                 attr = getattr(validation_metrics, field.name)
        #                 if isinstance(attr, QuantityContainer):
        #                     measure_name = field.name
        #                     for field_attr in fields(attr):
        #                         attr_name = field_attr.name
        #                         attr_value = getattr(attr, attr_name)
        #                         if attr_value.size > 0:
        #                             metrics_to_log[f"Validation Measures/{measure_name}/{attr_name}"] = attr_value[i]

        #             run.log(metrics_to_log, step=int(training_metrics.max_timestep[i]))

        #             # metric for used for wandb sweep (optional)
        #             site_rpos = validation_metrics.euclidean_distance.site_rpos[i]
        #             site_rrotvec = validation_metrics.euclidean_distance.site_rpos[i]
        #             site_rvel = validation_metrics.euclidean_distance.site_rpos[i]
        #             run.log({"Metric for Sweep": site_rpos + site_rrotvec + site_rvel},
        #                     step=int(training_metrics.max_timestep[i]))

        # print(f"Time taken to log metrics: {time.time() - t_start}s")

        # run the environment with the trained agent to record video
        rec = os.getenv("RECORD_POLICY")
        if rec is None or rec:
            PPOJax.play_policy(env, agent_conf, agent_state, deterministic=True, n_steps=1000, n_envs=1, record=True, train_state_seed=0)
            video_file = env.video_file_path
            run.log({"Agent Video": wandb.Video(video_file)})

        wandb.finish()

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    experiment()
