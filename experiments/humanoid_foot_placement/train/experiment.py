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
            '--xla_gpu_autotune_level=1 '
            '--xla_gpu_force_compilation_parallelism=1 '

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
        
        # setup wandb (uses the WANDB_API_KEY env var, or a cached `wandb login`)
        wandb.login()
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
        if "resume_from_path" in config.experiment and config.experiment["resume_from_path"] is not None:
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


        # run the environment with the trained agent to record video
        rec = os.getenv("RECORD_POLICY")
        if rec is None or rec == "true":
            PPOJax.play_policy(env, agent_conf, agent_state, deterministic=True, n_steps=1000, n_envs=1, record=True, train_state_seed=0)
            video_file = env.video_file_path
            run.log({"Agent Video": wandb.Video(video_file)})

        wandb.finish()

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    experiment()
