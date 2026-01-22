# Libraries
import os, argparse, sys, wandb, hydra, traceback, time, jax
import jax.numpy as jnp
from dataclasses import fields
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.utils.metrics import QuantityContainer
from loco_mujoco.utils import MetricsHandler

job_timestamp_str = "EVAL: " + time.strftime("%Y-%m-%d/%H-%M-%S", time.localtime())

@hydra.main(version_base=None, config_path="./", config_name="conf_eval_old")
def experiment(config: DictConfig):

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
    
    # setup wandb
    if config.wandb.log:
        wandb.login()
        config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

        # name the run as time stamp + overrides
        wandb_run_name = f"{';'.join(str_overrides + [job_timestamp_str])}"

        run = wandb.init(
            entity="",
            project=config.wandb.project,
            name=wandb_run_name,
            config=config_dict
        )
    else:
        run = None

    # get task factory
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

    # create env
    config.experiment.env_params["headless"] = False
    env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
    
    # env.create_observation_summary()

    # get initial agent configuration
    path = config.experiment.agent_path
    agent_conf, agent_state = PPOJax.load_agent(path)
    config = agent_conf.config

    # run the environment with the trained agent to record video
    PPOJax.play_policy(
        env, 
        agent_conf, 
        agent_state, 
        deterministic=True, 
        n_steps=1000, 
        n_envs=1, 
        record=True,
        train_state_seed=0
    )
    video_file = env.video_file_path
    if run is not None:
        run.log({"Agent Video": wandb.Video(video_file)})
        wandb.finish()

if __name__ == "__main__":
    experiment()