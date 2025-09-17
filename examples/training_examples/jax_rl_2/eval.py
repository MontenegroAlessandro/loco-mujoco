# td3_eval.py

import os
import argparse
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import TD3Jax # IMPORT TD3 INSTEAD OF PPO
from omegaconf import OmegaConf

os.environ['XLA_FLAGS'] = ('--xla_gpu_triton_gemm_any=True ')

# Set up argument parser
parser = argparse.ArgumentParser(description='Run evaluation with TD3Jax.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
args = parser.parse_args()

# Use the path from command line arguments
path = args.path
agent_conf, agent_state = TD3Jax.load_agent(path)
config = agent_conf.config

# Get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

# Create env
OmegaConf.set_struct(config, False)  # Allow modifications
config.experiment.env_params["headless"] = False
env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

# Run eval
TD3Jax.play_policy(env, agent_conf, agent_state, deterministic=True, n_steps=1000, n_envs=1, record=True)