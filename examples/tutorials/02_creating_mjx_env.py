import os
import jax
import time

from loco_mujoco import ImitationFactory, RLFactory

import omegaconf


# can increase the speed by ~30% on some GPUs
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True ')

# create env
# env = ImitationFactory.make("MjxUnitreeG1", default_dataset_conf=dict(task="stepinplace1"))

# load config
config = omegaconf.OmegaConf.load("/home/olegkaidanov/polimi_lmj/loco-mujoco/examples/training_examples/jax_rl/conf2.yaml")
env = RLFactory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

# create keys
key = jax.random.key(0)
n_envs = 100
keys = jax.random.split(key, n_envs + 1)
key, env_keys = keys[0], keys[1:]

# jit and vmap all functions needed
rng_reset = jax.jit(jax.vmap(env.mjx_reset))
rng_step = jax.jit(jax.vmap(env.mjx_step))
rng_sample_uni_action = jax.jit(jax.vmap(env.sample_action_space))

# reset env
state = rng_reset(env_keys)

step = 0
previous_time = time.time()
LOGGING_FREQUENCY = 100000
i = 0
while i < 100000:

    # step
    keys = jax.random.split(key, n_envs + 1)
    key, action_keys = keys[0], keys[1:]
    action = rng_sample_uni_action(action_keys) * 0.0
    state = rng_step(state, action)

    # parallel render
    env.mjx_render(state)

    step += n_envs

    # log speed (disable rendering for accurate speed measurement)
    if step % LOGGING_FREQUENCY == 0:
        current_time = time.time()
        print(f"{int(LOGGING_FREQUENCY / (current_time - previous_time))} steps per second.")
        previous_time = current_time

    i+=1

