import ale_py
import gymnasium
import jax
import jax.numpy as jnp
from jax_loop_utils.metric_writers.mlflow import MlflowMetricWriter

import earl.agents.r2d2.networks as r2d2_networks
from earl.agents.r2d2.r2d2 import R2D2, R2D2Config
from earl.core import env_info_from_gymnasium
from earl.environment_loop.gymnasium_loop import GymnasiumLoop

gymnasium.register_envs(ale_py)  # suppress unused import warning


env_name = "PongNoFrameskip-v4"

if __name__ == "__main__":
  env = gymnasium.make(env_name)
  env = gymnasium.wrappers.AtariPreprocessing(env, noop_max=0)
  stack_size = 4
  env = gymnasium.wrappers.FrameStackObservation(env, stack_size=stack_size)
  assert isinstance(env.action_space, gymnasium.spaces.Discrete)
  num_actions = int(env.action_space.n)

  hidden_size = 512
  key = jax.random.PRNGKey(0)
  networks_key, loop_key, agent_key = jax.random.split(key, 3)
  networks = r2d2_networks.make_networks_resnet(
    num_actions=num_actions,
    in_channels=stack_size,
    dtype=jnp.float32,
    hidden_size=hidden_size,
    key=networks_key,
  )
  num_envs = 16
  steps_per_cycle = 80
  env_info = env_info_from_gymnasium(env, num_envs)
  config = R2D2Config(
    num_envs_per_learner=num_envs,
    replay_seq_length=steps_per_cycle,
    burn_in=40,
  )
  agent = R2D2(env_info, config)
  agent_state = agent.new_state(networks, agent_key)

  # TODO: use tensorboard writer
  metric_writer = MlflowMetricWriter(experiment_name=env_name)
  gpus = jax.devices("gpu")
  loop = GymnasiumLoop(
    env,
    agent,
    num_envs,
    loop_key,
    metric_writer=metric_writer,
    # actor_devices=gpus[0:1],
    # learner_devices=gpus[1:2],
    vectorization_mode="sync",
  )

  num_cycles = 2000
  loop_state = loop.run(agent_state, num_cycles, steps_per_cycle)
  del agent_state
