import jax
import jax.numpy as jnp
from gymnax.environments.misc import pong
from gymnax.environments.spaces import Box, Discrete
from jax_loop_utils.metric_writers.mlflow import MlflowMetricWriter

import earl.agents.r2d2.networks as r2d2_networks
from earl.agents.r2d2.r2d2 import R2D2, R2D2Config
from earl.core import env_info_from_gymnax
from earl.environment_loop.gymnax_loop import GymnaxLoop

if __name__ == "__main__":
  env = pong.Pong()
  env_params = env.default_params
  action_space = env.action_space(env_params)
  assert isinstance(action_space, Discrete), action_space
  num_actions = int(action_space.n)
  observation_space = env.observation_space(env_params)
  assert isinstance(observation_space, Box), observation_space
  obs_shape = observation_space.shape

  hidden_size = 512
  key = jax.random.PRNGKey(0)
  networks_key, loop_key, agent_key = jax.random.split(key, 3)
  networks = r2d2_networks.make_networks_resnet(
    num_actions=num_actions,
    in_channels=obs_shape[-1],
    dtype=jnp.float32,
    hidden_size=hidden_size,
    key=networks_key,
    input_size=(observation_space.shape[0], observation_space.shape[1]),
    channel_last=True,
  )
  num_envs = 64
  steps_per_cycle = 80
  env_info = env_info_from_gymnax(env, env_params, num_envs)
  config = R2D2Config(
    num_envs_per_learner=num_envs,
    replay_seq_length=steps_per_cycle,
    burn_in=40,
  )
  agent = R2D2(env_info, config)
  agent_state = agent.new_state(networks, agent_key)

  # TODO: use tensorboard writer
  metric_writer = MlflowMetricWriter(experiment_name=env.name)
  gpus = jax.devices("gpu")
  loop = GymnaxLoop(
    env,
    env_params,
    agent,
    num_envs,
    loop_key,
    metric_writer=metric_writer,
    devices=gpus,
  )

  num_cycles = 2000
  loop_state = loop.run(agent_state, num_cycles, steps_per_cycle)
  del agent_state
