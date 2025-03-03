import math

import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments.classic_control import CartPole
from gymnax.environments.spaces import Box, Discrete
from jax_loop_utils.metric_writers import MemoryWriter

import earl.agents.r2d2.networks as r2d2_networks
from earl.agents.r2d2.r2d2 import R2D2, R2D2Config
from earl.core import env_info_from_gymnax
from earl.environment_loop.gymnax_loop import GymnaxLoop
from earl.metric_key import MetricKey


def test_learns_cartpole():
  # NOTE: this test is pretty fragile. I had to search for good hyperparameters to get it to learn.
  # I think if it were run for many more cycles it would learn, but that's not appropriate for CI.
  env = CartPole()
  env_params = env.default_params
  action_space = env.action_space(env_params)
  assert isinstance(action_space, Discrete), action_space
  num_actions = int(action_space.n)
  observation_space = env.observation_space(env_params)
  assert isinstance(observation_space, Box), observation_space

  hidden_size = 32
  key = jax.random.PRNGKey(1)
  networks_key, loop_key, agent_key = jax.random.split(key, 3)
  networks = r2d2_networks.make_networks_mlp(
    num_actions=num_actions,
    input_size=int(math.prod(observation_space.shape)),
    dtype=jnp.float32,
    hidden_size=hidden_size,
    use_lstm=True,
    key=networks_key,
  )

  num_envs = 512
  burn_in = 10
  steps_per_cycle = 80 + burn_in
  num_cycles = 1000
  env_info = env_info_from_gymnax(env, env_params, num_envs)
  num_off_policy_optims_per_cycle = 1
  config = R2D2Config(
    epsilon_greedy_schedule_args=dict(
      init_value=0.5, end_value=0.01, transition_steps=steps_per_cycle * num_cycles
    ),
    q_learning_n_steps=2,
    debug=False,
    buffer_capacity=steps_per_cycle * 10,
    num_envs_per_learner=num_envs,
    replay_seq_length=steps_per_cycle,
    burn_in=burn_in,
    value_rescaling_epsilon=0.0,
    num_off_policy_optims_per_cycle=num_off_policy_optims_per_cycle,
    gradient_clipping_max_delta=1.0,
    learning_rate_schedule_name="cosine_onecycle_schedule",
    learning_rate_schedule_args=dict(
      transition_steps=steps_per_cycle * num_cycles // 2,
      peak_value=2e-4,
    ),
    target_update_step_size=0.00,
    target_update_period=100,
  )
  agent = R2D2(env_info, config)
  memory_writer = MemoryWriter()
  # For tweaking of hyperparameters, you can use the mlflow writer
  # and view metrics with `uv run --with mlflow mlflow server`
  # metric_writer = MultiWriter(
  #   (memory_writer, MlflowMetricWriter(experiment_name=env.name)),
  # )
  metric_writer = memory_writer
  loop = GymnaxLoop(env, env.default_params, agent, num_envs, loop_key, metric_writer=metric_writer)
  agent_state = agent.new_state(networks, agent_key)
  _ = loop.run(agent_state, num_cycles, steps_per_cycle)
  del agent_state
  metrics = memory_writer.scalars
  metric_writer.close()

  episode_lengths = np.array(
    [step_metrics[MetricKey.COMPLETE_EPISODE_LENGTH_MEAN] for step_metrics in metrics.values()]
  )

  assert len(metrics) == num_cycles
  # Due to auto-resets, the reward is always constant, but it's a survival task
  # so longer episodes are better.
  mean_over_cycles = 20
  first_mean = float(np.mean(episode_lengths[:mean_over_cycles]))
  assert first_mean > 0
  last_mean = float(np.mean(episode_lengths[-mean_over_cycles:]))
  assert last_mean > 1.4 * first_mean
