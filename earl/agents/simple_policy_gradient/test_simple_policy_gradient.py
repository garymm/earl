import jax
import numpy as np
import optax
from gymnax.environments import CartPole
from jax_loop_utils.metric_writers.memory_writer import MemoryWriter

from earl.agents.simple_policy_gradient.simple_policy_gradient import (
  Config,
  SimplePolicyGradient,
  make_networks,
)
from earl.core import env_info_from_gymnax
from earl.environment_loop.gymnax_loop import GymnaxLoop, MetricKey


def test_learns_cart_pole():
  env = CartPole()
  num_envs = 500
  env_info = env_info_from_gymnax(env, env.default_params, num_envs)
  steps_per_cycle = 80
  num_cycles = 50

  config = Config(max_actor_state_history=steps_per_cycle, optimizer=optax.adam(5e-3))
  agent = SimplePolicyGradient(env_info, config)
  (input_shape,) = env.obs_shape

  nets_key, loop_key, agent_key = jax.random.split(jax.random.PRNGKey(0), 3)

  networks = make_networks([input_shape, 32, env.num_actions], nets_key)
  metric_writer = MemoryWriter()
  loop = GymnaxLoop(env, env.default_params, agent, num_envs, loop_key, metric_writer=metric_writer)
  agent_state = agent.new_state(networks, agent_key)
  agent_state = loop.run(agent_state, num_cycles, steps_per_cycle)
  assert agent_state is not None
  metrics = metric_writer.scalars
  assert len(metrics) == num_cycles
  episode_lengths = np.array(
    [step_metrics[MetricKey.COMPLETE_EPISODE_LENGTH_MEAN] for step_metrics in metrics.values()]
  )
  # Due to auto-resets, the reward is always constant, but it's a survival task
  # so longer episodes are better.
  first_five_mean = float(np.mean(episode_lengths[:5]))
  assert first_five_mean > 0
  last_five_mean = float(np.mean(episode_lengths[-5:]))
  assert last_five_mean > 1.5 * first_five_mean
