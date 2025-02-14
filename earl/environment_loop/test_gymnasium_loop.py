import dataclasses
import time

import gymnax
import gymnax.environments.spaces
import jax
import optax
import pytest
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from gymnax.environments.spaces import Box, Discrete
from jax_loop_utils.metric_writers.memory_writer import MemoryWriter

from earl.agents.random_agent.random_agent import RandomAgent
from earl.agents.simple_policy_gradient import simple_policy_gradient
from earl.core import ConflictingMetricError, Metrics, env_info_from_gymnasium
from earl.environment_loop import CycleResult
from earl.environment_loop.gymnasium_loop import GymnasiumLoop
from earl.metric_key import MetricKey
from earl.utils.prng import keygen


@pytest.mark.parametrize(
  "inference,num_off_policy_updates",
  [
    (True, 0),
    (False, 0),
    (False, 2),
  ],
)
# setting default device speeds up a little, but running without cuda enabled jaxlib is faster
@jax.default_device(jax.devices("cpu")[0])
def test_gymnasium_loop(inference: bool, num_off_policy_updates: int):
  num_envs = 2
  env = CartPoleEnv()
  env_info = env_info_from_gymnasium(env, num_envs)
  networks = None
  key_gen = keygen(jax.random.PRNGKey(0))
  agent = RandomAgent(env_info.action_space.sample, num_off_policy_updates)
  metric_writer = MemoryWriter()
  if not inference and not num_off_policy_updates:
    with pytest.raises(ValueError, match="On-policy training is not supported in GymnasiumLoop."):
      loop = GymnasiumLoop(
        env, agent, num_envs, next(key_gen), metric_writer=metric_writer, actor_only=inference
      )
    return

  loop = GymnasiumLoop(
    env,
    agent,
    num_envs,
    next(key_gen),
    metric_writer=metric_writer,
    actor_only=inference,
  )
  num_cycles = 2
  steps_per_cycle = 10
  agent_state = agent.new_state(networks, env_info, next(key_gen))
  result = loop.run(agent_state, num_cycles, steps_per_cycle)
  del agent_state
  metrics = metric_writer.scalars
  assert len(metrics) == num_cycles
  first_step_num, last_step_num = None, None
  for step_num, metrics_for_step in metrics.items():
    if first_step_num is None:
      first_step_num = step_num
    last_step_num = step_num
    assert MetricKey.DURATION_SEC in metrics_for_step
    assert MetricKey.REWARD_SUM in metrics_for_step
    assert MetricKey.REWARD_MEAN in metrics_for_step
    assert MetricKey.TOTAL_DONES in metrics_for_step
    action_count_sum = 0
    assert isinstance(env_info.action_space, Discrete)
    for i in range(env_info.action_space.n):
      action_count_i = metrics_for_step[f"action_counts/{i}"]
      action_count_sum += action_count_i
    assert action_count_sum > 0
    if not inference:
      assert MetricKey.LOSS in metrics_for_step
      assert agent._prng_metric_key in metrics_for_step
  if inference:
    assert result.agent_state.opt.opt_count == 0
  else:
    assert first_step_num is not None
    assert last_step_num is not None
    assert (
      metrics[first_step_num][agent._prng_metric_key]
      != metrics[last_step_num][agent._prng_metric_key]
    )
    expected_opt_count = num_cycles * (num_off_policy_updates or 1)
    assert result.agent_state.opt.opt_count == expected_opt_count

  assert isinstance(env_info.action_space, Discrete)
  assert env_info.action_space.n > 0
  assert all(not env.closed for env in loop._env_for_actor_thread)
  loop.close()
  assert all(env.closed for env in loop._env_for_actor_thread)


def test_bad_args():
  num_envs = 2
  env = CartPoleEnv()
  env_info = env_info_from_gymnasium(env, num_envs)
  agent = RandomAgent(env_info.action_space.sample, 0)
  metric_writer = MemoryWriter()
  loop = GymnasiumLoop(
    env, agent, num_envs, jax.random.PRNGKey(0), metric_writer=metric_writer, actor_only=True
  )
  agent_state = agent.new_state(None, env_info, jax.random.PRNGKey(0))
  with pytest.raises(ValueError, match="num_cycles"):
    loop.run(agent_state, 0, 10)
  with pytest.raises(ValueError, match="steps_per_cycle"):
    loop.run(agent_state, 10, 0)


def test_bad_metric_key():
  networks = None
  num_envs = 2
  env = CartPoleEnv()
  env_info = env_info_from_gymnasium(env, num_envs)
  key_gen = keygen(jax.random.PRNGKey(0))
  # make the agent return a metric with a key that conflicts with a built-in metric.
  agent = RandomAgent(env_info.action_space.sample, 1)
  agent = dataclasses.replace(agent, _prng_metric_key=MetricKey.DURATION_SEC)

  metric_writer = MemoryWriter()
  loop = GymnasiumLoop(env, agent, num_envs, next(key_gen), metric_writer=metric_writer)
  num_cycles = 1
  steps_per_cycle = 1
  agent_state = agent.new_state(networks, env_info, jax.random.PRNGKey(0))
  with pytest.raises(ConflictingMetricError):
    loop.run(agent_state, num_cycles, steps_per_cycle)


def test_continuous_action_space():
  num_envs = 2
  env = PendulumEnv()
  env_info = env_info_from_gymnasium(env, num_envs)
  networks = None
  key_gen = keygen(jax.random.PRNGKey(0))
  action_space = env_info.action_space
  assert isinstance(action_space, gymnax.environments.spaces.Box)
  assert isinstance(action_space.low, jax.Array)
  assert isinstance(action_space.high, jax.Array)
  agent = RandomAgent(action_space.sample, 0)
  metric_writer = MemoryWriter()
  loop = GymnasiumLoop(
    env, agent, num_envs, next(key_gen), metric_writer=metric_writer, actor_only=True
  )
  num_cycles = 1
  steps_per_cycle = 1
  agent_state = agent.new_state(networks, env_info, jax.random.PRNGKey(0))
  loop.run(agent_state, num_cycles, steps_per_cycle)
  for _, v in metric_writer.scalars.items():
    for k in v:
      assert not k.startswith("action_counts_")


def test_observe_cycle():
  num_envs = 2
  env = PendulumEnv()
  env_info = env_info_from_gymnasium(env, num_envs)
  networks = None
  key_gen = keygen(jax.random.PRNGKey(0))
  agent = RandomAgent(env_info.action_space.sample, 0)
  metric_writer = MemoryWriter()
  num_cycles = 2
  steps_per_cycle = 3

  def observe_cycle(cycle_result: CycleResult) -> Metrics:
    assert cycle_result.trajectory.obs.shape[0] == num_envs
    assert cycle_result.trajectory.obs.shape[1] == steps_per_cycle
    return {"ran": True}

  loop = GymnasiumLoop(
    env,
    agent,
    num_envs,
    next(key_gen),
    metric_writer=metric_writer,
    actor_only=True,
    observe_cycle=observe_cycle,
  )
  agent_state = agent.new_state(networks, env_info, jax.random.PRNGKey(0))
  loop.run(agent_state, num_cycles, steps_per_cycle)
  for _, v in metric_writer.scalars.items():
    assert v.get("ran", False)


def test_benchmark_gymnasium_inference():
  num_envs = 16
  env = CartPoleEnv()
  env_info = env_info_from_gymnasium(env, num_envs)
  networks = None
  key_gen = keygen(jax.random.PRNGKey(0))
  agent = simple_policy_gradient.SimplePolicyGradient(
    simple_policy_gradient.Config(
      max_actor_state_history=100,
      optimizer=optax.adam(1e-3),
      discount=0.99,
    )
  )
  metric_writer = MemoryWriter()
  loop = GymnasiumLoop(
    env,
    agent,
    num_envs,
    next(key_gen),
    metric_writer=metric_writer,
    actor_only=True,
    vectorization_mode="async",
  )
  assert isinstance(env_info.observation_space, Box)
  assert isinstance(env_info.action_space, Discrete)
  networks = simple_policy_gradient.make_networks(
    [env_info.observation_space.shape[0], 200, 200, 200, 200, 200, env_info.action_space.n],
    jax.random.PRNGKey(0),
  )
  agent_state = agent.new_state(networks, env_info, jax.random.PRNGKey(0))
  steps_per_cycle = 100
  result = loop.run(agent_state, 1, steps_per_cycle)  # warmup
  assert result.agent_state.actor.t == steps_per_cycle

  start = time.monotonic()
  num_cycles = 2
  loop.run(result, num_cycles, steps_per_cycle)
  end = time.monotonic()
  print(f"Time taken: {end - start} seconds")
  steps_per_second = num_cycles * steps_per_cycle * num_envs / (end - start)
  print(f"Steps per second: {steps_per_second}")
