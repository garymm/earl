from typing import Any

import gymnax
import gymnax.environments.spaces
import jax
import pytest
from jax_loop_utils.metric_writers import MemoryWriter
from jax_loop_utils.metric_writers.noop_writer import NoOpWriter

from earl.agents.random_agent.random_agent import RandomAgent
from earl.core import ConflictingMetricError, EnvStep, Metrics, env_info_from_gymnax
from earl.environment_loop.gymnax_loop import GymnaxLoop
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
def test_gymnax_loop(inference: bool, num_off_policy_updates: int):
  env, env_params = gymnax.make("CartPole-v1")
  networks = None
  num_envs = 2
  key_gen = keygen(jax.random.PRNGKey(0))
  env_info = env_info_from_gymnax(env, env_params, num_envs)
  agent = RandomAgent(env_info, env.action_space().sample, num_off_policy_updates)
  metric_writer = MemoryWriter()
  loop = GymnaxLoop(
    env,
    env_params,
    agent,
    num_envs,
    next(key_gen),
    metric_writer=metric_writer,
    actor_only=inference,
  )
  num_cycles = 2
  steps_per_cycle = 10
  agent_state = agent.new_state(networks, next(key_gen))
  result = loop.run(agent_state, num_cycles, steps_per_cycle)
  del agent_state
  assert result.agent_state.actor.t == num_cycles * steps_per_cycle
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
    for i in range(env.num_actions):
      action_count_i = metrics_for_step[f"action_counts/{i}"]
      action_count_sum += action_count_i
    assert action_count_sum > 0
    if not inference:
      assert MetricKey.LOSS in metrics_for_step
  if inference:
    assert result.agent_state.opt.opt_count == 0
  else:
    assert first_step_num is not None
    assert last_step_num is not None
    expected_opt_count = num_cycles * (num_off_policy_updates or 1)
    assert result.agent_state.opt.opt_count == expected_opt_count

  assert env.num_actions > 0


def test_bad_args():
  num_envs = 2
  env, env_params = gymnax.make("CartPole-v1")
  env_info = env_info_from_gymnax(env, env_params, num_envs)
  agent = RandomAgent(env_info, env.action_space().sample, 0)
  loop = GymnaxLoop(
    env, env_params, agent, num_envs, jax.random.PRNGKey(0), metric_writer=NoOpWriter()
  )
  agent_state = agent.new_state(None, jax.random.PRNGKey(0))
  with pytest.raises(ValueError, match="num_cycles"):
    loop.run(agent_state, 0, 10)
  with pytest.raises(ValueError, match="steps_per_cycle"):
    loop.run(agent_state, 10, 0)


def test_bad_metric_key():
  env, env_params = gymnax.make("CartPole-v1")
  networks = None
  num_envs = 2
  env_info = env_info_from_gymnax(env, env_params, num_envs)
  key_gen = keygen(jax.random.PRNGKey(0))

  def observe_cycle(trajectory: EnvStep, step_infos: dict[Any, Any]) -> Metrics:
    return {MetricKey.DURATION_SEC: 1}

  agent = RandomAgent(env_info, env.action_space().sample, 0)

  loop = GymnaxLoop(
    env,
    env_params,
    agent,
    num_envs,
    next(key_gen),
    metric_writer=NoOpWriter(),
    observe_cycle=observe_cycle,
  )
  num_cycles = 1
  steps_per_cycle = 1
  agent_state = agent.new_state(networks, jax.random.PRNGKey(0))
  with pytest.raises(ConflictingMetricError):
    loop.run(agent_state, num_cycles, steps_per_cycle)


def test_continuous_action_space():
  env, env_params = gymnax.make("Swimmer-misc")
  networks = None
  num_envs = 2
  env_info = env_info_from_gymnax(env, env_params, num_envs)
  key_gen = keygen(jax.random.PRNGKey(0))
  action_space = env.action_space()
  assert isinstance(action_space, gymnax.environments.spaces.Box)
  assert isinstance(action_space.low, jax.Array)
  assert isinstance(action_space.high, jax.Array)
  agent = RandomAgent(env_info, action_space.sample, 0)
  metric_writer = MemoryWriter()
  loop = GymnaxLoop(
    env, env_params, agent, num_envs, next(key_gen), metric_writer=metric_writer, actor_only=True
  )
  num_cycles = 1
  steps_per_cycle = 1
  agent_state = agent.new_state(networks, jax.random.PRNGKey(0))
  loop.run(agent_state, num_cycles, steps_per_cycle)
  for _, v in metric_writer.scalars.items():
    for k in v:
      assert not k.startswith("action_counts_")


def test_observe_cycle():
  env, env_params = gymnax.make("Swimmer-misc")
  networks = None
  num_envs = 2
  env_info = env_info_from_gymnax(env, env_params, num_envs)
  key_gen = keygen(jax.random.PRNGKey(0))
  agent = RandomAgent(env_info, env.action_space().sample, 0)
  num_cycles = 2
  steps_per_cycle = 3
  agent_state = agent.new_state(networks, jax.random.PRNGKey(0))

  def observe_cycle(trajectory: EnvStep, step_infos: dict) -> Metrics:
    assert trajectory.obs.shape[0] == num_envs
    assert trajectory.obs.shape[1] == steps_per_cycle
    assert "discount" in step_infos
    return {"ran": True}

  metric_writer = MemoryWriter()
  loop = GymnaxLoop(
    env,
    env_params,
    agent,
    num_envs,
    next(key_gen),
    metric_writer=metric_writer,
    actor_only=True,
    observe_cycle=observe_cycle,
  )
  loop.run(agent_state, num_cycles, steps_per_cycle)
  for _, v in metric_writer.scalars.items():
    assert v.get("ran", False)
