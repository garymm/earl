import dataclasses
from unittest import mock

import gymnax
import gymnax.environments.spaces
import jax
import jax.numpy as jnp
import pytest
from jax_loop_utils.metric_writers import MemoryWriter
from jax_loop_utils.metric_writers.noop_writer import NoOpWriter

from research.earl.agents.random_agent.random_agent import RandomAgent
from research.earl.core import ConflictingMetricError, EnvStep, Metrics, env_info_from_gymnax
from research.earl.environment_loop import CycleResult
from research.earl.environment_loop.gymnax_loop import GymnaxLoop, MetricKey, State
from research.utils.prng import keygen


@pytest.mark.parametrize(("inference", "num_off_policy_updates"), [(True, 0), (False, 0), (False, 2)])
# setting default device speeds things up a little, but running without cuda enabled jaxlib is even faster
@jax.default_device(jax.devices("cpu")[0])
def test_gymnax_loop(inference: bool, num_off_policy_updates: int):
    env, env_params = gymnax.make("CartPole-v1")
    networks = None
    num_envs = 2
    key_gen = keygen(jax.random.PRNGKey(0))
    agent = RandomAgent(env.action_space().sample, num_off_policy_updates)
    metric_writer = MemoryWriter()
    loop = GymnaxLoop(env, env_params, agent, num_envs, next(key_gen), metric_writer=metric_writer, inference=inference)
    num_cycles = 2
    steps_per_cycle = 10
    env_info = env_info_from_gymnax(env, env_params, num_envs)
    agent_state = agent.new_state(networks, env_info, next(key_gen))
    result = loop.run(agent_state, num_cycles, steps_per_cycle)
    del agent_state
    assert result.agent_state.step.t == num_cycles * steps_per_cycle
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
            assert agent._prng_metric_key in metrics_for_step
    if inference:
        assert result.agent_state.opt.opt_count == 0
    else:
        assert first_step_num is not None
        assert last_step_num is not None
        assert metrics[first_step_num][agent._prng_metric_key] != metrics[last_step_num][agent._prng_metric_key]
        expected_opt_count = num_cycles * (num_off_policy_updates or 1)
        assert result.agent_state.opt.opt_count == expected_opt_count

    assert env.num_actions > 0


def test_run_with_state():
    env, env_params = gymnax.make("CartPole-v1")
    num_envs = 2
    obs, env_state = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), num_envs))
    key_gen = keygen(jax.random.PRNGKey(0))
    agent = RandomAgent(env.action_space().sample, 1)
    env.reset = mock.Mock(spec=env.reset)
    loop = GymnaxLoop(env, env_params, agent, num_envs, next(key_gen), metric_writer=NoOpWriter())
    agent_state = agent.new_state(None, env_info_from_gymnax(env, env_params, num_envs), jax.random.PRNGKey(0))
    initial_step_num = 2
    prev_action = jax.vmap(env.action_space(None).sample)(jax.random.split(jax.random.PRNGKey(0), num_envs))
    assert isinstance(prev_action, jax.Array)
    env_step = EnvStep(
        new_episode=jnp.zeros((num_envs,), dtype=jnp.bool),
        obs=obs,
        prev_action=prev_action,
        reward=jnp.zeros((num_envs,)),
    )
    state = State(
        agent_state=jax.device_put_replicated(agent_state, devices=jax.local_devices()),
        env_state=jax.device_put_replicated(env_state, devices=jax.local_devices()),
        env_step=jax.device_put_replicated(env_step, devices=jax.local_devices()),
        step_num=initial_step_num,
    )
    steps_per_cycle = 10
    _result = loop.run(state, 1, steps_per_cycle)
    assert env.reset.call_count == 0


def test_bad_args():
    num_envs = 2
    env, env_params = gymnax.make("CartPole-v1")
    agent = RandomAgent(env.action_space().sample, 0)
    loop = GymnaxLoop(env, env_params, agent, num_envs, jax.random.PRNGKey(0), metric_writer=NoOpWriter())
    env_info = env_info_from_gymnax(env, env_params, num_envs)
    agent_state = agent.new_state(None, env_info, jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="num_cycles"):
        loop.run(agent_state, 0, 10)
    with pytest.raises(ValueError, match="steps_per_cycle"):
        loop.run(agent_state, 10, 0)


def test_bad_metric_key():
    env, env_params = gymnax.make("CartPole-v1")
    networks = None
    num_envs = 2
    key_gen = keygen(jax.random.PRNGKey(0))
    # make the agent return a metric with a key that conflicts with a built-in metric.
    agent = RandomAgent(env.action_space().sample, 0)
    agent = dataclasses.replace(agent, _prng_metric_key=MetricKey.DURATION_SEC)

    loop = GymnaxLoop(env, env_params, agent, num_envs, next(key_gen), metric_writer=NoOpWriter())
    num_cycles = 1
    steps_per_cycle = 1
    env_info = env_info_from_gymnax(env, env_params, num_envs)
    agent_state = agent.new_state(networks, env_info, jax.random.PRNGKey(0))
    with pytest.raises(ConflictingMetricError):
        loop.run(agent_state, num_cycles, steps_per_cycle)


def test_continuous_action_space():
    env, env_params = gymnax.make("Swimmer-misc")
    networks = None
    num_envs = 2
    key_gen = keygen(jax.random.PRNGKey(0))
    action_space = env.action_space()
    assert isinstance(action_space, gymnax.environments.spaces.Box)
    assert isinstance(action_space.low, jax.Array)
    assert isinstance(action_space.high, jax.Array)
    agent = RandomAgent(action_space.sample, 0)
    metric_writer = MemoryWriter()
    loop = GymnaxLoop(env, env_params, agent, num_envs, next(key_gen), metric_writer=metric_writer, inference=True)
    num_cycles = 1
    steps_per_cycle = 1
    env_info = env_info_from_gymnax(env, env_params, num_envs)
    agent_state = agent.new_state(networks, env_info, jax.random.PRNGKey(0))
    loop.run(agent_state, num_cycles, steps_per_cycle)
    for _, v in metric_writer.scalars.items():
        for k in v:
            assert not k.startswith("action_counts_")


def test_observe_cycle():
    env, env_params = gymnax.make("Swimmer-misc")
    networks = None
    num_envs = 2
    key_gen = keygen(jax.random.PRNGKey(0))
    agent = RandomAgent(env.action_space().sample, 0)
    num_cycles = 2
    steps_per_cycle = 3
    env_info = env_info_from_gymnax(env, env_params, num_envs)
    agent_state = agent.new_state(networks, env_info, jax.random.PRNGKey(0))

    def observe_cycle(cycle_result: CycleResult) -> Metrics:
        assert cycle_result.trajectory.obs.shape[0] == num_envs
        assert cycle_result.trajectory.obs.shape[1] == steps_per_cycle
        assert "discount" in cycle_result.step_infos
        return {"ran": True}

    metric_writer = MemoryWriter()
    loop = GymnaxLoop(
        env,
        env_params,
        agent,
        num_envs,
        next(key_gen),
        metric_writer=metric_writer,
        inference=True,
        observe_cycle=observe_cycle,
    )
    loop.run(agent_state, num_cycles, steps_per_cycle)
    for _, v in metric_writer.scalars.items():
        assert v.get("ran", False)
