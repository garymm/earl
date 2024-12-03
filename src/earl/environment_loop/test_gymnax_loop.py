import dataclasses
from unittest import mock

import gymnax
import gymnax.environments.spaces
import jax
import jax.numpy as jnp
import pytest

from research.earl.agents.uniform_random_agent import UniformRandom
from research.earl.core import EnvStep, env_info_from_gymnax
from research.earl.environment_loop.gymnax_loop import ConflictingMetricError, GymnaxLoop, MetricKey, State
from research.earl.logging import base
from research.utils.prng import keygen


@pytest.mark.parametrize(("inference", "num_off_policy_updates"), [(True, 0), (False, 0), (False, 2)])
# setting default device speeds things up a little, but running without cuda enabled jaxlib is even faster
@jax.default_device(jax.devices("cpu")[0])
def test_gymnax_loop(inference: bool, num_off_policy_updates: int):
    env, env_params = gymnax.make("CartPole-v1")
    networks = None
    num_envs = 2
    key_gen = keygen(jax.random.PRNGKey(0))
    agent = UniformRandom(env.action_space().sample, num_off_policy_updates)
    loop = GymnaxLoop(env, env_params, agent, num_envs, next(key_gen), inference=inference)
    num_cycles = 2
    steps_per_cycle = 10
    env_info = env_info_from_gymnax(env, env_params, num_envs)
    agent_state = agent.new_state(networks, env_info, next(key_gen))
    result, metrics = loop.run(agent_state, num_cycles, steps_per_cycle)
    del agent_state
    assert result.agent_state.step.t == num_cycles * steps_per_cycle
    assert len(metrics[MetricKey.DURATION_SEC]) == num_cycles
    if inference:
        assert result.agent_state.opt.opt_count == 0
    else:
        assert len(metrics[MetricKey.LOSS]) == num_cycles
        assert len(metrics[agent._prng_metric_key]) == num_cycles
        assert metrics[agent._prng_metric_key][0] != metrics[agent._prng_metric_key][1]
        expected_opt_count = num_cycles * (num_off_policy_updates or 1)
        assert result.agent_state.opt.opt_count == expected_opt_count

    assert env.num_actions > 0
    for i in range(env.num_actions):
        num_actions_i = metrics[f"action_counts/{i}"]
        assert len(num_actions_i) == num_cycles
        # technically this could fail due to chance but it's very unlikely
        assert sum(num_actions_i) > 0


def test_run_with_state():
    env, env_params = gymnax.make("CartPole-v1")
    num_envs = 2
    obs, env_state = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), num_envs))
    key_gen = keygen(jax.random.PRNGKey(0))
    agent = UniformRandom(env.action_space().sample, 1)
    env.reset = mock.Mock(spec=env.reset)
    loop = GymnaxLoop(env, env_params, agent, num_envs, next(key_gen))
    agent_state = agent.new_state(None, env_info_from_gymnax(env, env_params, num_envs), jax.random.PRNGKey(0))
    initial_step_num = 2
    prev_action = jax.vmap(env.action_space(None).sample)(jax.random.split(jax.random.PRNGKey(0), num_envs))
    assert isinstance(prev_action, jax.Array)
    state = State(
        agent_state,
        env_state,
        EnvStep(
            new_episode=jnp.zeros((num_envs,), dtype=jnp.bool),
            obs=obs,
            prev_action=prev_action,
            reward=jnp.zeros((num_envs,)),
        ),
        step_num=initial_step_num,
    )
    steps_per_cycle = 10
    result, metrics = loop.run(state, 1, steps_per_cycle)
    assert metrics[MetricKey.STEP_NUM] == [initial_step_num + steps_per_cycle]
    assert env.reset.call_count == 0


def test_bad_args():
    num_envs = 2
    env, env_params = gymnax.make("CartPole-v1")
    agent = UniformRandom(env.action_space().sample, 0)
    loop = GymnaxLoop(env, env_params, agent, num_envs, jax.random.PRNGKey(0))
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
    agent = UniformRandom(env.action_space().sample, 0)
    agent = dataclasses.replace(agent, _prng_metric_key=MetricKey.DURATION_SEC)

    loop = GymnaxLoop(env, env_params, agent, num_envs, next(key_gen))
    num_cycles = 1
    steps_per_cycle = 1
    env_info = env_info_from_gymnax(env, env_params, num_envs)
    agent_state = agent.new_state(networks, env_info, jax.random.PRNGKey(0))
    with pytest.raises(ConflictingMetricError):
        loop.run(agent_state, num_cycles, steps_per_cycle)


class _AppendLogger(base.MetricLogger):
    def __init__(self):
        super().__init__()
        self._metrics = []

    def write(self, metrics: base.Metrics):
        self._metrics.append(metrics)

    def _close(self):
        pass


# setting default device speeds things up a little, but running without cuda enabled jaxlib is even faster
@jax.default_device(jax.devices("cpu")[0])
def test_logs():
    env, env_params = gymnax.make("CartPole-v1")
    networks = None
    num_envs = 2
    key_gen = keygen(jax.random.PRNGKey(0))
    agent = UniformRandom(env.action_space().sample, 0)
    logger = _AppendLogger()
    inference = True
    loop = GymnaxLoop(env, env_params, agent, num_envs, next(key_gen), logger=logger, inference=inference)
    num_cycles = 2
    steps_per_cycle = 10
    env_info = env_info_from_gymnax(env, env_params, num_envs)
    agent_state = agent.new_state(networks, env_info, jax.random.PRNGKey(0))
    assert not agent_state.inference
    result, metrics = loop.run(agent_state, num_cycles, steps_per_cycle)
    assert result.agent_state.inference, "we set loop.inference, but agent_state.inference is False!"
    assert metrics
    for k in metrics:
        returned_values = metrics[k]
        logged_values = [m[k] for m in logger._metrics]
        assert returned_values == logged_values
    assert metrics[MetricKey.STEP_NUM] == list(
        range(steps_per_cycle, num_cycles * steps_per_cycle + 1, steps_per_cycle)
    )
    logger.close()


def test_continuous_action_space():
    env, env_params = gymnax.make("Swimmer-misc")
    networks = None
    num_envs = 2
    key_gen = keygen(jax.random.PRNGKey(0))
    action_space = env.action_space()
    assert isinstance(action_space, gymnax.environments.spaces.Box)
    assert isinstance(action_space.low, jax.Array)
    assert isinstance(action_space.high, jax.Array)
    agent = UniformRandom(action_space.sample, 0)
    loop = GymnaxLoop(env, env_params, agent, num_envs, next(key_gen), inference=True)
    num_cycles = 1
    steps_per_cycle = 1
    env_info = env_info_from_gymnax(env, env_params, num_envs)
    agent_state = agent.new_state(networks, env_info, jax.random.PRNGKey(0))
    _, metrics = loop.run(agent_state, num_cycles, steps_per_cycle)
    for k in metrics:
        assert not k.startswith("action_counts_")


def test_observe_cycle():
    env, env_params = gymnax.make("Swimmer-misc")
    networks = None
    num_envs = 2
    key_gen = keygen(jax.random.PRNGKey(0))
    agent = UniformRandom(env.action_space().sample, 0)
    num_cycles = 2
    steps_per_cycle = 3
    env_info = env_info_from_gymnax(env, env_params, num_envs)
    agent_state = agent.new_state(networks, env_info, jax.random.PRNGKey(0))

    def observe_cycle(cycle_result: base.CycleResult) -> base.Metrics:
        assert cycle_result.trajectory.obs.shape[0] == num_envs
        assert cycle_result.trajectory.obs.shape[1] == steps_per_cycle
        assert "discount" in cycle_result.step_infos

        return {"ran": True}

    loop = GymnaxLoop(env, env_params, agent, num_envs, next(key_gen), inference=True, observe_cycle=observe_cycle)
    _, metrics = loop.run(agent_state, num_cycles, steps_per_cycle)
    for v in metrics["ran"]:
        assert v
