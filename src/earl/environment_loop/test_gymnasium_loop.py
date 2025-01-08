import dataclasses

import gymnax
import gymnax.environments.spaces
import jax
import pytest
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from gymnax.environments.spaces import Discrete
from jax_loop_utils.metric_writers.memory_writer import MemoryWriter

from research.earl.agents.uniform_random_agent import UniformRandom
from research.earl.core import ConflictingMetricError, Metrics, env_info_from_gymnasium
from research.earl.environment_loop import CycleResult
from research.earl.environment_loop.gymnasium_loop import GymnasiumLoop
from research.earl.metric_key import MetricKey
from research.utils.prng import keygen


@pytest.mark.parametrize(("inference", "num_off_policy_updates"), [(True, 0), (False, 0), (False, 2)])
# setting default device speeds things up a little, but running without cuda enabled jaxlib is even faster
@jax.default_device(jax.devices("cpu")[0])
def test_gymnasium_loop(inference: bool, num_off_policy_updates: int):
    num_envs = 2
    env = CartPoleEnv()
    env_info = env_info_from_gymnasium(env, num_envs)
    networks = None
    key_gen = keygen(jax.random.PRNGKey(0))
    agent = UniformRandom(env_info.action_space.sample, num_off_policy_updates)
    metric_writer = MemoryWriter()
    if not inference and not num_off_policy_updates:
        with pytest.raises(ValueError, match="On-policy training is not supported in GymnasiumLoop."):
            loop = GymnasiumLoop(env, agent, num_envs, next(key_gen), metric_writer=metric_writer, inference=inference)
        return

    loop = GymnasiumLoop(env, agent, num_envs, next(key_gen), metric_writer=metric_writer, inference=inference)
    num_cycles = 2
    steps_per_cycle = 10
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
        assert metrics[first_step_num][agent._prng_metric_key] != metrics[last_step_num][agent._prng_metric_key]
        expected_opt_count = num_cycles * (num_off_policy_updates or 1)
        assert result.agent_state.opt.opt_count == expected_opt_count

    assert isinstance(env_info.action_space, Discrete)
    assert env_info.action_space.n > 0
    assert not loop._env.closed
    loop.close()
    assert loop._env.closed


def test_bad_args():
    num_envs = 2
    env = CartPoleEnv()
    env_info = env_info_from_gymnasium(env, num_envs)
    agent = UniformRandom(env_info.action_space.sample, 0)
    metric_writer = MemoryWriter()
    loop = GymnasiumLoop(env, agent, num_envs, jax.random.PRNGKey(0), metric_writer=metric_writer, inference=True)
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
    agent = UniformRandom(env_info.action_space.sample, 1)
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
    agent = UniformRandom(action_space.sample, 0)
    metric_writer = MemoryWriter()
    loop = GymnasiumLoop(env, agent, num_envs, next(key_gen), metric_writer=metric_writer, inference=True)
    num_cycles = 1
    steps_per_cycle = 1
    agent_state = agent.new_state(networks, env_info, jax.random.PRNGKey(0))
    _ = loop.run(agent_state, num_cycles, steps_per_cycle)
    for _, v in metric_writer.scalars.items():
        for k in v:
            assert not k.startswith("action_counts_")


def test_observe_cycle():
    num_envs = 2
    env = PendulumEnv()
    env_info = env_info_from_gymnasium(env, num_envs)
    networks = None
    key_gen = keygen(jax.random.PRNGKey(0))
    agent = UniformRandom(env_info.action_space.sample, 0)
    metric_writer = MemoryWriter()

    def observe_cycle(cycle_result: CycleResult) -> Metrics:
        assert cycle_result.trajectory.obs.shape[0] == num_envs
        assert cycle_result.trajectory.obs.shape[1] == steps_per_cycle
        return {"ran": True}

    loop = GymnasiumLoop(
        env, agent, num_envs, next(key_gen), metric_writer=metric_writer, inference=True, observe_cycle=observe_cycle
    )
    num_cycles = 2
    steps_per_cycle = 3
    agent_state = agent.new_state(networks, env_info, jax.random.PRNGKey(0))

    _ = loop.run(agent_state, num_cycles, steps_per_cycle)
    for _, v in metric_writer.scalars.items():
        assert v.get("ran", False)
