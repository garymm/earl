from typing import NamedTuple

import gymnax
import gymnax.environments.spaces
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import pytest
from jaxtyping import PRNGKeyArray, PyTree, Scalar

from research.earl.core import ActionAndStepState, Agent, EnvTimestep, Metrics
from research.earl.core import AgentState as CoreAgentState
from research.earl.environment_loop.gymnax_loop import ConflictingMetricError, GymnaxLoop, MetricKey
from research.earl.logging import base
from research.utils.prng import keygen


class StepState(NamedTuple):
    key: PRNGKeyArray
    t: jnp.ndarray
    update_count: jnp.ndarray


AgentState = CoreAgentState[None, None, StepState]


class UniformRandom(Agent[None, None, StepState]):
    """Agent that selects actions uniformly at random."""

    def __init__(self, action_space: gymnax.environments.spaces.Space, num_off_policy_updates: int):
        self._action_space = action_space
        self._num_off_policy_updates = num_off_policy_updates
        self._prng_metric_key = "prng"

    def _initial_state(self, nets: None, obs: PyTree, key: PRNGKeyArray) -> AgentState:
        return AgentState(
            nets=None,
            cycle=None,
            step=StepState(key, jnp.zeros((1,), dtype=jnp.uint32), jnp.zeros((1,), dtype=jnp.uint32)),
        )

    def _select_action(self, state: AgentState, env_timestep: EnvTimestep) -> ActionAndStepState:
        key, action_key = jax.random.split(state.step.key)
        num_envs = env_timestep.obs.shape[0]
        actions = jax.vmap(self._action_space.sample)(jax.random.split(action_key, num_envs))
        assert isinstance(actions, jnp.ndarray)
        return ActionAndStepState(actions, StepState(key, state.step.t + 1, state.step.update_count))

    def _partition_for_grad(self, nets: None) -> tuple[None, None]:
        return None, None

    def _loss(self, state: AgentState) -> tuple[Scalar, Metrics]:
        return jnp.array(0.0), {
            # metrics need to be scalars, so take elem 0.
            self._prng_metric_key: state.step.key[0],
        }

    def _update_from_grads(self, state: AgentState, nets_grads: PyTree) -> AgentState:
        return jdc.replace(state, step=StepState(state.step.key, state.step.t, state.step.update_count + 1))

    def num_off_policy_updates_per_cycle(self) -> int:
        return self._num_off_policy_updates


@pytest.mark.parametrize(("inference", "num_off_policy_updates"), [(True, 0), (False, 0), (False, 2)])
# setting default device speeds things up a little, but running without cuda enabled jaxlib is even faster
@jax.default_device(jax.devices("cpu")[0])
def test_gymnax_loop(inference: bool, num_off_policy_updates: int):
    env, env_params = gymnax.make("CartPole-v1")
    networks = None
    num_envs = 2
    key_gen = keygen(jax.random.PRNGKey(0))
    agent = UniformRandom(env.action_space(), num_off_policy_updates)
    loop = GymnaxLoop(env, env_params, agent, num_envs, next(key_gen), inference=inference)
    num_cycles = 2
    steps_per_cycle = 10
    agent_state = agent.initial_state(networks, loop.example_batched_obs(), next(key_gen))
    agent_state, metrics = loop.run(agent_state, num_cycles, steps_per_cycle)
    assert agent_state
    assert agent_state.step.t == num_cycles * steps_per_cycle
    assert len(metrics[MetricKey.DURATION_SEC]) == num_cycles
    for duration in metrics[MetricKey.DURATION_SEC]:
        assert duration > 0
    if inference:
        assert agent_state.step.update_count == 0
    else:
        assert len(metrics[MetricKey.LOSS]) == num_cycles
        assert len(metrics[agent._prng_metric_key]) == num_cycles
        assert metrics[agent._prng_metric_key][0] != metrics[agent._prng_metric_key][1]
        expected_update_count = num_cycles * (num_off_policy_updates or 1)
        assert agent_state.step.update_count == expected_update_count

    assert env.num_actions > 0
    for i in range(env.num_actions):
        num_actions_i = metrics[f"action_counts/{i}"]
        assert len(num_actions_i) == num_cycles
        # technically this could fail due to chance but it's very unlikely
        assert sum(num_actions_i) > 0


def test_bad_args():
    num_envs = 2
    agent = UniformRandom(gymnax.environments.spaces.Discrete(2), 0)
    env, env_params = gymnax.make("CartPole-v1")
    loop = GymnaxLoop(env, env_params, agent, num_envs, jax.random.PRNGKey(0))
    initial_state = agent.initial_state(None, loop.example_batched_obs(), jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="num_cycles"):
        loop.run(initial_state, 0, 10)
    with pytest.raises(ValueError, match="steps_per_cycle"):
        loop.run(initial_state, 10, 0)


def test_bad_metric_key():
    env, env_params = gymnax.make("CartPole-v1")
    networks = None
    num_envs = 2
    key_gen = keygen(jax.random.PRNGKey(0))
    agent = UniformRandom(env.action_space(), 0)
    # make the agent return a metric with a key that conflicts with a built-in metric.
    agent._prng_metric_key = MetricKey.DURATION_SEC
    loop = GymnaxLoop(env, env_params, agent, num_envs, next(key_gen))
    num_cycles = 1
    steps_per_cycle = 1
    agent_state = agent.initial_state(networks, loop.example_batched_obs(), next(key_gen))
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
    agent = UniformRandom(env.action_space(), 0)
    logger = _AppendLogger()
    inference = True
    loop = GymnaxLoop(env, env_params, agent, num_envs, next(key_gen), logger=logger, inference=inference)
    num_cycles = 2
    steps_per_cycle = 10
    agent_state = agent.initial_state(networks, loop.example_batched_obs(), next(key_gen))
    assert not agent_state.inference
    agent_state, metrics = loop.run(agent_state, num_cycles, steps_per_cycle)
    assert agent_state.inference, "we set loop.inference, but agent_state.inference is False!"
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
    agent = UniformRandom(env.action_space(), 0)
    loop = GymnaxLoop(env, env_params, agent, num_envs, next(key_gen), inference=True)
    num_cycles = 1
    steps_per_cycle = 1
    agent_state = agent.initial_state(networks, loop.example_batched_obs(), next(key_gen))
    agent_state, metrics = loop.run(agent_state, num_cycles, steps_per_cycle)
    for k in metrics:
        assert not k.startswith("action_counts_")


def test_observe_trajectory():
    env, env_params = gymnax.make("Swimmer-misc")
    networks = None
    num_envs = 2
    key_gen = keygen(jax.random.PRNGKey(0))
    agent = UniformRandom(env.action_space(), 0)
    loop = GymnaxLoop(env, env_params, agent, num_envs, next(key_gen), inference=True)
    num_cycles = 2
    steps_per_cycle = 3
    agent_state = agent.initial_state(networks, loop.example_batched_obs(), next(key_gen))

    def observe_trajectory(
        env_timesteps: EnvTimestep, step_num: int
    ):  # Crude check that we are getting the correct trajectory
        assert env_timesteps.obs.shape[0] == steps_per_cycle

    agent_state, _ = loop.run(agent_state, num_cycles, steps_per_cycle, observe_trajectory=observe_trajectory)
