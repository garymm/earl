"""Common code for environment loops."""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Generic, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from gymnax import EnvState
from jax_loop_utils.metric_writers.interface import Scalar as ScalarMetric
from jaxtyping import PRNGKeyArray, PyTree, Scalar

from src.earl.core import (
    AgentState,
    ConflictingMetricError,
    EnvStep,
    Image,
    Metrics,
    Video,
    _ExperienceState,
    _Networks,
    _OptState,
    _StepState,
)
from src.earl.metric_key import MetricKey

ArrayMetrics = dict[str, jax.Array]

_ALL_METRIC_KEYS = {str(k) for k in MetricKey}


class CycleResult(eqx.Module):
    agent_state: PyTree
    env_state: EnvState
    env_step: EnvStep
    key: PRNGKeyArray
    metrics: ArrayMetrics
    trajectory: EnvStep
    step_infos: dict[Any, Any]


CycleResult.__init__.__doc__ = """Represents the result of running one cycle of environment interactions.

Args:
    agent_state: The state of the agent after completing the cycle. Contains networks, optimizer state,
                experience buffer, and any other agent-specific state.
    env_state: The final state of the environment after the cycle. Used by Gymnax to track internal
              environment state.
    env_step: The final environment step, containing observation, reward, done flag, and previous action.
    key: The PRNG key after the cycle, used for maintaining reproducible randomness.
    metrics: A dictionary mapping metric names to JAX arrays containing various metrics collected during
            the cycle (e.g., rewards, losses, episode lengths).
    trajectory: The sequence of environment steps taken during this cycle. Shape is (num_envs, num_steps)
               for array fields.
    step_infos: Additional information returned by the environment's step function for each step in the
                cycle. Contents depend on the specific environment being used.
"""


class ObserveCycle(Protocol):
    def __call__(self, cycle_result: CycleResult) -> Metrics: ...

    """A function that takes a CycleResult representing the final state after a cycle of environment steps
    and produces a set of metrics using custom logic specific to the environment.
    """


def no_op_observe_cycle(cycle_result: CycleResult) -> Metrics:
    return {}


class StepCarry(eqx.Module, Generic[_StepState]):
    env_step: EnvStep
    env_state: EnvState
    step_state: _StepState
    key: PRNGKeyArray
    total_reward: Scalar
    total_dones: Scalar
    """Number of steps in current episode for each environment."""
    episode_steps: jax.Array
    complete_episode_length_sum: Scalar
    complete_episode_count: Scalar
    action_counts: jax.Array


def raise_if_metric_conflicts(metrics: Mapping):
    conflicting_keys = [k for k in metrics if k in _ALL_METRIC_KEYS]
    if not conflicting_keys:
        return
    raise ConflictingMetricError(
        "The following metrics conflict with Earl's default MetricKey: " + ", ".join(conflicting_keys)
    )


def pytree_leaf_means(pytree: PyTree, prefix: str) -> dict[str, jax.Array]:
    """Returns a dict with key = path to the array in pytree, val = mean of array."""

    def traverse(obj, current_path=""):
        if isinstance(obj, jax.Array):
            return {f"{prefix}/{current_path}": jnp.mean(obj)}

        if isinstance(obj, tuple | list):
            return {k: v for i, item in enumerate(obj) for k, v in traverse(item, f"{current_path}[{i}]").items()}

        if hasattr(obj, "__dict__"):
            return {
                k: v
                for attr, value in obj.__dict__.items()
                for k, v in traverse(value, f"{current_path}.{attr}" if current_path else attr).items()
            }

        return {}

    return traverse(pytree)


@dataclasses.dataclass(frozen=True)
class State(Generic[_Networks, _OptState, _ExperienceState, _StepState]):
    agent_state: AgentState[_Networks, _OptState, _ExperienceState, _StepState]
    env_state: EnvState | None = None
    env_step: EnvStep | None = None
    step_num: int = 0


State.__init__.__doc__ = """Initializes the State.

Args:
    agent_state: The agent state.
    env_step: The environment step. If None, the environment will be reset.
    step_num: The step number to start the metric logging at.
"""


@dataclasses.dataclass(frozen=True)
class Result(State[_Networks, _OptState, _ExperienceState, _StepState]):
    """A State but with all fields guaranteed to be not None."""

    # Note: defaults are just to please the type checker, since the base class has defaults.
    # We should not actually use the defaults.
    env_state: EnvState = EnvState(0)
    env_step: EnvStep = EnvStep(jnp.array(0), jnp.array(0), jnp.array(0), jnp.array(0))


def result_to_cycle_result(result: Result) -> CycleResult:
    """Converts a Result to a CycleResult.

    Result does not contain all the fields that CycleResult does, namely key, metrics, trajectory, and step_infos.
    This function fills in the missing fields with dummy values.
    """
    return CycleResult(
        agent_state=result.agent_state,
        env_state=result.env_state,
        env_step=result.env_step,
        key=jax.random.PRNGKey(0),
        metrics={},
        trajectory=EnvStep(jnp.array(0), jnp.array(0), jnp.array(0), jnp.array(0)),
        step_infos={},
    )


def to_num_envs_first(x):
    # For a trajectory of (num_steps, num_envs, ...), we convert to (num_envs, num_steps, ...)
    # Required for FlashBax.
    if isinstance(x, jax.Array) and x.ndim > 1:
        return jnp.transpose(x, (1, 0, *range(2, x.ndim)))
    return x


@dataclasses.dataclass
class MetricsByType:
    scalar: dict[str, float | int]
    image: dict[str, jax.Array]
    video: dict[str, jax.Array]


def extract_metrics(cycle_result: CycleResult, observe_cycle_metrics: Metrics) -> MetricsByType:
    """Extracts scalar and image metrics from a CycleResult and observe_cycle_metrics."""
    scalar_metrics: dict[str, float | int] = {}
    image_metrics: dict[str, jax.Array] = {}
    video_metrics: dict[str, jax.Array] = {}
    action_counts = cycle_result.metrics.pop(MetricKey.ACTION_COUNTS)
    assert isinstance(action_counts, jax.Array)
    if action_counts.shape:  # will be empty tuple if not discrete action space
        for i in range(action_counts.shape[0]):
            scalar_metrics[f"action_counts/{i}"] = int(action_counts[i])

    for k, v in cycle_result.metrics.items():
        assert v.shape == (), f"Expected scalar metric {k} to be scalar, got {v.shape}"
        scalar_metrics[k] = v.item()

    for k, v in observe_cycle_metrics.items():
        if isinstance(v, ScalarMetric):
            if isinstance(v, jax.Array):
                v = v.item()
            scalar_metrics[k] = v
        elif isinstance(v, Image):
            image_metrics[k] = v.data
        elif isinstance(v, Video):
            video_metrics[k] = v.data
        else:
            raise ValueError(f"Unknown metric type: {type(v)}")

    return MetricsByType(scalar=scalar_metrics, image=image_metrics, video=video_metrics)


def pixel_obs_to_video_observe_cycle(cycle_result: CycleResult) -> Metrics:
    obs = cycle_result.trajectory.obs
    if len(obs.shape) not in (4, 5):
        raise ValueError(f"Expected trajectory.obs to have shape (T, H, W, C) or (B, T, H, W, C),got {obs.shape}")
    if len(obs.shape) == 4:
        return {"video": Video(obs)}
    else:
        return {f"video_{i}": Video(obs[i]) for i in range(obs.shape[0])}


def multi_observe_cycle(observers: Sequence[ObserveCycle]) -> ObserveCycle:
    def _multi_observe_cycle(cycle_result: CycleResult) -> Metrics:
        return dict(item for observe_cycle in observers for item in observe_cycle(cycle_result).items())

    return _multi_observe_cycle
