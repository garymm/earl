"""Common code for environment loops."""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Generic, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from gymnax import EnvState
from jax_loop_utils.metric_writers.interface import Scalar as ScalarMetric
from jaxtyping import PRNGKeyArray, PyTree

from earl.core import (
  AgentState,
  ConflictingMetricError,
  EnvStep,
  Image,
  Metrics,
  Video,
  _ActorState,
  _ExperienceState,
  _Networks,
  _OptState,
)
from earl.metric_key import MetricKey

ArrayMetrics = dict[str, jax.Array]

_ALL_METRIC_KEYS = {str(k) for k in MetricKey}


class CycleResult(eqx.Module):
  """Represents the result of running one cycle of environment interactions."""

  agent_state: PyTree
  """The state of the agent after completing the cycle. Contains networks, optimizer state,
  experience buffer, and any other agent-specific state."""

  env_state: EnvState
  """The final state of the environment after the cycle. Used by Gymnax to track internal
  environment state."""

  env_step: EnvStep
  """The final environment step, containing observation, reward, done flag, and previous action."""

  key: PRNGKeyArray
  """The PRNG key after the cycle, used for maintaining reproducible randomness."""

  metrics: ArrayMetrics
  """A dictionary mapping metric names to JAX arrays containing various metrics collected during
  the cycle (e.g., rewards, losses, episode lengths)."""

  trajectory: EnvStep
  """The sequence of environment steps taken during this cycle. Shape is (num_envs, num_steps)
  for array fields."""

  step_infos: dict[Any, Any]
  """Additional information returned by the environment's step function for each step in the
  cycle. Contents depend on the specific environment being used."""


class ObserveCycle(Protocol):
  def __call__(self, trajectory: EnvStep, step_infos: dict[Any, Any]) -> Metrics: ...

  """A function that takes a trajectory and step_infos and produces a set of metrics using
  custom logic specific to the environment.
  """


def no_op_observe_cycle(trajectory: EnvStep, step_infos: dict[Any, Any]) -> Metrics:
  return {}


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
      return {
        k: v for i, item in enumerate(obj) for k, v in traverse(item, f"{current_path}_{i}").items()
      }

    if hasattr(obj, "__dict__"):
      return {
        k: v
        for attr, value in obj.__dict__.items()
        for k, v in traverse(value, f"{current_path}.{attr}" if current_path else attr).items()
      }

    return {}

  return traverse(pytree)


@dataclasses.dataclass(frozen=True)
class State(Generic[_Networks, _OptState, _ExperienceState, _ActorState]):
  agent_state: AgentState[_Networks, _OptState, _ExperienceState, _ActorState]
  env_state: EnvState | None = None
  env_step: EnvStep | list[EnvStep] | None = None
  step_num: int = 0


State.__init__.__doc__ = """Initializes the State.

Args:
    agent_state: The agent state.
    env_step: The environment step. If None, the environment will be reset.
    step_num: The step number to start the metric logging at.
"""


@dataclasses.dataclass(frozen=True)
class Result(State[_Networks, _OptState, _ExperienceState, _ActorState]):
  """A State but with all fields guaranteed to be not None."""

  # Note: defaults are just to please the type checker, since the base class has defaults.
  # We should not actually use the defaults.
  env_step: EnvStep | list[EnvStep] = dataclasses.field(
    default_factory=lambda: EnvStep(jnp.array(0), jnp.array(0), jnp.array(0), jnp.array(0))
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


def extract_metrics(
  cycle_result_metrics: ArrayMetrics, observe_cycle_metrics: Metrics
) -> MetricsByType:
  """Extracts scalar and image metrics from a CycleResult and observe_cycle_metrics."""
  scalar_metrics: dict[str, float | int] = {}
  image_metrics: dict[str, jax.Array] = {}
  video_metrics: dict[str, jax.Array] = {}
  # handle pmap case. Choice of metric to check is arbitrary.
  has_device_dim = cycle_result_metrics[MetricKey.REWARD_SUM].ndim > 0
  if has_device_dim:
    for k in cycle_result_metrics:
      if "mean" in k:
        cycle_result_metrics[k] = jnp.mean(cycle_result_metrics[k], axis=0)
      else:
        cycle_result_metrics[k] = jnp.sum(cycle_result_metrics[k], axis=0)

  action_counts = cycle_result_metrics.pop(MetricKey.ACTION_COUNTS)
  assert isinstance(action_counts, jax.Array)
  if action_counts.shape:  # will be empty tuple if not discrete action space
    for i in range(action_counts.shape[0]):
      scalar_metrics[f"action_counts/{i}"] = int(action_counts[i])

  for k, v in cycle_result_metrics.items():
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


def pixel_obs_to_video_observe_cycle(trajectory: EnvStep, step_infos: dict[Any, Any]) -> Metrics:
  obs = trajectory.obs
  if len(obs.shape) not in (4, 5):
    raise ValueError(
      f"Expected trajectory.obs to have shape (T, H, W, C) or (B, T, H, W, C),got {obs.shape}"
    )
  if len(obs.shape) == 4:
    return {"video": Video(obs)}
  else:
    return {f"video_{i}": Video(obs[i]) for i in range(obs.shape[0])}


def multi_observe_cycle(observers: Sequence[ObserveCycle]) -> ObserveCycle:
  def _multi_observe_cycle(trajectory: EnvStep, step_infos: dict[Any, Any]) -> Metrics:
    return dict(
      item for observe_cycle in observers for item in observe_cycle(trajectory, step_infos).items()
    )

  return _multi_observe_cycle
