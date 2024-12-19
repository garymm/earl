"""Common code for environment loops."""

from typing import Any, Protocol

import equinox as eqx
import jax
from gymnax import EnvState
from jaxtyping import PRNGKeyArray, PyTree

from research.earl.core import EnvStep, Metrics

ArrayMetrics = dict[str, jax.Array]


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
