import dataclasses
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree, Scalar

from src.earl.core import Agent, AgentStep, EnvInfo, EnvStep, Metrics
from src.earl.core import AgentState as CoreAgentState


class StepState(NamedTuple):
    """State that gets updated on every step."""

    key: PRNGKeyArray
    t: jax.Array


class OptState(NamedTuple):
    """State that gets updated on every optimization step."""

    opt_count: jax.Array


AgentState = CoreAgentState[None, OptState, None, StepState]


class RandomAgent(Agent[None, OptState, None, StepState]):
    """Agent that selects actions at random."""

    _sample_action_space: Callable
    _num_off_policy_updates: int
    _prng_metric_key: str = "prng"

    def _new_step_state(self, nets: None, env_info: EnvInfo, key: PRNGKeyArray) -> StepState:
        return StepState(key, jnp.zeros((1,), dtype=jnp.uint32))

    def _new_opt_state(self, nets: None, env_info: EnvInfo, key: PRNGKeyArray) -> OptState:
        return OptState(jnp.zeros((1,), dtype=jnp.uint32))

    def _new_experience_state(self, nets: None, env_info: EnvInfo, key: PRNGKeyArray) -> None:
        return None

    def _step(self, state: AgentState, env_step: EnvStep) -> AgentStep[StepState]:
        key, action_key = jax.random.split(state.step.key)
        num_envs = env_step.obs.shape[0]
        actions = jax.vmap(self._sample_action_space)(jax.random.split(action_key, num_envs))
        assert isinstance(actions, jax.Array)
        return AgentStep(actions, StepState(key, state.step.t + 1))

    def _partition_for_grad(self, nets: None) -> tuple[None, None]:
        return None, None

    def _loss(self, state: AgentState) -> tuple[Scalar, Metrics]:
        return jnp.array(0.0), {
            # metrics need to be scalars, so take elem 0.
            self._prng_metric_key: state.step.key[0],
        }

    def _optimize_from_grads(self, state: AgentState, nets_grads: PyTree) -> AgentState:
        assert state.opt is not None
        return dataclasses.replace(state, opt=OptState(state.opt.opt_count + 1))

    def _update_experience(self, state: AgentState, trajectory: EnvStep) -> None:
        return None

    def num_off_policy_optims_per_cycle(self) -> int:
        """Number of off-policy updates per cycle."""
        return self._num_off_policy_updates
