from collections.abc import Callable, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree, Scalar

from earl.core import ActionAndState, Agent, EnvStep
from earl.core import AgentState as CoreAgentState


class ActorState(NamedTuple):
  """State that gets updated on every act()."""

  key: PRNGKeyArray
  t: jax.Array


class OptState(NamedTuple):
  """State that gets updated on every optimization step."""

  opt_count: jax.Array


AgentState = CoreAgentState[None, OptState, None, ActorState]


class RandomAgent(Agent[None, OptState, None, ActorState]):
  """Agent that selects actions at random."""

  _sample_action_space: Callable
  _num_off_policy_updates: int

  def _new_actor_state(self, nets: None, key: PRNGKeyArray) -> ActorState:
    return ActorState(key, jnp.zeros((1,), dtype=jnp.uint32))

  def _new_opt_state(self, nets: None, key: PRNGKeyArray) -> OptState:
    return OptState(jnp.zeros((), dtype=jnp.uint32))

  def _new_experience_state(self, nets: None, key: PRNGKeyArray) -> None:
    return None

  def _act(
    self, actor_state: ActorState, nets: None, env_step: EnvStep
  ) -> ActionAndState[ActorState]:
    key, action_key = jax.random.split(actor_state.key)
    num_envs = env_step.obs.shape[0]
    actions = jax.vmap(self._sample_action_space)(jax.random.split(action_key, num_envs))
    assert isinstance(actions, jax.Array)
    return ActionAndState(actions, ActorState(key, actor_state.t + 1))

  def _partition_for_grad(self, nets: None) -> tuple[None, None]:
    return None, nets

  def _loss(self, nets: None, opt_state: OptState, experience_state: None) -> tuple[Scalar, None]:
    return jnp.array(0.0), None

  def _optimize_from_grads(
    self, nets: None, opt_state: OptState, nets_grads: PyTree
  ) -> tuple[None, OptState]:
    assert opt_state.opt_count is not None
    return nets, OptState(opt_count=opt_state.opt_count + 1)

  def _update_experience(
    self,
    experience_state: None,
    actor_state_pre: ActorState,
    actor_state_post: ActorState,
    trajectory: EnvStep,
  ) -> None:
    return None

  def num_off_policy_optims_per_cycle(self) -> int:
    """Number of off-policy updates per cycle."""
    return self._num_off_policy_updates

  def _prepare_for_actor_cycle(self, actor_state: ActorState) -> ActorState:
    """Resets the random key."""
    return actor_state

  def shard_actor_state(
    self, actor_state: ActorState, learner_devices: Sequence[jax.Device]
  ) -> ActorState:
    return ActorState(
      key=jax.device_put_replicated(actor_state.key, learner_devices),
      t=jax.device_put_replicated(actor_state.t, learner_devices),
    )
