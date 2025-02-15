import dataclasses
from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PRNGKeyArray, PyTree, Scalar

from earl.core import ActionAndState, Agent, EnvStep, Metrics
from earl.core import AgentState as CoreAgentState
from earl.utils.sharding import shard_along_axis_0


class ActorState(eqx.Module):
  chosen_action_log_probs: jax.Array
  key: PRNGKeyArray
  mask: jax.Array  # 0 if env was ever done.
  rewards: jax.Array
  t: jax.Array


class ExperienceState(eqx.Module):
  rewards: jax.Array
  chosen_action_log_probs: jax.Array


@dataclasses.dataclass(eq=True, frozen=True)
class Config:
  max_actor_state_history: int
  optimizer: optax.GradientTransformation
  discount: float = 0.99


def make_networks(layer_dims: list[int], key: PRNGKeyArray) -> eqx.nn.Sequential:
  layers: list[Callable] = []
  for i in range(len(layer_dims) - 1):
    layers.append(eqx.nn.Linear(layer_dims[i], layer_dims[i + 1], key=key))
    if i < len(layer_dims) - 2:
      layers.append(eqx.nn.Lambda(jnp.tanh))
  return eqx.nn.Sequential(layers)


AgentState = CoreAgentState[eqx.nn.Sequential, optax.OptState, ExperienceState, ActorState]


class SimplePolicyGradient(Agent[eqx.nn.Sequential, optax.OptState, ExperienceState, ActorState]):
  """Simple policy gradient agent.

  Based on https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html.

  The main differences are:
  - This acts on batched observations.
  - This discounts rewards.
  """

  config: Config

  def _new_actor_state(self, nets: eqx.nn.Sequential, key: PRNGKeyArray) -> ActorState:
    return ActorState(
      chosen_action_log_probs=jnp.zeros(
        (self.env_info.num_envs, self.config.max_actor_state_history)
      ),
      key=key,
      mask=jnp.ones((self.env_info.num_envs,), dtype=jnp.bool),
      rewards=jnp.zeros((self.env_info.num_envs, self.config.max_actor_state_history)),
      t=jnp.array(0, dtype=jnp.uint32),
    )

  def _new_experience_state(self, nets: eqx.nn.Sequential, key: PRNGKeyArray) -> ExperienceState:
    return ExperienceState(
      rewards=jnp.zeros((self.env_info.num_envs, self.config.max_actor_state_history)),
      chosen_action_log_probs=jnp.zeros(
        (self.env_info.num_envs, self.config.max_actor_state_history)
      ),
    )

  def _new_opt_state(self, nets: eqx.nn.Sequential, key: PRNGKeyArray) -> optax.OptState:
    return self.config.optimizer.init(eqx.filter(nets, eqx.is_array))

  def _act(
    self, actor_state: ActorState, nets: eqx.nn.Sequential, env_step: EnvStep
  ) -> ActionAndState:
    logits = jax.vmap(nets)(env_step.obs)
    actions_key, key = jax.random.split(actor_state.key)
    actions = jax.vmap(jax.random.categorical, in_axes=(None, 0))(actions_key, logits)
    log_probs = jax.vmap(jax.nn.log_softmax)(logits)
    log_probs_for_actions = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze(axis=1)

    # This will silently be a no-op if actor_state.t >= max_actor_state_history.
    def set_batch(buf: jax.Array, newval: jax.Array):
      return buf.at[:, actor_state.t].set(newval)

    actor_state = dataclasses.replace(
      actor_state,
      chosen_action_log_probs=set_batch(
        actor_state.chosen_action_log_probs, log_probs_for_actions * actor_state.mask
      ),
      key=key,
      mask=actor_state.mask & ~env_step.new_episode,
      rewards=set_batch(actor_state.rewards, env_step.reward * actor_state.mask),
      t=actor_state.t + 1,
    )
    return ActionAndState(actions, actor_state)

  def _prepare_for_actor_cycle(self, actor_state: ActorState) -> ActorState:
    """Prepares actor state for a new cycle of acting.

    Resets the buffers and counters while preserving the random key.
    """
    return ActorState(
      chosen_action_log_probs=jnp.zeros(
        (self.env_info.num_envs, self.config.max_actor_state_history)
      ),
      key=actor_state.key,
      mask=jnp.ones((self.env_info.num_envs,), dtype=jnp.bool),
      rewards=jnp.zeros((self.env_info.num_envs, self.config.max_actor_state_history)),
      t=jnp.array(0, dtype=jnp.uint32),
    )

  def _optimize_from_grads(
    self, nets: eqx.nn.Sequential, opt_state: optax.OptState, nets_grads: PyTree
  ) -> tuple[eqx.nn.Sequential, optax.OptState]:
    updates, opt_state = self.config.optimizer.update(nets_grads, opt_state)
    nets = eqx.apply_updates(nets, updates)
    return nets, opt_state

  def _loss(
    self, nets: eqx.nn.Sequential, opt_state: optax.OptState, experience_state: ExperienceState
  ) -> tuple[Scalar, Metrics]:
    def discounted_returns(carry, x):
      carry = x + self.config.discount * carry
      return carry, carry

    @eqx.filter_vmap
    def vmap_discounted_returns(rewards):
      _, ys = jax.lax.scan(discounted_returns, init=jnp.array(0), xs=rewards, reverse=True)
      return ys

    returns = vmap_discounted_returns(experience_state.rewards)
    return -jnp.mean(returns * experience_state.chosen_action_log_probs), {}

  def num_off_policy_optims_per_cycle(self) -> int:
    return 0

  def _update_experience(
    self,
    experience_state: ExperienceState,
    actor_state_pre: ActorState,
    actor_state_post: ActorState,
    trajectory: EnvStep,
  ) -> ExperienceState:
    return ExperienceState(
      rewards=actor_state_post.rewards,
      chosen_action_log_probs=actor_state_post.chosen_action_log_probs,
    )

  def shard_actor_state(
    self, actor_state: ActorState, learner_devices: Sequence[jax.Device]
  ) -> ActorState:
    return ActorState(
      chosen_action_log_probs=shard_along_axis_0(
        actor_state.chosen_action_log_probs, learner_devices
      ),
      key=jax.device_put_replicated(actor_state.key, learner_devices),
      mask=shard_along_axis_0(actor_state.mask, learner_devices),
      rewards=shard_along_axis_0(actor_state.rewards, learner_devices),
      t=jax.device_put_replicated(actor_state.t, learner_devices),
    )
