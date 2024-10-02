import dataclasses
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import optax
from jaxtyping import PRNGKeyArray, PyTree, Scalar

from research.earl.core import Agent, AgentStep, EnvInfo, EnvStep, Metrics
from research.earl.core import AgentState as CoreAgentState


@jdc.pytree_dataclass
class StepState:
    chosen_action_log_probs: jnp.ndarray
    key: PRNGKeyArray
    mask: jnp.ndarray  # 0 if env was ever done.
    rewards: jnp.ndarray
    t: jnp.ndarray


@dataclasses.dataclass(eq=True, frozen=True)
class Config:
    max_step_state_history: int
    optimizer: optax.GradientTransformation
    discount: float = 0.99


def make_networks(layer_dims: list[int], key: PRNGKeyArray) -> eqx.nn.Sequential:
    layers: list[Callable] = []
    for i in range(len(layer_dims) - 1):
        layers.append(eqx.nn.Linear(layer_dims[i], layer_dims[i + 1], key=key))
        if i < len(layer_dims) - 2:
            layers.append(eqx.nn.Lambda(jnp.tanh))
    return eqx.nn.Sequential(layers)


AgentState = CoreAgentState[eqx.nn.Sequential, optax.OptState, None, StepState]


class SimplePolicyGradient(Agent[eqx.nn.Sequential, optax.OptState, None, StepState]):
    """Simple policy gradient agent.

    Based on https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html.

    The main differences are:
    - This acts on batched observations.
    - This discounts rewards.
    """

    def __init__(self, config: Config):
        self.config = config

    def _new_step_state(self, nets: eqx.nn.Sequential, env_info: EnvInfo, key: PRNGKeyArray) -> StepState:
        return StepState(
            chosen_action_log_probs=jnp.zeros((env_info.num_envs, self.config.max_step_state_history)),
            key=key,
            mask=jnp.ones((env_info.num_envs,), dtype=jnp.bool),
            rewards=jnp.zeros((env_info.num_envs, self.config.max_step_state_history)),
            t=jnp.array(0, dtype=jnp.uint32),
        )

    def _new_experience_state(self, nets: eqx.nn.Sequential, env_info: EnvInfo, key: PRNGKeyArray) -> None:
        return None

    def _new_opt_state(self, nets: eqx.nn.Sequential, env_info: EnvInfo, key: PRNGKeyArray) -> optax.OptState:
        return self.config.optimizer.init(eqx.filter(nets, eqx.is_array))

    def _step(self, state: AgentState, env_step: EnvStep) -> AgentStep:
        logits = jax.vmap(state.nets)(env_step.obs)
        actions_key, key = jax.random.split(state.step.key)
        actions = jax.vmap(jax.random.categorical, in_axes=(None, 0))(actions_key, logits)
        log_probs = jax.vmap(jax.nn.log_softmax)(logits)
        log_probs_for_actions = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze(axis=1)

        # This will silently be a no-op if step.t >= max_step_state_history.
        def set_batch(buf: jnp.ndarray, newval: jnp.ndarray):
            return buf.at[:, state.step.t].set(newval)

        step = dataclasses.replace(
            state.step,
            chosen_action_log_probs=set_batch(
                state.step.chosen_action_log_probs, log_probs_for_actions * state.step.mask
            ),
            key=key,
            mask=state.step.mask & ~env_step.new_episode,
            rewards=set_batch(state.step.rewards, env_step.reward * state.step.mask),
            t=state.step.t + 1,
        )
        return AgentStep(actions, step)

    def _optimize_from_grads(self, state: AgentState, nets_grads: PyTree) -> AgentState:
        updates, opt_state = self.config.optimizer.update(nets_grads, state.opt)
        nets = eqx.apply_updates(state.nets, updates)
        step = dataclasses.replace(
            state.step,
            chosen_action_log_probs=jnp.zeros_like(state.step.chosen_action_log_probs),
            mask=jnp.ones_like(state.step.mask),
            rewards=jnp.zeros_like(state.step.rewards),
            t=jnp.zeros_like(state.step.t),
        )
        return dataclasses.replace(state, nets=nets, opt=opt_state, step=step)

    def _loss(self, state: AgentState) -> tuple[Scalar, Metrics]:
        def discounted_returns(carry, x):
            carry = x + self.config.discount * carry
            return carry, carry

        @eqx.filter_vmap
        def vmap_discounted_returns(rewards):
            _, ys = jax.lax.scan(discounted_returns, init=jnp.array(0), xs=rewards, reverse=True)
            return ys

        returns = vmap_discounted_returns(state.step.rewards)
        return -jnp.mean(returns * state.step.chosen_action_log_probs), {}

    def num_off_policy_optims_per_cycle(self) -> int:
        return 0

    def _update_experience(self, state: AgentState, trajectory: EnvStep) -> None:
        return None
