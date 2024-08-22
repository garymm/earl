import dataclasses
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import optax
from jaxtyping import PRNGKeyArray, PyTree, Scalar

from research.earl.core import ActionAndStepState, Agent, EnvTimestep, Metrics
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


AgentState = CoreAgentState[eqx.nn.Sequential, optax.OptState, StepState]


class SimplePolicyGradient(Agent[eqx.nn.Sequential, optax.OptState, StepState]):
    """Simple policy gradient agent.

    Based on https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html.

    The main differences are:
    - This acts on batched observations.
    - This discounts rewards.
    """

    def __init__(self, config: Config):
        self.config = config

    def _initial_state(self, nets: eqx.nn.Sequential, obs: PyTree, key: PRNGKeyArray) -> AgentState:
        assert isinstance(obs, jax.Array)
        num_envs = obs.shape[0]
        return AgentState(
            nets=nets,
            cycle=self.config.optimizer.init(eqx.filter(nets, eqx.is_array)),
            step=StepState(
                chosen_action_log_probs=jnp.zeros((num_envs, self.config.max_step_state_history)),
                key=key,
                mask=jnp.ones((num_envs,), dtype=jnp.bool),
                rewards=jnp.zeros((num_envs, self.config.max_step_state_history)),
                t=jnp.array(0, dtype=jnp.uint32),
            ),
        )

    def _select_action(self, state: AgentState, env_timestep: EnvTimestep, training: bool) -> ActionAndStepState:
        logits = jax.vmap(state.nets)(env_timestep.obs)
        actions_key, key = jax.random.split(state.step.key)
        actions = jax.vmap(jax.random.categorical, in_axes=(None, 0))(actions_key, logits)
        log_probs = jax.vmap(jax.nn.log_softmax)(logits)
        log_probs_for_actions = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze(axis=1)

        # This will silently be a no-op if step.t >= max_step_state_history.
        def set_batch(buf: jnp.ndarray, newval: jnp.ndarray):
            return buf.at[:, state.step.t].set(newval)

        step_state = jdc.replace(
            state.step,
            chosen_action_log_probs=set_batch(
                state.step.chosen_action_log_probs, log_probs_for_actions * state.step.mask
            ),
            key=key,
            mask=state.step.mask & ~env_timestep.done,
            rewards=set_batch(state.step.rewards, env_timestep.reward * state.step.mask),
            t=state.step.t + 1,
        )
        return ActionAndStepState(actions, step_state)

    def _update_from_grads(self, state: AgentState, nets_grads: PyTree) -> AgentState:
        updates, opt_state = self.config.optimizer.update(nets_grads, state.cycle)
        nets = eqx.apply_updates(state.nets, updates)
        step = jdc.replace(
            state.step,
            chosen_action_log_probs=jnp.zeros_like(state.step.chosen_action_log_probs),
            mask=jnp.ones_like(state.step.mask),
            rewards=jnp.zeros_like(state.step.rewards),
            t=jnp.zeros_like(state.step.t),
        )
        return jdc.replace(state, nets=nets, cycle=opt_state, step=step)

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

    def num_off_policy_updates_per_cycle(self) -> int:
        return 0
