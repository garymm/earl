"""Core types."""

import abc
from collections.abc import Mapping
from typing import Any, Generic, Protocol, TypeVar

import equinox as eqx
import jax.numpy as jnp
from jax_dataclasses import pytree_dataclass
from jaxtyping import PRNGKeyArray, PyTree, Scalar

_Networks = TypeVar("_Networks", bound=PyTree)
_CycleState = TypeVar("_CycleState")
_StepState = TypeVar("_StepState")

# Keyed by name.
Metrics = Mapping[str, Scalar | float | int]


class SupportsStr(Protocol):
    def __str__(self) -> str: ...


# A collection of configuration values, keyed by name. AKA hyperparameters.
ConfigForLog = Mapping[str, SupportsStr]


class AgentState(eqx.Module, Generic[_Networks, _CycleState, _StepState]):
    nets: _Networks
    cycle: _CycleState
    step: _StepState
    inference: bool = False


AgentState.__init__.__doc__ = """Args:

nets: neural networks. It must be a PyTree since it will
    be passed to equinox.combine(). It is updated in update_from_grads() based on gradients.
    Anything that needs to be optimized via gradient descent should be in nets.
    Any objects that need to change their behavior in inference or training mode should
    have a boolean member variable that is named "inference" for that purpose.

cycle: contains anything other than _Networks that also needs to be updated in
    once a cycle, i.e. by update_from_grads().
    This is where optimizer state belongs. If the only thing that your agent updates on each cycle is the optimizer
    state, you can set the _CycleState type var to optax.OptState.

step: contains anything that needs to be updated on each step. Typically
    random keys and RNN states go here.

inference: True means inference mode, False means training.
"""


@pytree_dataclass
class ActionAndStepState(Generic[_StepState]):
    action: jnp.ndarray
    step_state: _StepState


@pytree_dataclass
class EnvTimestep:
    """The result of taking an action in an environment.

    Note that it may be a batch of timesteps, in which case
    all of the members will have an additional leading batch dimension.
    """

    """whether this is the first timestep in an episode.

    Because our environments automatically reset, this is true in 2 cases:
    1. The first timestep in an experiment (after an explicit env.reset()).
    2. The last timestep of an episode (since the the environment automatically
       resets and the observation is the first of the next episode).
    """
    new_episode: jnp.ndarray
    obs: PyTree
    prev_action: jnp.ndarray  # the action taken in the previous timestep
    reward: jnp.ndarray


class Agent(abc.ABC, Generic[_Networks, _CycleState, _StepState]):
    """Abstract class for a reinforcement learning agent.

    Sub-classes should:
     * Create a sub-class of AgentState that fills in the type variables with concrete types.
       See that type's docs for guideance.
     * Implement the abstract methods.

    Sub-classes should not modify self after __init__(), since the methods are
    jax.jit-compiled, meaning the self argument is static.

    The framework will conceptually do one of the following to train an agent:

    1. On-policy training (if num_off_policy_updates_per_cycle() == 0):

    a = Agent(agent_config)
    env = vectorized_env(env_config)
    state = a.initial_state(env.observation_space.sample())

    def run_cycle_and_loss(nets_to_grad, nets_no_grad, state, env_timestep):
        state = AgentState(
            nets=eqx.combine(state_to_grad, state_no_grad),
            cycle=state.other,
            step=state.step,
        )

        for step in range(steps_per_cycle):
            action, state.step = a.select_action(state, env_timestep)
            env_timestep = env.step(action)
        return a.loss(state)

    env_timestep = env.reset()
    for cycle in range(num_cycles):
        nets_yes_grad, nets_no_grad = a.partition_for_grad(state.nets)
        grad = jax.grad(run_cycle_and_loss)(nets_yes_grad, nets_no_grad, state, env_timestep)
        if training:
            state = a.update_for_cycle(state, grad)

    2. Off-policy training (if num_off_policy_updates_per_cycle() > 0):

    a = Agent(agent_config)
    env = vectorized_env(env_config)
    state = a.initial_state(env.observation_space.sample())
    def run_cycle(state, env_timestep):

        for step in range(steps_per_cycle):
            action, state.step = a.select_action(state, env_timestep)
            env_timestep = env.step(action)
        return state

    def compute_loss(nets_yes_grad, nets_no_grad, state):
        state = AgentState(
            nets=eqx.combine(state_yes_grad, state_no_grad),
            cycle=state.cycle,
            step=state.step,
        )
        return a.loss(state)

    env_timestep = env.reset()
    for cycle in range(num_cycles):
        state = run_cycle(state, env_timestep)
        if training:
            for _ in range(a.num_off_policy_updates_per_cycle()):
                nets_to_grad, nets_no_grad = a.partition_for_grad(state.nets)
                nets_to_grad, nets_no_grad = a.partition_for_grad(state.nets)
                grad = jax.grad(a.loss)(nets_to_grad, nets_no_grad, state)
                state = a.update_for_cycle(state, grad)

    """

    def initial_state(self, nets: _Networks, obs: PyTree, key: PRNGKeyArray) -> AgentState:
        """Initializes agent state.

        Args:
            nets: The agent's neural networks. Donated, so callers should not access it after calling.
            obs: a batch of observations. All fields are batched. Provides shape and type info.
            key: a random key for the agent to use. Donated, so callers should not access it after calling.
        """

        @eqx.filter_jit(donate="all-except-first")
        # swap order of args so we can avoid donating obs
        def initial_state_jit(obs, nets, key):
            return self._initial_state(nets, obs, key)

        return initial_state_jit(obs, nets, key)

    @abc.abstractmethod
    def _initial_state(self, nets: _Networks, obs: PyTree, key: PRNGKeyArray) -> AgentState:
        """Initializes agent state.

        Must be jit-compatible.
        """

    def select_action(
        self, state: AgentState[_Networks, _CycleState, _StepState], env_timestep: EnvTimestep
    ) -> ActionAndStepState[_StepState]:
        """Selects a batch of actions and updates step state.

        Sub-classes should override _select_action. This method is a wrapper that adds jit-compilation.

        Args:
            state: The current agent state. Donated, so callers should not access it after calling.
            env_timestep: The current environment timestep. All fields are batched, so any vmap() should be done inside
                this method.

        Returns:
            ActionAndStepState which contains the batch of actions and the updated step state.
        """

        @eqx.filter_jit(donate="all-except-first")
        # swap order of args so we can avoid donating env_timestep
        def select_action_jit(env_timestep, state):
            return self._select_action(state, env_timestep)

        return select_action_jit(env_timestep, state)

    @abc.abstractmethod
    def _select_action(
        self, state: AgentState[_Networks, _CycleState, _StepState], env_timestep: EnvTimestep
    ) -> ActionAndStepState[_StepState]:
        """Selects a batch of actions and updates step state.

        Must be jit-compatible.
        """

    def partition_for_grad(self, nets: _Networks) -> tuple[_Networks, _Networks]:
        """Partitions nets into trainable and non-trainable parts.

        Nets arg is donated, meaning callers should not access it after calling.

        Sub-classes should override _partition_for_grad. This method is a wrapper that adds jit-compilation.

        Returns: A tuple of Networks, the first of which contains all the fields for which the gradients
            should be calculated, and the second contains the rest. They will be combined by with
            equinox.combine().
        """
        return eqx.filter_jit(donate="all")(self._partition_for_grad)(nets)

    def _partition_for_grad(self, nets: _Networks) -> tuple[_Networks, _Networks]:
        """Partitions nets into trainable and non-trainable parts.

        Must be jit-compatible.

        Default implementation assumes nets is a PyTree and that all the
        array leaves are trainable. If this is not the case, this method should be overridden.
        """
        return eqx.partition(nets, eqx.is_array)

    def loss(self, state: AgentState[_Networks, _CycleState, _StepState]) -> tuple[Scalar, Metrics]:
        """Returns loss and metrics. Called after some number of environment steps.

        State arg is donated, meaning callers should not access it after calling.

        Sub-classes should override _loss. This method is a wrapper that adds jit-compilation.

        Note: the returned metrics should not have any keys that conflict with gymnax_loop.MetricKey.
        """

        return eqx.filter_jit(donate="all")(self._loss)(state)

    @abc.abstractmethod
    def _loss(self, state: AgentState[_Networks, _CycleState, _StepState]) -> tuple[Scalar, Metrics]:
        """Returns loss and metrics. Called after some number of environment steps.

        Must be jit-compatible.
        """

    def update_from_grads(
        self, state: AgentState[_Networks, _CycleState, _StepState], nets_grads: PyTree
    ) -> AgentState[_Networks, _CycleState, _StepState]:
        """Updates agent state based on gradients of the losses returned by self.loss().

        Sub-classes should override _update_from_grads. This method is a wrapper that adds jit-compilation.

        Args:
            state: The current agent state. Donated, so callers should not access it after calling.
            nets_grads is the gradient of the loss w.r.t. the agent's networks. Donated,
                so callers should not access it after calling.
        """
        return eqx.filter_jit(donate="all")(self._update_from_grads)(state, nets_grads)

    @abc.abstractmethod
    def _update_from_grads(
        self, state: AgentState[_Networks, _CycleState, _StepState], nets_grads: PyTree
    ) -> AgentState[_Networks, _CycleState, _StepState]:
        """Updates agent state based on gradients of the losses returned by self._loss().

        Must be jit-compatible.
        """

    @abc.abstractmethod
    def num_off_policy_updates_per_cycle(self) -> int:
        """Returns the number of off-policy updates per cycle.

        An off-policy update conceptually consists of:
        - grads = jax.grad(self.loss)(state)
        - state = self.update_from_grads(state, grads).

        If 0, then the framework should only do on-policy updates, which consist of:
        - grads = jax.grad(multiple calls to self.select_action(state), one call to self.loss(state))()
        - state = self.update_from_grads(state, grads)
        """


class ObserveTrajectory(Protocol):
    def __call__(self, env_timesteps: EnvTimestep, step_infos: dict[Any, Any], step_num: int) -> None: ...

    """
    env_timesteps is a trajectory of env timesteps
    step_infos is the aggregated info returned from environment.step()
    step_num is the number of steps taken prior to the trajectory
    """
