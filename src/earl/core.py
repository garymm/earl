"""Core types."""

import abc
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Generic, NamedTuple, Protocol, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from gymnax.environments.environment import Environment as GymnaxEnv
from gymnax.environments.environment import EnvParams
from gymnax.environments.spaces import Space
from jax_dataclasses import pytree_dataclass
from jaxtyping import PRNGKeyArray, PyTree, Scalar

_Networks = TypeVar("_Networks", bound=PyTree)
_OptState = TypeVar("_OptState")
_StepState = TypeVar("_StepState")
_ExperienceState = TypeVar("_ExperienceState")
# Keyed by name.
Metrics = Mapping[str, Scalar | float | int]


class SupportsStr(Protocol):
    def __str__(self) -> str: ...


# A collection of configuration values, keyed by name. AKA hyperparameters.
ConfigForLog = Mapping[str, SupportsStr]


class AgentState(eqx.Module, Generic[_Networks, _OptState, _ExperienceState, _StepState]):
    step: _StepState
    nets: _Networks
    opt: _OptState
    experience: _ExperienceState
    inference: bool = False


AgentState.__init__.__doc__ = """Args:

step: contains anything that needs to be updated on each step (other than replay buffers).
    Typically random keys and RNN states go here.

nets: neural networks. It must be a PyTree since it will
    be passed to equinox.combine(). It is updated in update_from_grads() based on gradients.
    Anything that needs to be optimized via gradient descent should be in nets.
    Any objects that need to change their behavior in inference or training mode should
    have a boolean member variable that is named "inference" for that purpose
    (all equinox.nn built-in modules have this).

opt: contains anything other than nets that also needs to be updated when optimizing
    (i.e. in optimize_from_grads()). This is where optimizer state belongs.
    Set to optax.OptState if you only have one optimizer, or you can set it to a custom class.

experience: experience replay buffer state.

inference: True means inference mode, False means training.
"""


class AgentStep(NamedTuple, Generic[_StepState]):
    """A batch of actions and updated hidden state."""

    action: jnp.ndarray
    state: _StepState


@pytree_dataclass
class EnvStep:
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


@dataclass(frozen=True)
class EnvInfo:
    num_envs: int
    observation_space: Space
    action_space: Space
    name: str


def env_info_from_gymnax(env: GymnaxEnv, params: EnvParams, num_envs: int) -> EnvInfo:
    return EnvInfo(num_envs, env.observation_space(params), env.action_space(params), env.name)


class Agent(abc.ABC, Generic[_Networks, _OptState, _ExperienceState, _StepState]):
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
    state = a.new_training_state(...)

    def run_cycle_and_loss(nets_to_grad, nets_no_grad, state, env_step):
        state = replace(state, nets=eqx.combine(nets_to_grad, nets_no_grad))
        trajectory = []
        for step in range(steps_per_cycle):
            action, state.hidden = a.select_action(state, env_step)
            env_step = env.step(action)
            trajectory.append(env_step)
        state.experience = a.update_experience(state, trajectory)
        return a.loss(state)

    env_step = env.reset()
    for cycle in range(num_cycles):
        nets_yes_grad, nets_no_grad = a.partition_for_grad(state.nets)
        grad = jax.grad(run_cycle_and_loss)(nets_yes_grad, nets_no_grad, state, env_step)
        if training:
            state = a.optimize_from_grads(state, grad)

    2. Off-policy training (if num_off_policy_updates_per_cycle() > 0):

    a = Agent(agent_config)
    env = vectorized_env(env_config)
    state = a.new_training_state(...)
    def run_cycle(state, env_step):
        trajectory = []
        for step in range(steps_per_cycle):
            action, state.hidden = a.select_action(state, env_step)
            env_step = env.step(action)
            trajectory.append(env_step)
        return state, trajectory

    def compute_loss(nets_yes_grad, nets_no_grad, state):
        state = replace(state, nets=eqx.combine(nets_yes_grad, nets_no_grad))
        return a.loss(state)

    env_step = env.reset()
    for cycle in range(num_cycles):
        state, trajectory = run_cycle(state, env_step)
        state.experience = a.update_experience(state, trajectory)
        for _ in range(a.num_off_policy_updates_per_cycle()):
            nets_to_grad, nets_no_grad = a.partition_for_grad(state.nets)
            grad = jax.grad(compute_loss)(nets_to_grad, nets_no_grad, state)
            state = a.optimize_from_grads(state, grad)

    """

    def new_state(
        self, nets: _Networks, env_info: EnvInfo, key: PRNGKeyArray, inference: bool = False
    ) -> AgentState[_Networks, _OptState, _ExperienceState, _StepState]:
        """Initializes agent state.

        Args:
            nets: the agent's neural networks. Donated, so callers should not access it after calling.
            key: a PRNG key. Used to generate keys for hidden, opt, and experience.
                Donated, so callers should not access it after calling.
            inference: if True, initialize the state for inference (opt and experience are None).
        """

        # helper funcion to wrap with jit.
        @eqx.filter_jit(donate="all")
        def _helper(nets: _Networks, env_info: EnvInfo, key: PRNGKeyArray):
            hidden_key, opt_key, replay_key = jax.random.split(key, 3)
            if inference:
                return AgentState(
                    step=self.new_step_state(nets, env_info, key),
                    nets=nets,
                    opt=None,
                    experience=None,
                    inference=True,
                )
            else:
                return AgentState(
                    step=self.new_step_state(nets, env_info, hidden_key),
                    nets=nets,
                    opt=self.new_opt_state(nets, env_info, opt_key),
                    experience=self.new_experience_state(nets, env_info, replay_key),
                )

        # Not sure if the type ignore is the best thing.
        # I want to avoid all implementors of _loss and _optimize_from_grads having to assert
        # that opt and replay are not None, so I don't want to mark the fields optional.
        # I tried adding making the fields optional and then adding another class, TrainingState,
        # that is the same as AgentState but with the optional fields marked non-optional.
        # However this would require implementors to deal with another type, so it's not clear
        # it's a win.
        return _helper(nets, env_info, key)  # type: ignore[return-value]

    def new_step_state(self, nets: _Networks, env_info: EnvInfo, key: PRNGKeyArray) -> _StepState:
        """Returns a new step state.

        Args:
            nets: the agent's neural networks.
            env_info: info about the environment.
            key: a PRNG key.
        """

        return eqx.filter_jit(self._new_step_state)(nets, env_info, key)

    @abc.abstractmethod
    def _new_step_state(self, nets: _Networks, env_info: EnvInfo, key: PRNGKeyArray) -> _StepState:
        """Returns a new step state.

        Must be jit-compatible.
        """

    def new_experience_state(self, nets: _Networks, env_info: EnvInfo, key: PRNGKeyArray) -> _ExperienceState:
        """Initializes experience state.

        Args:
            nets: the agent's neural networks.
            env_info: info about the environment.
            key: a PRNG key.
        """
        return eqx.filter_jit(self._new_experience_state)(nets, env_info, key)

    @abc.abstractmethod
    def _new_experience_state(self, nets: _Networks, env_info: EnvInfo, key: PRNGKeyArray) -> _ExperienceState:
        """Initializes replay state.

        If an agent is on-policy (doesn't have any experience state), it can just return None.

        Must be jit-compatible.
        """

    def new_opt_state(self, nets: _Networks, env_info: EnvInfo, key: PRNGKeyArray) -> _OptState:
        """Initializes optimizer state.

        Args:
            nets: the agent's neural networks.
            env_info: info about the environment.
            key: a PRNG key.
        """
        return eqx.filter_jit(self._new_opt_state)(nets, env_info, key)

    @abc.abstractmethod
    def _new_opt_state(self, nets: _Networks, env_info: EnvInfo, key: PRNGKeyArray) -> _OptState:
        """Initializes optimizer state.

        Must be jit-compatible.
        """

    def step(
        self, state: AgentState[_Networks, _OptState, _ExperienceState, _StepState], env_step: EnvStep
    ) -> AgentStep[_StepState]:
        """Selects a batch of actions and updates step state.

        Sub-classes should override _select_action. This method is a wrapper that adds jit-compilation.

        Args:
            state: The current agent state. Donated, so callers should not access it after calling.
            env_step: The current environment step. All fields are batched, so any vmap() should be done inside
                this method.

        Returns:
            AgentStep which contains the batch of actions and the updated hidden state.
        """

        @eqx.filter_jit(donate="all-except-first")
        # swap order of args so we can avoid donating env_step
        def select_action_jit(env_step, state):
            return self._step(state, env_step)

        return select_action_jit(env_step, state)

    @abc.abstractmethod
    def _step(
        self, state: AgentState[_Networks, _OptState, _ExperienceState, _StepState], env_step: EnvStep
    ) -> AgentStep[_StepState]:
        """Selects a batch of actions and updates hidden state.

        Must be jit-compatible.
        """

    def update_experience(
        self, state: AgentState[_Networks, _OptState, _ExperienceState, _StepState], trajectory: EnvStep
    ) -> _ExperienceState:
        """Observes a trajectory of environment timesteps and updates the experience state.

        Sub-classes should override _observe_trajectory. This method is a wrapper that adds jit-compilation.

        Args:
            state: The current agent state. State.experience is donated, so callers should not access it after calling.
            trajectory: A trajectory of env steps where the shape of each field is
            (num_envs, num_steps, *).

        Returns:
            The updated experience.
        """

        @eqx.filter_jit(donate="all-except-first")
        def _update_experience_jit(
            others: tuple[AgentState[_Networks, _OptState, _ExperienceState, _StepState], EnvStep],
            experience: _ExperienceState,
        ):
            agent_state_no_exp, trajectory = others
            agent_state = replace(agent_state_no_exp, experience=experience)
            return self._update_experience(agent_state, trajectory)

        exp = state.experience
        agent_state_no_exp = replace(state, experience=None)

        return _update_experience_jit((agent_state_no_exp, trajectory), exp)

    @abc.abstractmethod
    def _update_experience(
        self, state: AgentState[_Networks, _OptState, _ExperienceState, _StepState], trajectory: EnvStep
    ) -> _ExperienceState:
        """Observes a trajectory of environment timesteps and updates the experience state.

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

    def loss(self, state: AgentState[_Networks, _OptState, _ExperienceState, _StepState]) -> tuple[Scalar, Metrics]:
        """Returns loss and metrics. Called after some number of environment steps.

        State arg is donated, meaning callers should not access it after calling.

        Sub-classes should override _loss. This method is a wrapper that adds jit-compilation.

        Note: the returned metrics should not have any keys that conflict with gymnax_loop.MetricKey.
        """
        return eqx.filter_jit(donate="all")(self._loss)(state)

    @abc.abstractmethod
    def _loss(self, state: AgentState[_Networks, _OptState, _ExperienceState, _StepState]) -> tuple[Scalar, Metrics]:
        """Returns loss and metrics. Called after some number of environment steps.

        Must be jit-compatible.
        """

    def optimize_from_grads(
        self, state: AgentState[_Networks, _OptState, _ExperienceState, _StepState], nets_grads: PyTree
    ) -> AgentState[_Networks, _OptState, _ExperienceState, _StepState]:
        """Optimizes agent state based on gradients of the losses returned by self.loss().

        Sub-classes should override _optimize_from_grads. This method is a wrapper that adds jit-compilation.

        Args:
            state: The current agent state. Donated, so callers should not access it after calling.
            nets_grads is the gradient of the loss w.r.t. the agent's networks. Donated,
                so callers should not access it after calling.
        """
        return eqx.filter_jit(donate="all")(self._optimize_from_grads)(state, nets_grads)

    @abc.abstractmethod
    def _optimize_from_grads(
        self, state: AgentState[_Networks, _OptState, _ExperienceState, _StepState], nets_grads: PyTree
    ) -> AgentState[_Networks, _OptState, _ExperienceState, _StepState]:
        """Optimizes agent state based on gradients of the losses returned by self._loss().

        Must be jit-compatible.
        """

    @abc.abstractmethod
    def num_off_policy_optims_per_cycle(self) -> int:
        """Returns the number of off-policy optimization steps per cycle.

        An off-policy optimization step conceptually consists of:
        - grads = jax.grad(self.loss)(state)
        - state = self.optimize_from_grads(state, grads).

        If 0, then the framework should only do on-policy updates, which consist of:
        - grads = jax.grad(multiple calls to self.select_action(state), one call to self.loss(state))()
        - state = self.optimize_from_grads(state, grads)
        """


class ObserveTrajectory(Protocol):
    def __call__(self, env_steps: EnvStep, step_infos: dict[Any, Any], step_num: int) -> Metrics: ...

    """Args:
        env_steps: a trajectory of env timesteps where the shape of each field is
            (num_envs, num_steps, *).
        step_infos: the aggregated info returned from environment.step().
        step_num: number of steps taken prior to the trajectory.
    """
