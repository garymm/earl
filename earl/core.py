"""Core types.

The goal of the Agent and AgentState classes is to be structured such that it is feasible to
implement distributed training following the Sebulba architecture described in
[Podracer architectures for scalable Reinforcement Learning](https://arxiv.org/abs/2104.06272),
and otherwise allowing users flexibility to implement things as they wish.
"""

import abc
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Generic, NamedTuple, Protocol, TypeVar

import equinox as eqx
import gymnasium.spaces as gym_spaces
import gymnax.environments.spaces as gymnax_spaces
import jax
import jax.numpy as jnp
from gymnasium.core import Env as GymnasiumEnv
from gymnasium.vector import VectorEnv
from gymnax.environments.environment import Environment as GymnaxEnv
from gymnax.environments.spaces import Space
from jaxtyping import PRNGKeyArray, PyTree, Scalar


class ConflictingMetricError(Exception):
  pass


class Image(eqx.Module):
  """When returning an image from observe_cycle(), wrap it in this class."""

  data: jax.Array


class Video(eqx.Module):
  """When returning a video from observe_cycle(), wrap it in this class."""

  data: jax.Array


_Networks = TypeVar("_Networks", bound=PyTree)
_OptState = TypeVar("_OptState")
_ActorState = TypeVar("_ActorState")
_ExperienceState = TypeVar("_ExperienceState")
# Keyed by name.
Metrics = Mapping[str, Scalar | float | int | Image | Video]


class SupportsStr(Protocol):
  def __str__(self) -> str: ...


# A collection of configuration values, keyed by name. AKA hyperparameters.
ConfigForLog = Mapping[str, SupportsStr]


class AgentState(eqx.Module, Generic[_Networks, _OptState, _ExperienceState, _ActorState]):
  """The state of an agent.

  To enable implementation of the Sebulba architecture, Earl requires the user to split their
  Agent's state into the fields in this class.
  The Agent method signatures enforce that the different pieces of state are read and written only
  when appropriate.

  All pytree leaves in the fields of any subclass must be one of the following types:
  bool, int, float, jax.Array. This allows them to be saved and restored
  by orbax.
  """

  actor: _ActorState
  """Actor state. This is read and written by the actor.

  This is also read by the learner when calculating the loss.
  In agents that use recurrent networks, this includes the recurrent hidden states.
  """
  nets: _Networks
  """Neural networks.

  This is read by the actor. It is read and written by the learner. Anything that needs a
  gradient computed needs to be in the networks.

  The nets must be a PyTree since it will be passed to equinox.combine().

  Any objects that need to change their behavior in inference or training mode should
  have a boolean member variable that is named "inference" for that purpose
  (all equinox.nn built-in modules have this).
  """
  opt: _OptState
  """Optimization state.

  Anything other than nets that also needs to be updated when optimizing
  (i.e. in optimize_from_grads()). This is where optimizer state belongs.

  Can be optax.OptState if that's all you need, or you can set it to a custom class."""
  experience: _ExperienceState
  """State based on trajectories accumulated by actors and sent learners.

  For agents that use experience replay, this contains replay buffers.
  """


class ActionAndState(NamedTuple, Generic[_ActorState]):
  """A batch of actions and updated actor state."""

  action: jax.Array
  state: _ActorState


class EnvStep(eqx.Module):
  """The result of taking an action in an environment.

  Note that it may be a batch of timesteps, in which case
  all of the members will have an additional leading batch dimension.
  """

  new_episode: jax.Array
  """Whether this is the first timestep in an episode.

  Because our environments automatically reset, this is true in 2 cases:
  1. The first timestep in an experiment (after an explicit env.reset()).
  2. The last timestep of an episode (since the the environment automatically
     resets and the observation is the first of the next episode).
  """
  obs: PyTree
  prev_action: jax.Array  # the action taken in the previous timestep
  reward: jax.Array


class EnvInfo(NamedTuple):
  num_envs: int
  observation_space: Space
  action_space: Space
  name: str


def env_info_from_gymnax(env: GymnaxEnv, params: Any, num_envs: int) -> EnvInfo:
  return EnvInfo(num_envs, env.observation_space(params), env.action_space(params), env.name)


def env_info_from_gymnasium(env: GymnasiumEnv | VectorEnv, num_envs: int) -> EnvInfo:
  if isinstance(env, VectorEnv):
    env_obs_space = env.single_observation_space
    env_action_space = env.single_action_space
  else:
    env_obs_space = env.observation_space
    env_action_space = env.action_space
  observation_space = _convert_gymnasium_space_to_gymnax_space(env_obs_space)
  action_space = _convert_gymnasium_space_to_gymnax_space(env_action_space)
  return EnvInfo(num_envs, observation_space, action_space, str(env))


# Need to define these functions up here instead of nested so that the
# functions are not reconstructed on each call.
@eqx.filter_jit(donate="all-except-first")
def _act_jit(
  non_donated: tuple[EnvStep, _Networks],
  actor_state: _ActorState,
  act: Callable[[_ActorState, _Networks, EnvStep], ActionAndState[_ActorState]],
) -> ActionAndState[_ActorState]:
  env_step, nets = non_donated
  return act(actor_state, nets, env_step)


@eqx.filter_jit(donate="all-except-first")
def _update_experience_jit(
  non_donated: tuple[EnvStep, _ActorState],
  experience: _ExperienceState,
  actor_state_pre: _ActorState,
  update_experience: Callable[
    [_ExperienceState, _ActorState, _ActorState, EnvStep],
    _ExperienceState,
  ],
) -> _ExperienceState:
  trajectory, actor_state_post = non_donated
  return update_experience(experience, actor_state_pre, actor_state_post, trajectory)


@eqx.filter_jit(donate="all")
def _partition_for_grad_jit(
  nets: _Networks, partition_for_grad: Callable[[_Networks], tuple[_Networks, _Networks]]
):
  return partition_for_grad(nets)


@eqx.filter_jit(donate="all")
def _optimize_from_grads_jit(
  nets: _Networks,
  opt_state: _OptState,
  nets_grads: PyTree,
  optimize_from_grads: Callable[[_Networks, _OptState, PyTree], tuple[_Networks, _OptState]],
):
  return optimize_from_grads(nets, opt_state, nets_grads)


@eqx.filter_jit(donate="warn-except-first")
def _loss_jit(
  non_donated: tuple[_Networks, _OptState],
  experience_state: _ExperienceState,
  loss: Callable[[_Networks, _OptState, _ExperienceState], tuple[Scalar, _ExperienceState]],
):
  nets, opt_state = non_donated
  return loss(nets, opt_state, experience_state)


@eqx.filter_jit(donate="warn")
def _prepare_for_actor_cycle_jit(
  actor_state: _ActorState, prepare_for_actor_cycle: Callable[[_ActorState], _ActorState]
) -> _ActorState:
  return prepare_for_actor_cycle(actor_state)


def _convert_gymnasium_space_to_gymnax_space(gym_space: gym_spaces.Space) -> gymnax_spaces.Space:
  """Convert a Gymnasium space into a Gymnax space.

  Args:
      gym_space: The Gymnasium space to convert.

  """
  dtype = jnp.dtype(gym_space.dtype)
  if isinstance(gym_space, gym_spaces.Box):
    return gymnax_spaces.Box(
      low=jnp.asarray(gym_space.low, dtype=dtype),
      high=jnp.asarray(gym_space.high, dtype=dtype),
      shape=gym_space.shape,
      dtype=dtype,
    )
  elif isinstance(gym_space, gym_spaces.Discrete):
    return gymnax_spaces.Discrete(num_categories=gym_space.n.item())
  else:
    raise ValueError(f"Unsupported Gymnasium space type: {type(gym_space)}")


class Agent(eqx.Module, Generic[_Networks, _OptState, _ExperienceState, _ActorState]):
  """Abstract class for a reinforcement learning agent.

  Sub-classes should:
   * Create a sub-class of AgentState that fills in the type variables with concrete types.
     See that type's docs for guideance.
   * Implement the abstract methods.

  Agents are equinox.Modules, which means they are also pytrees and dataclasses.
  Therefore you can use dataclass syntax when defining them, and __init__() will
  automatically be defined for you.

  E.g.:
  class MyAgent(Agent[Networks, CycleState, StepState]):
      foo: int = eqx.field(static=True)

  All pytree leaves in the fields of any subclass must be one of the following types:
  int, float, jax.Array, or str. This allows them to be saved and restored
  by orbax.

  The framework will conceptually do one of the following to train an agent:

  1. On-policy training (if num_off_policy_updates_per_cycle() == 0):

  a = Agent()
  env = vectorized_env(env_config)
  state = a.new_training_state(...)

  def run_cycle_and_loss(nets_to_grad, nets_no_grad, state, env_step):
      state = replace(state, nets=eqx.combine(nets_to_grad, nets_no_grad))
      trajectory = []
      for step in range(steps_per_cycle):
          action, state.actor = a.act(state, env_step)
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

  a = Agent()
  env = vectorized_env(env_config)
  state = a.new_training_state(...)
  def run_cycle(state, env_step):
      trajectory = []
      for step in range(steps_per_cycle):
          action, state.actor = a.act(state, env_step)
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

  env_info: EnvInfo

  def new_state(
    self, nets: _Networks, key: PRNGKeyArray
  ) -> AgentState[_Networks, _OptState, _ExperienceState, _ActorState]:
    """Initializes agent state.

    Args:
        nets: the agent's neural networks. Donated, so callers should not access it after calling.
        key: a PRNG key. Used to generate keys for actor, opt, and experience.
            Donated, so callers should not access it after calling.
    """

    # helper funcion to wrap with jit.
    @eqx.filter_jit(donate="all")
    def _helper(nets: _Networks, key: PRNGKeyArray):
      step_key, opt_key, replay_key = jax.random.split(key, 3)
      return AgentState(
        actor=self.new_actor_state(nets, step_key),
        nets=nets,
        opt=self.new_opt_state(nets, opt_key),
        experience=self.new_experience_state(nets, replay_key),
      )

    return _helper(nets, key)

  def new_actor_state(self, nets: _Networks, key: PRNGKeyArray) -> _ActorState:
    """Returns a new actor state.

    Args:
        nets: the agent's neural networks.
        env_info: info about the environment.
        key: a PRNG key.
    """
    return eqx.filter_jit(self._new_actor_state)(nets, key)

  @abc.abstractmethod
  def _new_actor_state(self, nets: _Networks, key: PRNGKeyArray) -> _ActorState:
    """Returns a new actor state.

    Must be jit-compatible.
    """

  def new_experience_state(self, nets: _Networks, key: PRNGKeyArray) -> _ExperienceState:
    """Initializes experience state.

    Args:
        nets: the agent's neural networks.
        key: a PRNG key.
    """
    return eqx.filter_jit(self._new_experience_state)(nets, key)

  @abc.abstractmethod
  def _new_experience_state(self, nets: _Networks, key: PRNGKeyArray) -> _ExperienceState:
    """Initializes replay state.

    If an agent is on-policy (doesn't have any experience state), it can just return None.

    Must be jit-compatible.
    """

  def new_opt_state(self, nets: _Networks, key: PRNGKeyArray) -> _OptState:
    """Initializes optimizer state.

    Args:
        nets: the agent's neural networks.
        env_info: info about the environment.
        key: a PRNG key.
    """
    return eqx.filter_jit(self._new_opt_state)(nets, key)

  @abc.abstractmethod
  def _new_opt_state(self, nets: _Networks, key: PRNGKeyArray) -> _OptState:
    """Initializes optimizer state.

    Must be jit-compatible.
    """

  def act(
    self, actor_state: _ActorState, nets: _Networks, env_step: EnvStep
  ) -> ActionAndState[_ActorState]:
    """Selects a batch of actions and updates actor state.

    Sub-classes should override _act. This method is a wrapper that adds jit-compilation.

    Args:
        state: The current agent state. Donated, so callers should not access it after calling.
        env_step: The current environment step. All fields are batched, so any vmap() should be done
          inside this method.

    Returns:
        AgentStep which contains the batch of actions and the updated step state.
    """
    # we only want to donate actor_state
    return _act_jit((env_step, nets), actor_state, self._act)

  @abc.abstractmethod
  def _act(
    self, actor_state: _ActorState, nets: _Networks, env_step: EnvStep
  ) -> ActionAndState[_ActorState]:
    """Selects a batch of actions and updates actor state.

    Must be jit-compatible.
    """

  def prepare_for_actor_cycle(self, actor_state: _ActorState) -> _ActorState:
    """Prepares actor state for a new cycle of acting.

    Generally reset any fixed-size buffers that fill up during a cycle.

    Sub-classes should override _prepare_for_actor_cycle. This method is a wrapper that adds
    jit-compilation.

    Args:
        actor_state: The actor state to prepare. Donated, so callers should not access it
          `after calling.

    Returns:
        The prepared actor state.
    """
    return _prepare_for_actor_cycle_jit(actor_state, self._prepare_for_actor_cycle)

  @abc.abstractmethod
  def _prepare_for_actor_cycle(self, actor_state: _ActorState) -> _ActorState:
    """Prepares actor state for a new cycle of acting.

    Must be jit-compatible.
    """

  def update_experience(
    self,
    experience_state: _ExperienceState,
    actor_state_pre: _ActorState,
    actor_state_post: _ActorState,
    trajectory: EnvStep,
  ) -> _ExperienceState:
    """Observes a trajectory of environment timesteps and updates the experience state.

    Sub-classes should override _update_experience. This method is a wrapper that adds
      jit-compilation.

    Args:
        experience_state: The state to be updated. Donated, so callers should not access it
          after calling.
        actor_state_pre: The actor state before the trajectory. Donated, so callers should
          not access it after calling.
        actor_state_post: The actor state after the trajectory.
        trajectory: A trajectory of env steps where the shape of each field is
          (num_envs, num_steps, *).

    Returns:
        The updated experience.
    """
    return _update_experience_jit(
      (trajectory, actor_state_post), experience_state, actor_state_pre, self._update_experience
    )

  @abc.abstractmethod
  def _update_experience(
    self,
    experience_state: _ExperienceState,
    actor_state_pre: _ActorState,
    actor_state_post: _ActorState,
    trajectory: EnvStep,
  ) -> _ExperienceState:
    """Observes a trajectory of environment timesteps and updates the experience state.

    Must be jit-compatible.
    """

  def partition_for_grad(self, nets: _Networks) -> tuple[_Networks, _Networks]:
    """Partitions nets into trainable and non-trainable parts.

    Nets arg is donated, meaning callers should not access it after calling.

    Sub-classes should override _partition_for_grad. This method is a wrapper that adds
    jit-compilation.

    Returns: A tuple of Networks, the first of which contains all the fields for which the gradients
        should be calculated, and the second contains the rest. They will be combined by with
        equinox.combine().
    """
    return _partition_for_grad_jit(nets, self._partition_for_grad)

  def _partition_for_grad(self, nets: _Networks) -> tuple[_Networks, _Networks]:
    """Partitions nets into trainable and non-trainable parts.

    Must be jit-compatible.

    Default implementation assumes nets is a PyTree and that all the
    array leaves are trainable. If this is not the case, this method should be overridden.
    """
    return eqx.partition(nets, eqx.is_array)

  def loss(
    self, nets: _Networks, opt_state: _OptState, experience_state: _ExperienceState
  ) -> tuple[Scalar, _ExperienceState]:
    """Returns loss and experience state. Called after some number of environment steps.

    Sub-classes should override _loss. This method is a wrapper that adds jit-compilation.

    The experience_state arg is donated, meaning callers should not access it after calling.
    """
    return _loss_jit((nets, opt_state), experience_state, self._loss)

  @abc.abstractmethod
  def _loss(
    self, nets: _Networks, opt_state: _OptState, experience_state: _ExperienceState
  ) -> tuple[Scalar, _ExperienceState]:
    """Returns loss and experience state. Called after some number of environment steps.

    Must be jit-compatible.
    """

  def optimize_from_grads(
    self, nets: _Networks, opt_state: _OptState, nets_grads: PyTree
  ) -> tuple[_Networks, _OptState]:
    """Optimizes agent state based on gradients of the losses returned by self.loss().

    Sub-classes should override _optimize_from_grads. This method is a wrapper that adds
    jit-compilation.

    Args:
        nets: The current network state. Donated, so callers should not access it after calling.
        opt_state: The current optimizer state. Donated, so callers should not access it after
            calling.
        nets_grads is the gradient of the loss w.r.t. the agent's networks. Donated,
            so callers should not access it after calling.
    """
    return _optimize_from_grads_jit(nets, opt_state, nets_grads, self._optimize_from_grads)

  @abc.abstractmethod
  def _optimize_from_grads(
    self, nets: _Networks, opt_state: _OptState, nets_grads: PyTree
  ) -> tuple[_Networks, _OptState]:
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

  @abc.abstractmethod
  def shard_actor_state(
    self, actor_state: _ActorState, learner_devices: Sequence[jax.Device]
  ) -> _ActorState:
    """Shards the actor state for distributed training using pmap.

    The output should have a leading axis equal to the number of devices so that it cam
    be passed to jax.pmap().

    Typically any per-environment state should be sharded using
    earl.utils.sharding.shard_along_axis_0() and other state should be replicated
    using jax.device_put_replicated().

    Args:
        actor_state: The actor state to shard.
        learner_devices: The devices to shard the actor state to.

    Returns:
        The sharded actor state.
    """
