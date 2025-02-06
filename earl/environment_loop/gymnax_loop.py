import dataclasses
import time
import typing
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from gymnax import EnvState
from gymnax.environments.environment import Environment as GymnaxEnvironment
from gymnax.environments.spaces import Discrete
from jax_loop_utils.metric_writers.interface import MetricWriter
from jaxtyping import PRNGKeyArray, PyTree, Scalar
from tqdm import tqdm

from earl.core import (
  Agent,
  AgentState,
  EnvStep,
  _ExperienceState,
  _Networks,
  _OptState,
  _StepState,
)
from earl.environment_loop import (
  ArrayMetrics,
  CycleResult,
  ObserveCycle,
  Result,
  State,
  StepCarry,
  no_op_observe_cycle,
)
from earl.environment_loop._common import (
  extract_metrics,
  pytree_leaf_means,
  raise_if_metric_conflicts,
  to_num_envs_first,
)
from earl.metric_key import MetricKey
from earl.utils.eqx_filter import filter_scan  # TODO: remove deps on research


@eqx.filter_grad(has_aux=True)
def _loss_for_cycle_grad(
  nets_yes_grad, nets_no_grad, other_agent_state, agent: Agent
) -> tuple[Scalar, ArrayMetrics]:
  # this is a free function so we don't have to pass self as first arg, since filter_grad
  # takes gradient with respect to the first arg.
  agent_state = dataclasses.replace(
    other_agent_state, nets=eqx.combine(nets_yes_grad, nets_no_grad)
  )
  loss, metrics = agent.loss(agent_state)
  raise_if_metric_conflicts(metrics)
  # inside jit, return values are guaranteed to be arrays
  mutable_metrics: ArrayMetrics = typing.cast(ArrayMetrics, dict(metrics))
  mutable_metrics[MetricKey.LOSS] = loss
  return loss, mutable_metrics


class GymnaxLoop:
  """Runs an Agent in a Gymnax environment.

  Runs an agent and a Gymnax environment for a certain number of cycles.
  Each cycle is some (caller-specified) number of environment steps. It supports three modes:
  * training=False: just environment steps, no agent updates. Useful for evaluation.
  * training=True, agent.num_off_policy_updates_per_cycle() == 0: one on-policy update per cycle.
    The entire cycle of environment interaction is used to calculate the gradient.
  * training=True, agent.num_off_policy_updates_per_cycle() > 0: the specified number
    of off-policy updates per cycle. The gradient is calculated only for the Agent.loss() call,
    not the interaction with the environment.
  """

  _PMAP_AXIS_NAME = "device"

  def __init__(
    self,
    env: GymnaxEnvironment,
    env_params: Any,
    agent: Agent,
    num_envs: int,
    key: PRNGKeyArray,
    metric_writer: MetricWriter,
    observe_cycle: ObserveCycle = no_op_observe_cycle,
    inference: bool = False,
    assert_no_recompile: bool = True,
    devices: typing.Sequence[jax.Device] | None = None,
  ):
    """Initializes the GymnaxLoop.

    Args:
        env: The environment.
        env_params: The environment's parameters.
        agent: The agent.
        num_envs: The number of environments to run in parallel.
        key: The PRNG key. Must not be shared with the agent state, as this will cause jax buffer
            donation errors.
        metric_writer: The metric writer to write metrics to.
        observe_cycle: A function that takes a CycleResult representing a final environment state
            and a trajectory of length steps_per_cycle and runs any custom logic on it.
        inference: If False, agent.update_for_cycle() will not be called.
        assert_no_recompile: Whether to fail if the inner loop gets compiled more than once.
        devices: sequence of devices for data parallelism.
            Each device will run num_envs, and the gradients will be averaged across devices.
            If not set, runs on jax.local_devices()[0].

    """
    self._env = env
    sample_key, key = jax.random.split(key)
    self._action_space = self._env.action_space(env_params)
    self._example_action = self._action_space.sample(sample_key)
    self._env_reset = partial(self._env.reset, params=env_params)
    self._env_step = partial(self._env.step, params=env_params)
    self._agent = agent
    self._num_envs = num_envs
    self._key = key
    self._metric_writer = metric_writer
    self._observe_cycle = observe_cycle
    self._inference = inference
    self._devices = devices or jax.local_devices()[0:1]
    _run_cycle_and_update = self._run_cycle_and_update
    # max_traces=2 because of https://github.com/patrick-kidger/equinox/issues/932
    if assert_no_recompile:
      _run_cycle_and_update = eqx.debug.assert_max_traces(_run_cycle_and_update, max_traces=2)
      self._run_cycle_and_update = eqx.filter_pmap(
        _run_cycle_and_update,
        donate="warn",
        axis_name=self._PMAP_AXIS_NAME,
        devices=self._devices,  # pyright: ignore[reportCallIssue]
      )

  def reset_env(self) -> tuple[EnvState, EnvStep]:
    """Resets the environment.

    Should probably not be called by most users.
    Exposed so that callers can get the env_state and env_step to restore from a checkpoint.
    """
    env_key, self._key = jax.random.split(self._key, 2)
    num_devices = len(self._devices)
    env_keys = jax.random.split(env_key, (num_devices, self._num_envs))
    obs, env_state = jax.pmap(
      jax.vmap(self._env_reset), axis_name=self._PMAP_AXIS_NAME, devices=self._devices
    )(env_keys)

    env_step = EnvStep(
      new_episode=jnp.ones((num_devices, self._num_envs), dtype=jnp.bool),
      obs=obs,
      prev_action=jnp.zeros(
        (num_devices, self._num_envs) + self._example_action.shape, dtype=self._example_action.dtype
      ),
      reward=jnp.zeros((num_devices, self._num_envs)),
    )
    return env_state, env_step

  def run(
    self,
    state: State[_Networks, _OptState, _ExperienceState, _StepState]
    | AgentState[_Networks, _OptState, _ExperienceState, _StepState],
    num_cycles: int,
    steps_per_cycle: int,
    print_progress: bool = True,
  ) -> Result[_Networks, _OptState, _ExperienceState, _StepState]:
    """Runs the agent for num_cycles cycles, each with steps_per_cycle steps.

    Args:
        state: The initial state. Donated, meaning callers should not access it
            after calling this function. They can instead use the returned state.
            Callers can pass in an AgentState, which is equivalent to passing in a LoopState
            with the same agent_state and all other fields set to their default values.
            state.agent_state will be replaced with
            `equinox.nn.inference_mode(agent_state, value=inference)` before running, where
            `inference` is the value that was passed into GymnaxLoop.__init__().
        num_cycles: The number of cycles to run.
        steps_per_cycle: The number of steps to run in each cycle.
        print_progress: Whether to print progress to std out.
        step_num_metric_start: The value to start the step number metric at. This is useful
            when calling this function multiple times. Typically should be incremented by
            num_cycles * steps_per_cycle between calls.


    Returns:
        The final loop state and a dictionary of metrics.
        Each metric is a list of length num_cycles. All returned metrics are per-cycle.

    """
    if num_cycles <= 0:
      raise ValueError("num_cycles must be positive.")
    if steps_per_cycle <= 0:
      raise ValueError("steps_per_cycle must be positive.")

    if isinstance(state, AgentState):
      state = State(state)

    agent_state = eqx.nn.inference_mode(state.agent_state, value=self._inference)
    agent_state = self.replicate(agent_state)

    if state.env_state is None:
      assert state.env_step is None
      env_state, env_step = self.reset_env()
    else:
      env_state = state.env_state
      assert state.env_step is not None
      env_step = state.env_step

    step_num_metric_start = state.step_num
    del state

    key = jax.random.split(self._key, len(self._devices))

    cycles_iter = range(num_cycles)
    if print_progress:
      cycles_iter = tqdm(cycles_iter, desc="cycles", unit="cycle", leave=False)

    for cycle_num in cycles_iter:
      cycle_start = time.monotonic()
      # We saw some environments inconsistently returning weak_type, which triggers recompilation
      # when the previous cycle's output is passed back in to _run_cycle, so strip all weak_type.
      # The astype looks like a no-op but it strips the weak_type.
      env_step = jax.tree.map(
        lambda x: x.astype(x.dtype) if isinstance(x, jax.Array) else x, env_step
      )
      env_state = typing.cast(
        EnvState,
        jax.tree.map(lambda x: x.astype(x.dtype) if isinstance(x, jax.Array) else x, env_state),
      )
      cycle_result = self._run_cycle_and_update(
        agent_state, env_state, env_step, key, steps_per_cycle
      )

      cycle_result_device_0 = jax.tree.map(
        lambda x: x[0] if isinstance(x, jax.Array) else x, cycle_result
      )
      observe_cycle_metrics = self._observe_cycle(cycle_result_device_0)
      if cycle_num == 0:  # Could be slow if lots of metrics, so just cycle 0
        raise_if_metric_conflicts(observe_cycle_metrics)

      agent_state, env_state, env_step, key = (
        cycle_result.agent_state,
        cycle_result.env_state,
        cycle_result.env_step,
        cycle_result.key,
      )

      metrics_by_type = extract_metrics(cycle_result.metrics, observe_cycle_metrics)
      metrics_by_type.scalar[MetricKey.DURATION_SEC] = time.monotonic() - cycle_start

      step_num = step_num_metric_start + (cycle_num + 1) * steps_per_cycle
      self._metric_writer.write_scalars(step_num, metrics_by_type.scalar)
      self._metric_writer.write_images(step_num, metrics_by_type.image)
      self._metric_writer.write_videos(step_num, metrics_by_type.video)

    self._key = key[0]

    return Result(
      agent_state, env_state, env_step, step_num_metric_start + num_cycles * steps_per_cycle
    )

  def _off_policy_update(
    self, agent_state: AgentState[_Networks, _OptState, _ExperienceState, _StepState], _
  ) -> tuple[AgentState[_Networks, _OptState, _ExperienceState, _StepState], ArrayMetrics]:
    nets_yes_grad, nets_no_grad = self._agent.partition_for_grad(agent_state.nets)
    nets_grad, metrics = _loss_for_cycle_grad(
      nets_yes_grad, nets_no_grad, dataclasses.replace(agent_state, nets=None), self._agent
    )
    nets_grad = jax.lax.pmean(nets_grad, axis_name=self._PMAP_AXIS_NAME)
    grad_means = pytree_leaf_means(nets_grad, "grad_mean")
    metrics.update(grad_means)
    agent_state = self._agent.optimize_from_grads(agent_state, nets_grad)
    return agent_state, metrics

  def _run_cycle_and_update(
    self,
    agent_state: AgentState[_Networks, _OptState, _ExperienceState, _StepState],
    env_state: EnvState,
    env_step: EnvStep,
    key: PRNGKeyArray,
    steps_per_cycle: int,
  ) -> CycleResult:
    # this is a nested function so we don't have to pass self as first arg, since filter_grad
    # takes gradient with respect to the first arg.
    @eqx.filter_grad(has_aux=True)
    def _run_cycle_and_loss_grad(
      nets_yes_grad: PyTree,
      nets_no_grad: PyTree,
      other_agent_state: AgentState[_Networks, _OptState, _ExperienceState, _StepState],
      env_state: EnvState,
      env_step: EnvStep,
      num_steps: int,
      key: PRNGKeyArray,
    ) -> tuple[Scalar, CycleResult]:
      agent_state = dataclasses.replace(
        other_agent_state, nets=eqx.combine(nets_yes_grad, nets_no_grad)
      )
      cycle_result = self._run_cycle(agent_state, env_state, env_step, num_steps, key)
      experience_state = self._agent.update_experience(
        cycle_result.agent_state, cycle_result.trajectory
      )
      agent_state = dataclasses.replace(cycle_result.agent_state, experience=experience_state)
      loss, metrics = self._agent.loss(agent_state)
      raise_if_metric_conflicts(metrics)
      # inside jit, return values are guaranteed to be arrays
      mutable_metrics: ArrayMetrics = typing.cast(ArrayMetrics, dict(metrics))
      mutable_metrics[MetricKey.LOSS] = loss
      mutable_metrics.update(cycle_result.metrics)
      return loss, dataclasses.replace(
        cycle_result, metrics=mutable_metrics, agent_state=agent_state
      )

    if not self._inference and not self._agent.num_off_policy_optims_per_cycle():
      # On-policy update. Calculate the gradient through the entire cycle.
      nets_yes_grad, nets_no_grad = self._agent.partition_for_grad(agent_state.nets)
      nets_grad, cycle_result = _run_cycle_and_loss_grad(
        nets_yes_grad,
        nets_no_grad,
        dataclasses.replace(agent_state, nets=None),
        env_state,
        env_step,
        steps_per_cycle,
        key,
      )
      nets_grad = jax.lax.pmean(nets_grad, axis_name=self._PMAP_AXIS_NAME)
      grad_means = pytree_leaf_means(nets_grad, "grad_mean")
      cycle_result.metrics.update(grad_means)
      agent_state = self._agent.optimize_from_grads(cycle_result.agent_state, nets_grad)
    else:
      cycle_result = self._run_cycle(agent_state, env_state, env_step, steps_per_cycle, key)
      agent_state = cycle_result.agent_state
      if not self._inference:
        experience_state = self._agent.update_experience(
          cycle_result.agent_state, cycle_result.trajectory
        )
        agent_state = dataclasses.replace(agent_state, experience=experience_state)

    metrics = cycle_result.metrics
    if not self._inference and self._agent.num_off_policy_optims_per_cycle():
      agent_state, off_policy_metrics = filter_scan(
        self._off_policy_update,
        init=agent_state,
        xs=None,
        length=self._agent.num_off_policy_optims_per_cycle(),
      )
      # Take mean of each metric.
      # This is potentially misleading, but not sure what else to do.
      metrics.update({k: jnp.mean(v) for k, v in off_policy_metrics.items()})

    return CycleResult(
      agent_state,
      cycle_result.env_state,
      cycle_result.env_step,
      cycle_result.key,
      metrics,
      cycle_result.trajectory,
      cycle_result.step_infos,
    )

  def _run_cycle(
    self,
    agent_state: AgentState[_Networks, _OptState, _ExperienceState, _StepState],
    env_state: EnvState,
    env_step: EnvStep,
    num_steps: int,
    key: PRNGKeyArray,
  ) -> CycleResult:
    """Runs self._agent and self._env for num_steps."""

    def scan_body(
      inp: StepCarry[_StepState], _
    ) -> tuple[StepCarry[_StepState], tuple[EnvStep, dict[Any, Any]]]:
      agent_state_for_step = dataclasses.replace(agent_state, step=inp.step_state)

      agent_step = self._agent.step(agent_state_for_step, inp.env_step)
      action = agent_step.action
      if isinstance(self._action_space, Discrete):
        one_hot_actions = jax.nn.one_hot(
          action, self._action_space.n, dtype=inp.action_counts.dtype
        )
        action_counts = inp.action_counts + jnp.sum(one_hot_actions, axis=0)
      else:
        action_counts = inp.action_counts
      env_key, key = jax.random.split(inp.key)
      env_keys = jax.random.split(env_key, self._num_envs)
      obs, env_state, reward, done, info = jax.vmap(self._env_step)(env_keys, inp.env_state, action)
      next_timestep = EnvStep(done, obs, action, reward)

      episode_steps = inp.episode_steps + 1

      # Update episode statistics
      completed_episodes = next_timestep.new_episode
      episode_length_sum = inp.complete_episode_length_sum + jnp.sum(
        episode_steps * completed_episodes
      )
      episode_count = inp.complete_episode_count + jnp.sum(completed_episodes, dtype=jnp.uint32)

      # Reset steps for completed episodes
      episode_steps = jnp.where(completed_episodes, jnp.zeros_like(episode_steps), episode_steps)

      total_reward = inp.total_reward + jnp.sum(next_timestep.reward)
      total_dones = inp.total_dones + jnp.sum(next_timestep.new_episode, dtype=jnp.uint32)

      return (
        StepCarry(
          env_step=next_timestep,
          env_state=env_state,
          step_state=agent_step.state,
          key=key,
          total_reward=total_reward,
          total_dones=total_dones,
          episode_steps=episode_steps,
          complete_episode_length_sum=episode_length_sum,
          complete_episode_count=episode_count,
          action_counts=action_counts,
        ),
        # NOTE: the very last env step in the last cycle is never returned in a trajectory.
        # I can't think of a clean way to do it, and losing a single step is unlikely to matter.
        (inp.env_step, info),
      )

    if isinstance(self._action_space, Discrete):
      action_counts = jnp.zeros(self._action_space.n, dtype=jnp.uint32)
    else:
      action_counts = jnp.array(0, dtype=jnp.uint32)

    final_carry, (trajectory, step_infos) = filter_scan(
      scan_body,
      init=StepCarry(
        env_step=env_step,
        env_state=env_state,
        step_state=agent_state.step,
        key=key,
        total_reward=jnp.array(0.0),
        total_dones=jnp.array(0, dtype=jnp.uint32),
        episode_steps=jnp.zeros(self._num_envs, dtype=jnp.uint32),
        complete_episode_length_sum=jnp.array(0, dtype=jnp.uint32),
        complete_episode_count=jnp.array(0, dtype=jnp.uint32),
        action_counts=action_counts,
      ),
      xs=None,
      length=num_steps,
    )

    agent_state = dataclasses.replace(agent_state, step=final_carry.step_state)
    # mean across complete episodes
    complete_episode_length_mean = jnp.where(
      final_carry.complete_episode_count > 0,
      final_carry.complete_episode_length_sum / final_carry.complete_episode_count,
      0,
    )
    assert isinstance(complete_episode_length_mean, jax.Array)

    metrics = {}
    # mean across environments
    metrics[MetricKey.COMPLETE_EPISODE_LENGTH_MEAN] = jnp.mean(complete_episode_length_mean)
    metrics[MetricKey.NUM_ENVS_THAT_DID_NOT_COMPLETE] = jnp.sum(
      final_carry.complete_episode_count == 0
    )
    metrics[MetricKey.TOTAL_DONES] = final_carry.total_dones
    metrics[MetricKey.REWARD_SUM] = final_carry.total_reward
    metrics[MetricKey.REWARD_MEAN] = final_carry.total_reward / self._num_envs
    metrics[MetricKey.ACTION_COUNTS] = final_carry.action_counts

    # Somewhat arbitrary, but flashbax expects (num_envs, num_steps, ...)
    # so we make it easier on users by transposing here.
    trajectory = jax.tree.map(to_num_envs_first, trajectory)

    return CycleResult(
      agent_state,
      final_carry.env_state,
      final_carry.env_step,
      final_carry.key,
      metrics,
      trajectory,
      step_infos,
    )

  def close(self):
    """Currently just for consistency with GymnasiumLoop."""
    pass

  def replicate(self, agent_state: AgentState) -> AgentState:
    """Replicates the agent state for data parallel training."""
    # Don't require the caller to replicate the agent state.
    agent_state_leaves = jax.tree.leaves(agent_state, is_leaf=eqx.is_array)
    assert agent_state_leaves
    assert isinstance(agent_state_leaves[0], jax.Array)
    if isinstance(agent_state_leaves[0].sharding, jax.sharding.SingleDeviceSharding):
      agent_state_arrays, agent_state_static = eqx.partition(agent_state, eqx.is_array)
      agent_state_arrays = jax.device_put_replicated(agent_state_arrays, self._devices)
      agent_state = eqx.combine(agent_state_arrays, agent_state_static)
    return agent_state
