import contextlib
import copy
import dataclasses
import logging
import os
import threading
import time
import typing

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.core import Env as GymnasiumEnv
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import Autoreset
from gymnax import EnvState
from gymnax.environments.spaces import Discrete
from jax_loop_utils.metric_writers.interface import MetricWriter
from jaxtyping import PRNGKeyArray, PyTree, Scalar
from tqdm import tqdm

try:
  import nvtx  # pyright: ignore[reportMissingImports]

  nvtx_annotate = nvtx.annotate  # pyright: ignore[reportAssignmentType]
except ImportError:

  @contextlib.contextmanager
  def nvtx_annotate(message: str):
    yield


from earl.core import (
  Agent,
  AgentState,
  EnvStep,
  _ExperienceState,
  _Networks,
  _OptState,
  _StepState,
  env_info_from_gymnasium,
)
from earl.environment_loop import (
  ArrayMetrics,
  CycleResult,
  ObserveCycle,
  Result,
  State,
  no_op_observe_cycle,
)
from earl.environment_loop._common import (
  extract_metrics,
  pytree_leaf_means,
  raise_if_metric_conflicts,
  to_num_envs_first,
)
from earl.metric_key import MetricKey
from earl.utils.eqx_filter import filter_scan

_logger = logging.getLogger(__name__)


# Cannot donate all because when stacking leaves for the trajectory
# because we still need the final carry to be on the GPU.
@eqx.filter_jit(donate="all")
def _stack_leaves(pytree_list: list[PyTree]) -> PyTree:
  pytree = jax.tree.map(lambda *leaves: jnp.stack(leaves), *pytree_list)
  pytree = jax.tree.map(to_num_envs_first, pytree)
  return pytree


@eqx.filter_jit
def _copy_pytree(pytree: PyTree) -> PyTree:
  return jax.tree.map(lambda x: x.copy(), pytree)


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


def _filter_device_put(x: PyTree[typing.Any], device: jax.Device | None):
  """Filtered version of `jax.device_put`.

  TODO: simplify now that I don't need to copy back to the update device

  The equinox docs suggest filter_shard should work, but it doesn't work
  for all cases it seems.

  **Arguments:**

  - `x`: A PyTree, with potentially a mix of arrays and non-arrays on the leaves.
      Arrays will be moved to the specified device.
  - `device`: Either a singular device (e.g. CPU or GPU) or PyTree of
      devices. The structure should be a prefix of `x`.

  **Returns:**

  A copy of `x` with the specified device.
  """
  dynamic, static = eqx.partition(x, eqx.is_array)
  dynamic = jax.device_put(dynamic, device)
  return eqx.combine(dynamic, static)


class _InferenceThread(threading.Thread):
  def __init__(
    self,
    target: typing.Callable[[AgentState, EnvStep, int, PRNGKeyArray], CycleResult],
    agent_state: AgentState,
    env_step: EnvStep,
    num_steps: int,
    key: PRNGKeyArray,
    run_on_device: jax.Device,
    copy_back_to_device: jax.Device,
    fake_thread: bool = False,
  ):
    """Initializes the _InferenceThread.

    Args:
        target: The target function to run.
        agent_state: The agent state to run the target function on.
        env_step: The environment step to run the target function on.
        num_steps: The number of steps to run the target function on.
        key: The PRNG key to run the target function on.
        run_on_device: The device to run the target function on.
        copy_back_to_device: The device to copy the result back to.
        fake_thread: Whether to fake the thread.
           Python debugger can't handle threads, so when debugging this can be very helpful.
    """
    super().__init__()
    self._target = target
    self._agent_state = agent_state
    self._env_step = env_step
    self._num_steps = num_steps
    self._key = key
    self._result: CycleResult | None = None
    self._run_on_device = run_on_device
    self._copy_back_to_device = copy_back_to_device
    self._fake_thread = fake_thread

  def run(self):
    agent_state, env_step, num_steps, key = _filter_device_put(
      (self._agent_state, self._env_step, self._num_steps, self._key), self._run_on_device
    )
    with jax.default_device(self._run_on_device):
      self._result = self._target(agent_state, env_step, num_steps, key)
    assert self._result is not None
    # don't copy the nets back to the update device
    self._result = dataclasses.replace(
      self._result, agent_state=dataclasses.replace(self._result.agent_state, nets=None)
    )
    self._result = _filter_device_put(self._result, self._copy_back_to_device)

  def start(self):
    if self._fake_thread:
      self.run()
    else:
      super().start()

  def join_and_return(self) -> CycleResult:
    if not self._fake_thread:
      self.join()
    assert self._result is not None
    return self._result


@contextlib.contextmanager
def _jax_platform_cpu():
  """Sets env var to force the JAX platform to CPU.

  This is a hacky way to force the JAX platform to CPU for subprocesses started
  by Gymnasium. This prevents the subprocesses from allocating GPU memory.

  Yields:
      Yields for the context manager
  """
  prev_value = os.environ.get("JAX_PLATFORM")
  os.environ["JAX_PLATFORMS"] = "cpu"
  try:
    yield
  finally:
    if prev_value is None:
      del os.environ["JAX_PLATFORMS"]
    else:
      os.environ["JAX_PLATFORMS"] = prev_value


class GymnasiumLoop:
  """Runs an Agent in a Gymnasium environment.

  Runs an agent and a Gymnasium environment for a certain number of cycles.
  Each cycle is some (caller-specified) number of environment steps. It supports three modes:
  * training=False: just environment steps, no agent updates. Useful for evaluation.
  * training=True, agent.num_off_policy_optims_per_cycle() > 0: the specified number
    of off-policy updates per cycle. The gradient is calculated only for the Agent.loss() call,
    not the interaction with the environment.
  * On-policy training is not supported (training=True, agent.num_off_policy_optims_per_cycle() = 0)
  """

  def __init__(
    self,
    env: GymnasiumEnv,
    agent: Agent,
    num_envs: int,
    key: PRNGKeyArray,
    metric_writer: MetricWriter,
    observe_cycle: ObserveCycle = no_op_observe_cycle,
    inference: bool = False,
    assert_no_recompile: bool = True,
    vectorization_mode: typing.Literal["sync", "async"] = "async",
    devices: typing.Sequence[jax.Device] | None = None,
  ):
    """Initializes the GymnasiumLoop.

    Args:
        env: The environment.
        agent: The agent.
        num_envs: The number of environments to run in parallel.
        key: The PRNG key.
        metric_writer: The metric writer to write metrics to.
        observe_cycle: A function that takes a CycleResult representing a final environment state
            and a trajectory of length steps_per_cycle and runs any custom logic on it.
        inference: If False, agent.optimize_from_grads() will not be called.
        assert_no_recompile: Whether to fail if the inner loop gets compiled more than once.
        vectorization_mode: Whether to create a synchronous or asynchronous vectorized environment
            from the provided Gymnasium environment.
        devices: The devices to use for the environment and agent.
            First device used for inference, second for updates.
            If None, will use jax.local_devices()[0] for both inference and updates.
    """
    env = Autoreset(env)  # run() assumes autoreset.

    def _env_factory() -> GymnasiumEnv:
      return copy.deepcopy(env)

    if vectorization_mode == "sync":
      self._env = SyncVectorEnv([_env_factory for _ in range(num_envs)])
    else:
      assert vectorization_mode == "async"
      # context="spawn" because others are unsafe with multithreaded JAX
      with _jax_platform_cpu():
        self._env = AsyncVectorEnv([_env_factory for _ in range(num_envs)], context="spawn")

    sample_key, key = jax.random.split(key)
    sample_key = jax.random.split(sample_key, num_envs)
    env_info = env_info_from_gymnasium(env, num_envs)
    self._action_space = env_info.action_space
    self._example_action = jax.vmap(self._action_space.sample)(sample_key)
    self._agent = agent
    self._num_envs = num_envs
    self._key = key
    self._metric_writer: MetricWriter = metric_writer
    self._observe_cycle = observe_cycle
    self._inference_only = inference
    update = self._update
    if assert_no_recompile:
      update = eqx.debug.assert_max_traces(update, max_traces=1)
    self._update = eqx.filter_jit(update, donate="warn")

    if assert_no_recompile:
      self._inference_cycle_bookkeeping = eqx.debug.assert_max_traces(
        self._inference_cycle_bookkeeping, max_traces=1
      )
    self._inference_cycle_bookkeeping = eqx.filter_jit(
      self._inference_cycle_bookkeeping, donate="warn-except-first"
    )

    if not self._inference_only and not self._agent.num_off_policy_optims_per_cycle():
      raise ValueError("On-policy training is not supported in GymnasiumLoop.")

    devices = devices or jax.local_devices()[0:1]
    self._inference_device: jax.Device = devices[0]
    self._update_device: jax.Device = devices[0]
    if len(devices) > 1:
      self._inference_device = devices[0]
      self._update_device = devices[1]
      if len(devices) > 2:
        _logger.warning(
          "Multiple update devices are not supported yet. Using only the first device."
        )

  def reset_env(self) -> EnvStep:
    """Resets the environment.

    Should probably not be called by most users.
    Exposed so that callers can get the env_step to restore from a checkpoint.
    """
    obs, _info = self._env.reset()

    return EnvStep(
      new_episode=jnp.ones((self._num_envs,), dtype=jnp.bool),
      obs=jnp.array(obs),
      prev_action=jnp.zeros_like(self._example_action),
      reward=jnp.zeros((self._num_envs,)),
    )

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
            `inference` is the value that was passed into GymnasiumLoop.__init__().
        num_cycles: The number of cycles to run.
        steps_per_cycle: The number of steps to run in each cycle.
        print_progress: Whether to print progress to std out.


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

    agent_state = eqx.nn.inference_mode(state.agent_state, value=self._inference_only)
    # everything needs to start on the update device
    agent_state = _filter_device_put(agent_state, self._update_device)
    env_step = _filter_device_put(self.reset_env(), self._update_device)
    self._key = jax.device_put(self._key, self._update_device)

    step_num_metric_start = state.step_num
    del state

    cycles_iter = range(num_cycles)
    if print_progress:
      cycles_iter = tqdm(cycles_iter, desc="cycles", unit="cycle", leave=False)

    env_state = typing.cast(EnvState, None)
    if self._inference_device != self._update_device:
      inference_thread = _InferenceThread(
        self._inference_cycle,
        # avoid copying the unnecessary state to the inference device
        dataclasses.replace(agent_state, experience=None, opt=None),
        env_step,
        steps_per_cycle,
        self._key,
        self._inference_device,
        self._update_device,
      )
      inference_thread.start()
    else:
      inference_thread = None
    inference_wait_duration = 0
    for cycle_num in cycles_iter:
      cycle_start = time.monotonic()
      agent_state_for_inference = dataclasses.replace(agent_state, experience=None, opt=None)
      if inference_thread:
        inference_result = inference_thread.join_and_return()
        inference_wait_duration = time.monotonic() - cycle_start
        if cycle_num < num_cycles - 1:
          inference_thread = _InferenceThread(
            self._inference_cycle,
            agent_state_for_inference,
            env_step,
            steps_per_cycle,
            self._key,
            self._inference_device,
            self._update_device,
          )
          inference_thread.start()
      else:
        inference_result = self._inference_cycle(
          agent_state_for_inference, env_step, steps_per_cycle, self._key
        )
      agent_state = dataclasses.replace(agent_state, step=inference_result.agent_state.step)
      cycle_result = dataclasses.replace(inference_result, agent_state=agent_state)

      if self._inference_only:
        update_duration = 0
      else:
        with jax.default_device(self._update_device):
          experience_state = self._agent.update_experience(agent_state, inference_result.trajectory)
          agent_state = dataclasses.replace(agent_state, experience=experience_state)
          agent_state, metrics = self._update(agent_state, inference_result.metrics)
          cycle_result = dataclasses.replace(
            inference_result, agent_state=agent_state, metrics=metrics
          )
        update_duration = time.monotonic() - cycle_start - inference_wait_duration
        if cycle_num == 1 and inference_wait_duration > 5 * update_duration:
          _logger.warning(
            "inference is much slower than update. inference duration: %fs, update duration: %fs",
            inference_wait_duration,
            update_duration,
          )

      env_state, env_step, self._key = (
        cycle_result.env_state,
        cycle_result.env_step,
        cycle_result.key,
      )

      observe_cycle_metrics = self._observe_cycle(cycle_result)
      # Could potentially be very slow if there are lots
      # of metrics to compare, so only do it once.
      if cycle_num == 0:
        raise_if_metric_conflicts(observe_cycle_metrics)

      metrics_by_type = extract_metrics(cycle_result.metrics, observe_cycle_metrics)
      metrics_by_type.scalar[MetricKey.DURATION_SEC] = time.monotonic() - cycle_start
      metrics_by_type.scalar[MetricKey.INFERENCE_WAIT_DURATION_SEC] = inference_wait_duration
      metrics_by_type.scalar[MetricKey.UPDATE_DURATION_SEC] = update_duration

      step_num = step_num_metric_start + (cycle_num + 1) * steps_per_cycle
      self._metric_writer.write_scalars(step_num, metrics_by_type.scalar)
      self._metric_writer.write_images(step_num, metrics_by_type.image)
      self._metric_writer.write_videos(step_num, metrics_by_type.video)

    return Result(
      agent_state, env_state, env_step, step_num_metric_start + num_cycles * steps_per_cycle
    )

  def _off_policy_update(
    self, agent_state: AgentState[_Networks, _OptState, _ExperienceState, _StepState], _
  ) -> tuple[AgentState[_Networks, _OptState, _ExperienceState, _StepState], ArrayMetrics]:
    nets_yes_grad, nets_no_grad = self._agent.partition_for_grad(agent_state.nets)
    grad, metrics = _loss_for_cycle_grad(
      nets_yes_grad, nets_no_grad, dataclasses.replace(agent_state, nets=None), self._agent
    )
    grad_means = pytree_leaf_means(grad, "grad_mean")
    metrics.update(grad_means)
    agent_state = self._agent.optimize_from_grads(agent_state, grad)
    return agent_state, metrics

  def _update(
    self,
    agent_state: AgentState[_Networks, _OptState, _ExperienceState, _StepState],
    metrics: ArrayMetrics,
  ) -> tuple[AgentState[_Networks, _OptState, _ExperienceState, _StepState], ArrayMetrics]:
    agent_state, off_policy_metrics = filter_scan(
      self._off_policy_update,
      init=agent_state,
      xs=None,
      length=self._agent.num_off_policy_optims_per_cycle(),
    )
    # Take mean of each metric.
    # This is potentially misleading, but not sure what else to do.
    metrics.update({k: jnp.mean(v) for k, v in off_policy_metrics.items()})

    return agent_state, metrics

  def _inference_cycle(
    self,
    agent_state: AgentState[_Networks, _OptState, _ExperienceState, _StepState],
    env_step: EnvStep,
    num_steps: int,
    key: PRNGKeyArray,
  ) -> CycleResult:
    """Runs self._agent and self._env for num_steps."""
    trajectory = []
    step_infos = []

    for _ in range(num_steps):
      with nvtx_annotate("inference loop body"):
        with nvtx_annotate("agent.step"):
          # implicit copy of env_step to device
          agent_step = self._agent.step(agent_state, env_step)
        agent_state = dataclasses.replace(agent_state, step=agent_step.state)

        with nvtx_annotate("env.step"):
          obs, reward, done, trunc, info = self._env.step(np.array(agent_step.action))
        env_step = EnvStep(done | trunc, obs, agent_step.action, reward)

        trajectory.append(env_step)
        step_infos.append(info)

    metrics, trajectory_stacked = self._inference_cycle_bookkeeping(trajectory=trajectory)

    with jax.default_device(jax.devices("cpu")[0]):
      step_infos_stacked = typing.cast(dict[typing.Any, typing.Any], _stack_leaves(step_infos))
    return CycleResult(
      agent_state,
      # not really the env state in any meaningful way, but we need to pass something
      # valid to make this code more similar to GymnaxLoop.
      EnvState(time=num_steps),
      env_step,
      key,
      metrics,
      trajectory_stacked,
      step_infos_stacked,
    )

  def _inference_cycle_bookkeeping(self, trajectory: list[EnvStep]) -> tuple[ArrayMetrics, EnvStep]:
    """Computes metrics."""
    trajectory_stacked = _stack_leaves(trajectory)

    # Shape: (num_envs, num_steps)
    completed_episodes = trajectory_stacked.new_episode
    # Shape: (num_envs,)
    total_dones = jnp.sum(completed_episodes, axis=1, dtype=jnp.uint32)
    # Shape: (num_steps,)
    step_indices = jnp.arange(1, len(trajectory) + 1, dtype=jnp.uint32)
    # Shape: (num_envs,)
    complete_episode_count = jnp.sum(completed_episodes, axis=1, dtype=jnp.uint32)
    # Shape: (num_envs,)
    complete_episode_length_sum = jnp.sum(
      step_indices[None, :] * completed_episodes, axis=1, dtype=jnp.uint32
    )

    if isinstance(self._action_space, Discrete):
      one_hot_actions = jax.nn.one_hot(
        trajectory_stacked.prev_action,
        self._action_space.n,
        dtype=jnp.uint32,
      )
      action_counts = jnp.sum(one_hot_actions, axis=(0, 1))
    else:
      action_counts = jnp.array(0, dtype=jnp.uint32)

    total_reward = jnp.sum(trajectory_stacked.reward)

    complete_episode_length_mean = jnp.where(
      complete_episode_count > 0,
      complete_episode_length_sum / complete_episode_count,
      0,
    )
    assert isinstance(complete_episode_length_mean, jax.Array)

    metrics: ArrayMetrics = {
      MetricKey.COMPLETE_EPISODE_LENGTH_MEAN: jnp.mean(complete_episode_length_mean),
      MetricKey.NUM_ENVS_THAT_DID_NOT_COMPLETE: jnp.sum(complete_episode_count == 0),
      MetricKey.TOTAL_DONES: jnp.sum(total_dones),
      MetricKey.REWARD_SUM: total_reward,
      MetricKey.REWARD_MEAN: total_reward / self._num_envs,
      MetricKey.ACTION_COUNTS: action_counts,
    }

    return metrics, trajectory_stacked

  def close(self):
    self._env.close()

  def replicate(self, agent_state: AgentState) -> AgentState:
    """Replicates the agent state for distributed training.

    This is a no-op because GymnasiumLoop does not support data parallel training.
    """
    return agent_state
