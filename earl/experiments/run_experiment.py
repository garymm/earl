import dataclasses
import pathlib
from collections.abc import Callable, Iterable
from typing import Any

import equinox as eqx
import gymnax
import jax
import orbax.checkpoint as ocp
from gymnasium.core import Env as GymnasiumEnv
from jax_loop_utils.metric_writers import KeepLastWriter
from jax_loop_utils.metric_writers.interface import MetricWriter
from jaxtyping import PRNGKeyArray

from earl.core import Agent
from earl.environment_loop import ObserveCycle, no_op_observe_cycle
from earl.environment_loop.gymnasium_loop import GymnasiumLoop
from earl.environment_loop.gymnax_loop import GymnaxLoop
from earl.environment_loop.gymnax_loop import Result as LoopResult
from earl.environment_loop.gymnax_loop import State as LoopState
from earl.experiments.config import CheckpointRestoreMode, ExperimentConfig
from earl.metric_key import MetricKey


def _checkpoint_save_args(
  agent: Agent, env_params: Any, state: LoopResult
) -> ocp.args.CheckpointArgs:
  agent_state = state.agent_state
  num_envs = (
    state.env_step[0].obs.shape[0]
    if isinstance(state.env_step, list)
    else state.env_step.obs.shape[0]
  )
  # We split the agent state into arrays and non-arrays because it seems
  # StandardSave/Restore has special handling for arrays, and when we use PyTreeSave/Restore
  # we get warnings about missing sharding metadata. Possibly a bug in orbax that we should file.
  agent_arrays, agent_non_arrays = eqx.partition(agent_state, eqx.is_array)
  _, agent_non_callable = eqx.partition(agent, lambda x: isinstance(x, Callable))
  env_state = state.env_state or gymnax.EnvState(time=0)
  env_arrays, env_non_arrays = eqx.partition(env_state, eqx.is_array)
  return ocp.args.Composite(
    agent=ocp.args.PyTreeSave(agent_non_callable),  # type: ignore[call-arg]
    agent_state_arrays=ocp.args.StandardSave(agent_arrays),  # type: ignore[call-arg]
    agent_state_non_arrays=ocp.args.PyTreeSave(agent_non_arrays),  # type: ignore[call-arg]
    env_params=ocp.args.PyTreeSave(env_params),  # type: ignore[call-arg]
    env_state_arrays=ocp.args.StandardSave(env_arrays),  # type: ignore[call-arg]
    env_state_non_arrays=ocp.args.PyTreeSave(env_non_arrays),  # type: ignore[call-arg]
    env_step=ocp.args.StandardSave(state.env_step),  # type: ignore[call-arg]
    # Have to use ArraySave because https://github.com/google/orbax/issues/1288
    num_envs=ocp.args.ArraySave(num_envs),  # type: ignore[call-arg]
  )


def _step_num_to_restore(
  checkpoint_manager: ocp.CheckpointManager, restore_from_checkpoint: CheckpointRestoreMode | int
) -> int:
  step_num_to_restore = None
  if isinstance(restore_from_checkpoint, int):
    step_num_to_restore = restore_from_checkpoint
  elif restore_from_checkpoint == CheckpointRestoreMode.LATEST:
    step_num_to_restore = checkpoint_manager.latest_step()
  elif restore_from_checkpoint == CheckpointRestoreMode.BEST:
    step_num_to_restore = checkpoint_manager.best_step()
  if step_num_to_restore is None:
    raise ValueError(f"Did not find checkpoint to restore from: {restore_from_checkpoint}")
  return step_num_to_restore


def _restore_checkpoint(
  checkpoint_manager: ocp.CheckpointManager,
  restore_from_checkpoint: CheckpointRestoreMode | int,
  agent: Agent,
  env_params: Any,
  state: LoopResult,
) -> tuple[Agent, Any, LoopResult]:
  step_num_to_restore = _step_num_to_restore(checkpoint_manager, restore_from_checkpoint)
  agent_arrays, agent_non_arrays = eqx.partition(state.agent_state, eqx.is_array)
  agent_callable, agent_non_callable = eqx.partition(agent, lambda x: isinstance(x, Callable))
  env_state = state.env_state or gymnax.EnvState(time=0)
  env_arrays, env_non_arrays = eqx.partition(env_state, eqx.is_array)
  args = ocp.args.Composite(
    agent=ocp.args.PyTreeRestore(agent_non_callable),  # type: ignore[call-arg]
    agent_state_arrays=ocp.args.StandardRestore(agent_arrays),  # type: ignore[call-arg]
    agent_state_non_arrays=ocp.args.PyTreeRestore(agent_non_arrays),  # type: ignore[call-arg]
    env_params=ocp.args.PyTreeRestore(env_params),  # type: ignore[call-arg]
    env_state_arrays=ocp.args.StandardRestore(env_arrays),  # type: ignore[call-arg]
    env_state_non_arrays=ocp.args.PyTreeRestore(env_non_arrays),  # type: ignore[call-arg]
    env_step=ocp.args.StandardRestore(state.env_step),  # type: ignore[call-arg]
  )
  restored = checkpoint_manager.restore(step_num_to_restore, args=args)
  agent_state = eqx.combine(restored["agent_state_arrays"], restored["agent_state_non_arrays"])
  env_state = eqx.combine(restored["env_state_arrays"], restored["env_state_non_arrays"])
  agent = eqx.combine(agent_callable, restored["agent"])
  return (
    agent,
    restored["env_params"],
    LoopResult(
      agent_state=agent_state,
      env_state=env_state,
      env_step=restored["env_step"],
      step_num=step_num_to_restore,
    ),
  )


# Type comes from
# https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_hparams.
_ConfigValueType = str | bool | int | float | None


def _is_dataclass_instance(obj):
  return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def _dataclass_to_dict(obj):
  return dict((field.name, getattr(obj, field.name)) for field in dataclasses.fields(obj))


def _config_to_dict(obj, prefix="") -> dict[str, _ConfigValueType]:
  out = {}

  kv_iter: Iterable[tuple[object, object]]
  if isinstance(obj, dict):
    kv_iter = obj.items()
  elif _is_dataclass_instance(obj):
    kv_iter = _dataclass_to_dict(obj).items()
  else:
    raise ValueError(f"parameter obj must be a dict or dataclass but has type {type(obj)}")

  for k_outer, v_outer in kv_iter:
    # We are iterating arbitrary objects and the convention is that fields
    # starting with `_` are private. We assume that these fields should not
    # be serialized.
    if not isinstance(k_outer, str) or k_outer.startswith("_"):
      continue

    k_outer = f"{prefix}.{k_outer}" if prefix else f"{k_outer}"

    if isinstance(v_outer, dict) or _is_dataclass_instance(v_outer):
      for k_inner, v_inner in _config_to_dict(v_outer, prefix=k_outer).items():
        out[k_inner] = v_inner
    else:
      out[k_outer] = v_outer if isinstance(v_outer, _ConfigValueType) else str(v_outer)

  return out


def _new_checkpoint_manager(
  directory: str | pathlib.Path, opts: ocp.CheckpointManagerOptions
) -> ocp.CheckpointManager:
  if opts.best_fn is None:
    opts.best_fn = lambda metrics: metrics[MetricKey.REWARD_MEAN]
  return ocp.CheckpointManager(directory, options=opts)


def _new_gymnasium_loop(
  env: GymnasiumEnv,
  env_params: Any,
  agent: Agent,
  num_envs: int,
  key: PRNGKeyArray,
  metric_writer: MetricWriter,
  observe_cycle: ObserveCycle = no_op_observe_cycle,
  actor_only: bool = False,
  assert_no_recompile: bool = True,
  actor_devices: list[jax.Device] | None = None,
  learner_devices: list[jax.Device] | None = None,
) -> GymnasiumLoop:
  return GymnasiumLoop(
    env,
    agent,
    num_envs,
    key,
    metric_writer,
    observe_cycle,
    actor_only=actor_only,
    assert_no_recompile=assert_no_recompile,
    actor_devices=actor_devices,
    learner_devices=learner_devices,
  )


def _new_gymnax_loop(
  env: gymnax.environments.environment.Environment,
  env_params: Any,
  agent: Agent,
  num_envs: int,
  key: PRNGKeyArray,
  metric_writer: MetricWriter,
  observe_cycle: ObserveCycle = no_op_observe_cycle,
  actor_only: bool = False,
  assert_no_recompile: bool = True,
  actor_devices: list[jax.Device] | None = None,
  learner_devices: list[jax.Device] | None = None,
) -> GymnaxLoop:
  return GymnaxLoop(
    env,
    env_params,
    agent,
    num_envs,
    key,
    metric_writer,
    observe_cycle,
    actor_only=actor_only,
    assert_no_recompile=assert_no_recompile,
    devices=learner_devices,
  )


def run_experiment(config: ExperimentConfig) -> LoopResult:
  """Runs an experiment as specified in config."""
  agent = config.new_agent()
  key = jax.random.PRNGKey(config.random_seed)
  env = config.new_env()
  env_params = config.env
  if env_params is None:
    env_params = "dummy_env_params"  # valid pytree for snapshot / restore
  networks = config.new_networks()
  metric_writers = config.new_metric_writers()
  train_metric_writer = KeepLastWriter(metric_writers.train)
  observe_cycles = config.new_cycle_observers()
  train_observe_cycle = observe_cycles.train
  train_key, key = jax.random.split(key)
  checkpoint_manager = None
  num_envs = config.num_envs
  # need to restore num_envs from checkpoint first so that we can get the shapes of the other
  # arrays to restore.
  if config.checkpoint:
    checkpoint_manager = _new_checkpoint_manager(
      config.checkpoint.directory, config.checkpoint.manager_options
    )
    if config.checkpoint.restore_from_checkpoint is not None:
      step_num_to_restore = _step_num_to_restore(
        checkpoint_manager, config.checkpoint.restore_from_checkpoint
      )
      num_envs = checkpoint_manager.restore(
        step_num_to_restore,
        args=ocp.args.Composite(
          num_envs=ocp.args.ArrayRestore(num_envs),  # type: ignore[call-arg]
        ),
      )["num_envs"]
      num_envs = int(num_envs)

  if isinstance(env, GymnasiumEnv):
    loop_factory = _new_gymnasium_loop
  else:
    assert isinstance(env, gymnax.environments.environment.Environment)
    loop_factory = _new_gymnax_loop

  train_loop: GymnasiumLoop | GymnaxLoop = loop_factory(
    env,  # pyright: ignore[reportArgumentType]
    env_params,
    agent,
    config.num_envs,
    train_key,
    metric_writer=train_metric_writer,
    observe_cycle=train_observe_cycle,
    actor_devices=config.jax_actor_devices(),
    learner_devices=config.jax_learner_devices(),
  )

  train_agent_state = agent.new_state(networks, train_key)
  del networks

  train_agent_state = train_loop.replicate(train_agent_state)

  train_loop_state = LoopState(train_agent_state)
  del train_agent_state

  if checkpoint_manager:
    assert config.checkpoint is not None
    if config.checkpoint.restore_from_checkpoint is not None:
      if isinstance(train_loop, GymnaxLoop):
        env_state, env_step = train_loop.reset_env()
      else:
        env_state, env_step = None, train_loop.reset_env()
      train_loop_state = LoopResult(train_loop_state.agent_state, env_state, env_step)
      agent, env_params, train_loop_state = _restore_checkpoint(
        checkpoint_manager,
        config.checkpoint.restore_from_checkpoint,
        agent,
        env_params,
        train_loop_state,
      )
    if not config.num_train_cycles:
      checkpoint_manager = None  # disable checkpointing

  config_dict = _config_to_dict(config)
  train_metric_writer.write_hparams(config_dict)

  train_cycles_per_eval = config.num_train_cycles
  eval_loop = None
  if config.num_eval_cycles:
    train_cycles_per_eval = config.num_train_cycles // config.num_eval_cycles
    eval_key, key = jax.random.split(key)
    eval_loop = loop_factory(
      env,  # pyright: ignore[reportArgumentType]
      env_params,
      agent,
      config.num_envs,
      eval_key,
      metric_writer=metric_writers.eval,
      observe_cycle=observe_cycles.eval,
      actor_only=True,
    )

  eval_loop_state = None

  for _ in range(config.num_eval_cycles or 1):
    if train_cycles_per_eval:
      train_loop_state = train_loop.run(
        train_loop_state, train_cycles_per_eval, config.steps_per_cycle
      )
      if checkpoint_manager:
        checkpoint_manager.save(
          train_loop_state.step_num,
          metrics=train_metric_writer.scalars,
          args=_checkpoint_save_args(agent, env_params, train_loop_state),
        )
    if eval_loop:
      eval_new_state_key, key = jax.random.split(key)
      eval_agent_state = agent.new_state(train_loop_state.agent_state.nets, eval_new_state_key)
      eval_loop_state = LoopState(eval_agent_state, step_num=train_loop_state.step_num)
      eval_loop_state = eval_loop.run(eval_loop_state, 1, config.steps_per_cycle)
      # we don't expect nets to be modified, but the memory is donated to loop.run,
      # so we need to move it back into train_agent_state.
      train_loop_state = dataclasses.replace(
        train_loop_state,
        agent_state=dataclasses.replace(
          train_loop_state.agent_state, nets=eval_loop_state.agent_state.nets
        ),
      )

  if checkpoint_manager:
    checkpoint_manager.close()

  train_loop.close()
  if eval_loop:
    eval_loop.close()

  if train_cycles_per_eval:
    assert isinstance(train_loop_state, LoopResult)
    return train_loop_state
  else:  # eval only
    assert isinstance(eval_loop_state, LoopResult)
    return eval_loop_state
