import dataclasses
import pathlib
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import orbax.checkpoint as ocp
from gymnax import EnvParams

from research.earl.core import Agent, env_info_from_gymnax
from research.earl.environment_loop.gymnax_loop import GymnaxLoop
from research.earl.environment_loop.gymnax_loop import Result as LoopResult
from research.earl.environment_loop.gymnax_loop import State as LoopState
from research.earl.experiments.config import CheckpointRestoreMode, ExperimentConfig, Phase
from research.earl.logging.metric_key import MetricKey


def _checkpoint_save_args(agent: Agent, env_params: EnvParams, state: LoopResult) -> ocp.args.CheckpointArgs:
    agent_state = state.agent_state
    # We split the agent state into arrays and non-arrays because it seems
    # StandardSave/Restore has special handling for arrays, and when we use PyTreeSave/Restore
    # we get warnings about missing sharding metadata. Possibly a bug in orbax that we should file.
    agent_arrays, agent_non_arrays = eqx.partition(agent_state, eqx.is_array)
    _, agent_non_callable = eqx.partition(agent, lambda x: isinstance(x, Callable))
    env_arrays, env_non_arrays = eqx.partition(state.env_state, eqx.is_array)
    return ocp.args.Composite(
        agent=ocp.args.PyTreeSave(agent_non_callable),  # type: ignore[call-arg]
        agent_state_arrays=ocp.args.StandardSave(agent_arrays),  # type: ignore[call-arg]
        agent_state_non_arrays=ocp.args.PyTreeSave(agent_non_arrays),  # type: ignore[call-arg]
        env_params=ocp.args.PyTreeSave(env_params),  # type: ignore[call-arg]
        env_state_arrays=ocp.args.StandardSave(env_arrays),  # type: ignore[call-arg]
        env_state_non_arrays=ocp.args.PyTreeSave(env_non_arrays),  # type: ignore[call-arg]
        env_step=ocp.args.StandardSave(state.env_step),  # type: ignore[call-arg]
        # Have to use ArraySave because https://github.com/google/orbax/issues/1288
        num_envs=ocp.args.ArraySave(state.env_step.obs.shape[0]),  # type: ignore[call-arg]
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
    env_params: EnvParams,
    state: LoopResult,
) -> tuple[Agent, EnvParams, LoopResult]:
    step_num_to_restore = _step_num_to_restore(checkpoint_manager, restore_from_checkpoint)
    agent_arrays, agent_non_arrays = eqx.partition(state.agent_state, eqx.is_array)
    agent_callable, agent_non_callable = eqx.partition(agent, lambda x: isinstance(x, Callable))
    env_arrays, env_non_arrays = eqx.partition(state.env_state, eqx.is_array)
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


def _config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    config_dict = dataclasses.asdict(config)
    # Only flattens 1 level of nesting because that's
    # what seems to result in readable logs.
    flat_dict = {}
    for k, v in config_dict.items():
        assert isinstance(k, str)
        if isinstance(v, dict):
            for k2, v2 in v.items():
                flat_dict[f"{k}/{k2}"] = v2
        else:
            flat_dict[k] = v
    return flat_dict


def _new_checkpoint_manager(directory: str | pathlib.Path, opts: ocp.CheckpointManagerOptions) -> ocp.CheckpointManager:
    if opts.best_fn is None:
        opts.best_fn = lambda metrics: metrics[MetricKey.REWARD_MEAN_SMOOTH]
    return ocp.CheckpointManager(directory, options=opts)


def run_experiment(config: ExperimentConfig) -> LoopResult:
    """Runs an experiment as specified in config."""

    agent = config.new_agent()
    key = jax.random.PRNGKey(config.random_seed)
    env = config.new_env()
    env_params = config.env
    networks = config.new_networks()
    train_logger = config.new_metric_logger(Phase.TRAIN)
    train_observe_trajectory = config.new_observe_trajectory(Phase.TRAIN)
    train_key, key = jax.random.split(key)
    checkpoint_manager = None
    num_envs = config.num_envs
    # need to restore num_envs from checkpoint first so that we can get the shapes of the other
    # arrays to restore.
    if config.checkpoint:
        checkpoint_manager = _new_checkpoint_manager(config.checkpoint.directory, config.checkpoint.manager_options)
        if config.checkpoint.restore_from_checkpoint is not None:
            step_num_to_restore = _step_num_to_restore(checkpoint_manager, config.checkpoint.restore_from_checkpoint)
            num_envs = checkpoint_manager.restore(
                step_num_to_restore,
                args=ocp.args.Composite(
                    num_envs=ocp.args.ArrayRestore(num_envs),  # type: ignore[call-arg]
                ),
            )["num_envs"]
            num_envs = int(num_envs)
    train_loop = GymnaxLoop(
        env, env_params, agent, num_envs, key, logger=train_logger, observe_trajectory=train_observe_trajectory
    )
    env_info = env_info_from_gymnax(env, env_params, num_envs)
    train_agent_state = agent.new_state(networks, env_info, train_key)
    del networks

    train_loop_state = LoopState(train_agent_state)
    del train_agent_state

    if checkpoint_manager:
        assert config.checkpoint is not None
        if config.checkpoint.restore_from_checkpoint is not None:
            env_state, env_step = train_loop.reset_env()
            train_loop_state = LoopResult(
                agent_state=train_loop_state.agent_state, env_state=env_state, env_step=env_step
            )
            agent, env_params, train_loop_state = _restore_checkpoint(
                checkpoint_manager, config.checkpoint.restore_from_checkpoint, agent, env_params, train_loop_state
            )
        if not config.num_train_cycles:
            checkpoint_manager = None  # disable checkpointing

    config_dict = _config_to_dict(config)
    config.new_config_logger().write(config_dict)

    train_cycles_per_eval = config.num_train_cycles
    eval_loop = None
    if config.num_eval_cycles:
        train_cycles_per_eval = config.num_train_cycles // config.num_eval_cycles
        eval_key, key = jax.random.split(key)
        eval_loop = GymnaxLoop(
            env,
            env_params,
            agent,
            config.num_envs,
            eval_key,
            logger=config.new_metric_logger(Phase.EVAL),
            observe_trajectory=config.new_observe_trajectory(Phase.EVAL),
            inference=True,
        )

    for _ in range(config.num_eval_cycles or 1):
        if train_cycles_per_eval:
            train_loop_state, metrics = train_loop.run(train_loop_state, train_cycles_per_eval, config.steps_per_cycle)
            if checkpoint_manager:
                final_cycle_metrics = {k: v[-1] for k, v in metrics.items()}
                checkpoint_manager.save(
                    train_loop_state.step_num,
                    metrics=final_cycle_metrics,
                    args=_checkpoint_save_args(agent, env_params, train_loop_state),
                )
        if eval_loop:
            eval_new_state_key, key = jax.random.split(key)
            eval_agent_state = agent.new_state(train_loop_state.agent_state.nets, env_info, eval_new_state_key)
            eval_loop_state = LoopState(eval_agent_state, step_num=train_loop_state.step_num)
            eval_loop_state, _ = eval_loop.run(eval_loop_state, 1, config.steps_per_cycle)
            # we don't expect nets to be modified, but the memory is donated to loop.run,
            # so we need to move it back into train_agent_state.
            train_loop_state = dataclasses.replace(
                train_loop_state,
                agent_state=dataclasses.replace(train_loop_state.agent_state, nets=eval_loop_state.agent_state.nets),
            )

    if checkpoint_manager:
        checkpoint_manager.wait_until_finished()

    assert isinstance(train_loop_state, LoopResult)
    return train_loop_state
