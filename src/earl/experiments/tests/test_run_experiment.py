import os
import shutil

import chex
import gymnax
import gymnax.environments.spaces
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import pytest
from gymnax import EnvParams
from gymnax.environments.environment import Environment
from jaxtyping import PyTree

from research.earl.agents.uniform_random_agent import UniformRandom
from research.earl.core import Agent, Image, Metrics
from research.earl.experiments.config import CheckpointConfig, CheckpointRestoreMode, ExperimentConfig, Phase
from research.earl.experiments.run_experiment import _new_checkpoint_manager, _restore_checkpoint, run_experiment
from research.earl.logging import MemoryLogger
from research.earl.logging.base import MetricLogger
from research.earl.logging.metric_key import MetricKey
from research.utils.expect import expect_sequence


class FakeExperimentConfig(ExperimentConfig):
    def __init__(self, env_obj: Environment, agent_obj: Agent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._env_obj = env_obj
        self._agent_obj = agent_obj

        self.train_logger = MemoryLogger()
        self.eval_logger = MemoryLogger()

    def new_agent(self) -> Agent:
        return self._agent_obj

    def new_env(self) -> Environment:
        return self._env_obj

    def new_networks(self) -> PyTree:
        return None

    def new_metric_logger(self, phase: Phase) -> MetricLogger:
        assert ExperimentConfig.new_metric_logger(self, phase)  # just for test coverage
        return {"train": self.train_logger, "eval": self.eval_logger}[phase]


def test_run_experiment_num_train_cycles_not_divisible():
    with pytest.raises(ValueError, match="num_train_cycles must be divisible by num_eval_cycles"):
        FakeExperimentConfig(
            env_obj=None,  # type: ignore[arg-type]
            agent_obj=None,  # type: ignore[arg-type]
            env=EnvParams(),
            num_eval_cycles=3,
            num_train_cycles=10,  # Not divisible by 3
            num_envs=1,
            random_seed=42,
            steps_per_cycle=100,
        )


@pytest.mark.parametrize("num_eval_cycles", [0, 2])
def test_run_experiment_no_eval_cycles(num_eval_cycles: int):
    env, env_params = gymnax.make("CartPole-v1")
    agent = UniformRandom(env.action_space().sample, 0)
    experiment = FakeExperimentConfig(
        env_obj=env,
        agent_obj=agent,
        env=env_params,
        num_eval_cycles=num_eval_cycles,
        num_train_cycles=10,
        random_seed=42,
        num_envs=1,
        steps_per_cycle=2,
    )

    _ = run_experiment(experiment)

    # Verify train metrics
    train_metrics = experiment.train_logger.metrics()
    for _, val in train_metrics.items():
        assert len(val) == experiment.num_train_cycles

    # Check that step numbers increase by steps_per_cycle
    step_nums = train_metrics[MetricKey.STEP_NUM]
    assert all(step_nums[i] == (i + 1) * experiment.steps_per_cycle for i in range(len(step_nums)))

    if not num_eval_cycles:
        assert not experiment.eval_logger.metrics()
    else:
        # Verify eval metrics
        eval_metrics = experiment.eval_logger.metrics()
        for _, val in eval_metrics.items():
            assert len(val) == experiment.num_eval_cycles

        # Check that eval happens at the right steps
        steps_between_evals = experiment.num_train_cycles // experiment.num_eval_cycles * experiment.steps_per_cycle

        step_nums = eval_metrics[MetricKey.STEP_NUM]
        assert all(
            step_nums[i] == (i + 1) * steps_between_evals + experiment.steps_per_cycle for i in range(len(step_nums))
        )


def test_restore_with_no_checkpoint(tmp_path):
    env, env_params = gymnax.make("CartPole-v1")
    agent = UniformRandom(env.action_space().sample, 0)
    max_to_keep = 2
    checkpoint_manager_options = ocp.CheckpointManagerOptions(save_interval_steps=1, max_to_keep=max_to_keep)
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoint_config = CheckpointConfig(
        checkpoints_dir,
        manager_options=checkpoint_manager_options,
        restore_from_checkpoint=CheckpointRestoreMode.BEST,
    )
    experiment = FakeExperimentConfig(
        env_obj=env,
        agent_obj=agent,
        env=env_params,
        num_eval_cycles=0,
        num_train_cycles=10,
        random_seed=42,
        num_envs=1,
        steps_per_cycle=2,
        checkpoint=checkpoint_config,
    )
    with pytest.raises(ValueError, match="Did not find checkpoint"):
        run_experiment(experiment)


def test_checkpointing(tmp_path):
    def create_experiment(
        checkpoints_dir: str, num_eval_cycles: int, restore_from_checkpoint: CheckpointRestoreMode | int | None = None
    ):
        env, env_params = gymnax.make("CartPole-v1")

        return FakeExperimentConfig(
            env_obj=env,
            agent_obj=UniformRandom(env.action_space().sample, 0),
            env=env_params,
            num_eval_cycles=num_eval_cycles,
            num_train_cycles=10,
            random_seed=42,
            num_envs=1,
            steps_per_cycle=2,
            checkpoint=CheckpointConfig(
                checkpoints_dir,
                manager_options=ocp.CheckpointManagerOptions(save_interval_steps=1, max_to_keep=2),
                restore_from_checkpoint=restore_from_checkpoint,
            ),
        )

    checkpoints_dir = tmp_path / "checkpoints"

    experiment = create_experiment(checkpoints_dir, num_eval_cycles=0)
    run_experiment(experiment)
    train_metrics = experiment.train_logger.metrics()

    checkpoint_dirs = os.listdir(checkpoints_dir)
    assert len(checkpoint_dirs) == 1  # when no eval cycles, just one checkpoint at the end
    expected_step_num = experiment.num_train_cycles * experiment.steps_per_cycle
    assert max(int(d) for d in checkpoint_dirs) == expected_step_num
    assert train_metrics[MetricKey.STEP_NUM][-1] == expected_step_num

    # reset
    shutil.rmtree(checkpoints_dir)

    experiment = create_experiment(checkpoints_dir, num_eval_cycles=2)
    run_experiment(experiment)
    train_metrics = experiment.train_logger.metrics()

    checkpoint_dirs = os.listdir(checkpoints_dir)
    assert len(checkpoint_dirs) == experiment.num_eval_cycles
    expected_step_num = experiment.num_train_cycles * experiment.steps_per_cycle
    assert max(int(d) for d in checkpoint_dirs) == expected_step_num
    assert train_metrics[MetricKey.STEP_NUM][-1] == expected_step_num

    # Restore from latest
    experiment = create_experiment(
        checkpoints_dir, num_eval_cycles=2, restore_from_checkpoint=CheckpointRestoreMode.LATEST
    )
    loop_state = run_experiment(experiment)
    train_metrics = experiment.train_logger.metrics()

    checkpoint_dirs = os.listdir(checkpoints_dir)
    assert experiment.checkpoint
    assert len(checkpoint_dirs) == experiment.checkpoint.manager_options.max_to_keep
    expected_step_num = 2 * experiment.num_train_cycles * experiment.steps_per_cycle
    assert max(int(d) for d in checkpoint_dirs) == expected_step_num
    assert train_metrics[MetricKey.STEP_NUM][-1] == expected_step_num

    # Restore from best
    best_reward, best_step = -float("inf"), None
    # If no best_fn is specified, the best step is the one with the highest reward_mean.
    reward_mean = expect_sequence(train_metrics[MetricKey.REWARD_MEAN], float)
    step_nums = expect_sequence(train_metrics[MetricKey.STEP_NUM], int)
    for r, step_num in zip(reward_mean, step_nums, strict=False):
        # Orbax seems to use the latest one when there is a tie.
        if r >= best_reward:
            best_reward = r
            best_step = step_num
    checkpoint_manager = _new_checkpoint_manager(checkpoints_dir, experiment.checkpoint.manager_options)
    _, _, restored_loop_state, _ = _restore_checkpoint(
        checkpoint_manager, CheckpointRestoreMode.BEST, experiment._agent_obj, experiment.env, loop_state
    )
    assert restored_loop_state.step_num == best_step

    # Restore from step
    step_to_restore = int(checkpoint_dirs[-2])
    _, _, restored_loop_state, _ = _restore_checkpoint(
        checkpoint_manager, step_to_restore, experiment._agent_obj, experiment.env, loop_state
    )
    assert restored_loop_state.step_num == step_to_restore


def test_error_on_restore_only_no_training():
    env, env_params = gymnax.make("CartPole-v1")
    agent = UniformRandom(env.action_space().sample, 0)
    with pytest.raises(ValueError, match="num_train_cycles must be positive"):
        FakeExperimentConfig(
            env_obj=env,
            agent_obj=agent,
            env=env_params,
            num_eval_cycles=1,
            num_train_cycles=0,  # No training
            random_seed=42,
            num_envs=1,
            steps_per_cycle=2,
        )


def test_metric_serialization():
    path = ocp.test_utils.erase_and_create_empty("/tmp/my-checkpoints/")

    metrics: Metrics = {
        "int": 1,
        "float": 3.14,
        "float16[]": jnp.array([3.14, 6.9], dtype=jnp.float16),
        "imagergb8[]": Image(
            jnp.array(
                [
                    [[0, 0, 0], [255, 255, 255]],
                ],
                dtype=jnp.uint8,
            )
        ),
    }

    metrics_shape = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, metrics)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(path / "checkpoint_name", metrics)
    restored_metrics = checkpointer.restore(path / "checkpoint_name/", metrics_shape, strict=True)
    chex.assert_trees_all_equal(restored_metrics, metrics, strict=True)
