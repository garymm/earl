import os
import shutil

import gymnax
import gymnax.environments.spaces
import orbax.checkpoint as ocp
import pytest
from gymnax import EnvParams
from gymnax.environments.environment import Environment
from jaxtyping import PyTree

from research.earl.agents.uniform_random_agent import UniformRandom
from research.earl.core import Agent, Metrics
from research.earl.experiments.config import CheckpointConfig, CheckpointRestoreMode, ExperimentConfig, Phase
from research.earl.experiments.run_experiment import _new_checkpoint_manager, _restore_checkpoint, run_experiment
from research.earl.logging.base import MetricLogger
from research.earl.logging.metric_key import MetricKey


class AppendMetricLogger(MetricLogger):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def write(self, metrics: Metrics):
        self.metrics.append(metrics)

    def _close(self):
        self.metrics = []


class FakeExperimentConfig(ExperimentConfig):
    def __init__(self, env_obj: Environment, agent_obj: Agent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._env_obj = env_obj
        self._agent_obj = agent_obj

        self.train_logger = AppendMetricLogger()
        self.eval_logger = AppendMetricLogger()

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
    experiment = FakeExperimentConfig(
        env_obj=None,  # type: ignore[arg-type]
        agent_obj=None,  # type: ignore[arg-type]
        env=EnvParams(),
        num_eval_cycles=3,
        num_train_cycles=10,  # Not divisible by 3
        num_envs=1,
        random_seed=42,
        steps_per_cycle=100,
    )

    with pytest.raises(ValueError, match="num_train_cycles must be divisible by num_eval_cycles"):
        run_experiment(experiment)


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
    train_metrics = experiment.train_logger.metrics
    assert len(train_metrics) == experiment.num_train_cycles
    # Check that step numbers increase by steps_per_cycle
    for i, metrics in enumerate(train_metrics):
        assert metrics[MetricKey.STEP_NUM] == (i + 1) * experiment.steps_per_cycle

    if not num_eval_cycles:
        assert not experiment.eval_logger.metrics
    else:
        # Verify eval metrics
        eval_metrics = experiment.eval_logger.metrics
        assert len(eval_metrics) == experiment.num_eval_cycles
        # Check that eval happens at the right steps
        steps_between_evals = experiment.num_train_cycles // experiment.num_eval_cycles * experiment.steps_per_cycle
        for i, metrics in enumerate(eval_metrics):
            assert metrics[MetricKey.STEP_NUM] == (i + 1) * steps_between_evals + experiment.steps_per_cycle


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
    env, env_params = gymnax.make("CartPole-v1")
    agent = UniformRandom(env.action_space().sample, 0)
    max_to_keep = 2
    checkpoint_manager_options = ocp.CheckpointManagerOptions(save_interval_steps=1, max_to_keep=max_to_keep)
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoint_config = CheckpointConfig(
        checkpoints_dir,
        manager_options=checkpoint_manager_options,
    )

    num_train_cycles = 10
    steps_per_cycle = 2
    experiment = FakeExperimentConfig(
        env_obj=env,
        agent_obj=agent,
        env=env_params,
        num_eval_cycles=0,
        num_train_cycles=num_train_cycles,
        random_seed=42,
        num_envs=1,
        steps_per_cycle=steps_per_cycle,
        checkpoint=checkpoint_config,
    )

    run_experiment(experiment)

    checkpoint_dirs = os.listdir(checkpoints_dir)
    assert len(checkpoint_dirs) == 1  # when no eval cycles, just one checkpoint at the end
    latest_checkpoint_num = max(int(d) for d in checkpoint_dirs)
    assert latest_checkpoint_num == num_train_cycles * steps_per_cycle
    final_step_num = experiment.train_logger.metrics[-1][MetricKey.STEP_NUM]
    assert final_step_num == num_train_cycles * steps_per_cycle

    # reset
    experiment.train_logger.close()
    experiment.eval_logger.close()
    shutil.rmtree(checkpoints_dir)

    experiment.num_eval_cycles = 2
    run_experiment(experiment)

    checkpoint_dirs = os.listdir(checkpoints_dir)
    assert len(checkpoint_dirs) == experiment.num_eval_cycles
    latest_checkpoint_num = max(int(d) for d in checkpoint_dirs)
    assert latest_checkpoint_num == num_train_cycles * steps_per_cycle
    final_step_num = experiment.train_logger.metrics[-1][MetricKey.STEP_NUM]
    assert final_step_num == num_train_cycles * steps_per_cycle

    # Restore from latest
    assert experiment.checkpoint
    experiment.checkpoint.restore_from_checkpoint = CheckpointRestoreMode.LATEST
    loop_state = run_experiment(experiment)
    checkpoint_dirs = os.listdir(checkpoints_dir)
    assert len(checkpoint_dirs) == max_to_keep
    latest_checkpoint_num = max(int(d) for d in checkpoint_dirs)
    assert latest_checkpoint_num == 2 * num_train_cycles * steps_per_cycle

    # Restore from best
    best_reward, best_step = -float("inf"), None
    # If no best_fn is specified, the best step is the one with the highest reward_mean_smooth.
    for m in experiment.train_logger.metrics:
        # Orbax seems to use the latest one when there is a tie.
        if m[MetricKey.REWARD_MEAN_SMOOTH] >= best_reward:
            best_reward = m[MetricKey.REWARD_MEAN_SMOOTH]
            best_step = m[MetricKey.STEP_NUM]
    checkpoint_manager = _new_checkpoint_manager(checkpoints_dir, experiment.checkpoint.manager_options)
    _, _, restored_loop_state = _restore_checkpoint(
        checkpoint_manager, CheckpointRestoreMode.BEST, agent, env_params, loop_state
    )
    assert restored_loop_state.step_num == best_step

    # Restore from step
    step_to_restore = int(checkpoint_dirs[-2])
    _, _, restored_loop_state = _restore_checkpoint(checkpoint_manager, step_to_restore, agent, env_params, loop_state)
    assert restored_loop_state.step_num == step_to_restore


def test_restore_only_no_training(tmp_path):
    # Set up initial experiment to create a checkpoint
    env, env_params = gymnax.make("CartPole-v1")
    agent = UniformRandom(env.action_space().sample, 0)
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoint_config = CheckpointConfig(
        checkpoints_dir,
        manager_options=ocp.CheckpointManagerOptions(save_interval_steps=1),
    )

    # Run initial experiment to create checkpoint
    initial_experiment = FakeExperimentConfig(
        env_obj=env,
        agent_obj=agent,
        env=env_params,
        num_eval_cycles=1,
        num_train_cycles=10,
        random_seed=42,
        num_envs=1,
        steps_per_cycle=2,
        checkpoint=checkpoint_config,
    )
    initial_loop_state = run_experiment(initial_experiment)
    initial_checkpoint_dirs = set(os.listdir(checkpoints_dir))
    assert len(initial_checkpoint_dirs) == 1

    # Run restore-only experiment
    restore_experiment = FakeExperimentConfig(
        env_obj=env,
        agent_obj=agent,
        env=env_params,
        num_eval_cycles=1,
        num_train_cycles=0,  # No training cycles
        random_seed=42,
        num_envs=1,
        steps_per_cycle=2,
        checkpoint=CheckpointConfig(
            checkpoints_dir,
            manager_options=ocp.CheckpointManagerOptions(save_interval_steps=1),
            restore_from_checkpoint=CheckpointRestoreMode.LATEST,
        ),
    )
    restored_loop_state = run_experiment(restore_experiment)

    # Verify no new checkpoints were written
    final_checkpoint_dirs = set(os.listdir(checkpoints_dir))
    assert final_checkpoint_dirs == initial_checkpoint_dirs

    # Verify state was properly restored
    assert restored_loop_state.step_num == initial_loop_state.step_num
    assert len(restore_experiment.eval_logger.metrics) == 1  # One eval cycle was run
    assert len(restore_experiment.train_logger.metrics) == 0  # No training metrics
