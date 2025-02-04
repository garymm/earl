import dataclasses
import os
import shutil
from unittest.mock import patch

import chex
import gymnasium
import gymnax
import gymnax.environments.spaces
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import pytest
from gymnasium.core import Env as GymnasiumEnv
from gymnax import EnvParams
from gymnax.environments.environment import Environment as GymnaxEnv
from jax_loop_utils.metric_writers.memory_writer import MemoryWriter
from jaxtyping import PyTree

from earl.agents.random_agent.random_agent import RandomAgent
from earl.core import Agent, Image, Metrics, env_info_from_gymnasium
from earl.environment_loop.gymnasium_loop import GymnasiumLoop
from earl.experiments.config import CheckpointConfig, CheckpointRestoreMode, ExperimentConfig, MetricWriters
from earl.experiments.run_experiment import (
    _config_to_dict,
    _new_checkpoint_manager,
    _restore_checkpoint,
    run_experiment,
)
from earl.metric_key import MetricKey


class MockGymnasiumLoop(GymnasiumLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.close_called = False

    def close(self):
        super().close()
        self.close_called = True


class FakeExperimentConfig(ExperimentConfig):
    def __init__(self, env_obj: GymnasiumEnv | GymnaxEnv, agent_obj: Agent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._env_obj = env_obj
        self._agent_obj = agent_obj

        self.train_writer = MemoryWriter()
        self.eval_writer = MemoryWriter()

    def new_agent(self) -> Agent:
        return self._agent_obj

    def new_env(self) -> GymnasiumEnv | GymnaxEnv:
        return self._env_obj

    def new_networks(self) -> PyTree:
        return None

    def new_metric_writers(self) -> MetricWriters:
        assert ExperimentConfig.new_metric_writers(self)  # just for test coverage
        return MetricWriters(train=self.train_writer, eval=self.eval_writer)


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


@pytest.mark.parametrize("env_backend", ["gymnasium", "gymnax"])
@pytest.mark.parametrize("num_eval_cycles", [0, 2])
def test_run_experiment_no_eval_cycles(env_backend: str, num_eval_cycles: int):
    if env_backend == "gymnax":
        env, env_params = gymnax.make("CartPole-v1")
        action_space = env.action_space(env_params)  # pyright: ignore[reportArgumentType]
    else:
        env = gymnasium.make("CartPole-v1")
        env_params = None
        action_space = env_info_from_gymnasium(env, 1).action_space

    agent = RandomAgent(action_space.sample, 1)
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

    run_experiment(experiment)

    # Verify train metrics
    train_metrics = experiment.train_writer.scalars
    assert len(train_metrics) == experiment.num_train_cycles

    # Check that step numbers increase by steps_per_cycle
    step_nums = sorted(train_metrics.keys())
    assert all(step_nums[i] == (i + 1) * experiment.steps_per_cycle for i in range(len(step_nums)))

    if not num_eval_cycles:
        assert not experiment.eval_writer.scalars
    else:
        # Verify eval metrics
        eval_metrics = experiment.eval_writer.scalars
        assert len(eval_metrics) == experiment.num_eval_cycles

        # Check that eval happens at the right steps
        steps_between_evals = experiment.num_train_cycles // experiment.num_eval_cycles * experiment.steps_per_cycle
        step_nums = sorted(eval_metrics.keys())
        assert all(
            step_nums[i] == (i + 1) * steps_between_evals + experiment.steps_per_cycle for i in range(len(step_nums))
        )


@pytest.mark.parametrize("env_backend", ["gymnasium", "gymnax"])
def test_restore_with_no_checkpoint(env_backend: str, tmp_path):
    if env_backend == "gymnax":
        env, env_params = gymnax.make("CartPole-v1")
        action_space = env.action_space(env_params)  # pyright: ignore[reportArgumentType]
    else:
        env = gymnasium.make("CartPole-v1")
        env_params = None
        action_space = env_info_from_gymnasium(env, 1).action_space

    agent = RandomAgent(action_space.sample, 0)
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


@pytest.mark.parametrize("env_backend", ["gymnasium", "gymnax"])
def test_checkpointing(env_backend: str, tmp_path):
    def create_experiment(
        checkpoints_dir: str, num_eval_cycles: int, restore_from_checkpoint: CheckpointRestoreMode | int | None = None
    ):
        if env_backend == "gymnax":
            env, env_params = gymnax.make("CartPole-v1")
            action_space = env.action_space(env_params)  # pyright: ignore[reportArgumentType]
        else:
            env = gymnasium.make("CartPole-v1")
            env_params = None
            action_space = env_info_from_gymnasium(env, 1).action_space
        return FakeExperimentConfig(
            env_obj=env,
            agent_obj=RandomAgent(action_space.sample, 1),
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
    train_metrics = experiment.train_writer.scalars

    checkpoint_dirs = os.listdir(checkpoints_dir)
    assert len(checkpoint_dirs) == 1  # when no eval cycles, just one checkpoint at the end
    expected_step_num = experiment.num_train_cycles * experiment.steps_per_cycle
    assert max(int(d) for d in checkpoint_dirs) == expected_step_num
    step_nums = sorted(train_metrics.keys())
    assert step_nums[-1] == expected_step_num

    # reset
    shutil.rmtree(checkpoints_dir)

    experiment = create_experiment(checkpoints_dir, num_eval_cycles=2)
    run_experiment(experiment)
    train_metrics = experiment.train_writer.scalars

    checkpoint_dirs = os.listdir(checkpoints_dir)
    assert len(checkpoint_dirs) == experiment.num_eval_cycles
    expected_step_num = experiment.num_train_cycles * experiment.steps_per_cycle
    assert max(int(d) for d in checkpoint_dirs) == expected_step_num
    step_nums = sorted(train_metrics.keys())
    assert step_nums[-1] == expected_step_num

    # Restore from latest
    experiment = create_experiment(
        checkpoints_dir, num_eval_cycles=2, restore_from_checkpoint=CheckpointRestoreMode.LATEST
    )
    loop_state = run_experiment(experiment)
    train_metrics = experiment.train_writer.scalars

    checkpoint_dirs = os.listdir(checkpoints_dir)
    assert experiment.checkpoint
    assert len(checkpoint_dirs) == experiment.checkpoint.manager_options.max_to_keep
    expected_step_num = 2 * experiment.num_train_cycles * experiment.steps_per_cycle
    assert max(int(d) for d in checkpoint_dirs) == expected_step_num
    step_nums = sorted(train_metrics.keys())
    assert step_nums[-1] == expected_step_num

    # Restore from best
    best_reward, best_step = -float("inf"), None
    # If no best_fn is specified, the best step is the one with the highest reward_mean.
    for step_num, step_metrics in train_metrics.items():
        # Orbax seems to use the latest one when there is a tie.
        if step_metrics[MetricKey.REWARD_MEAN] >= best_reward:
            best_reward = step_metrics[MetricKey.REWARD_MEAN]
            best_step = step_num
    checkpoint_manager = _new_checkpoint_manager(checkpoints_dir, experiment.checkpoint.manager_options)
    _, _, restored_loop_state = _restore_checkpoint(
        checkpoint_manager, CheckpointRestoreMode.BEST, experiment._agent_obj, experiment.env, loop_state
    )
    assert restored_loop_state.step_num == best_step

    # Restore from step
    step_to_restore = int(checkpoint_dirs[-2])
    _, _, restored_loop_state = _restore_checkpoint(
        checkpoint_manager, step_to_restore, experiment._agent_obj, experiment.env, loop_state
    )
    assert restored_loop_state.step_num == step_to_restore


@pytest.mark.parametrize("env_backend", ["gymnasium", "gymnax"])
def test_error_on_restore_only_no_training(env_backend: str):
    if env_backend == "gymnax":
        env, env_params = gymnax.make("CartPole-v1")
        action_space = env.action_space(env_params)  # pyright: ignore[reportArgumentType]
    else:
        env = gymnasium.make("CartPole-v1")
        env_params = None
        action_space = env_info_from_gymnasium(env, 1).action_space

    agent = RandomAgent(action_space.sample, 0)
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


def test_run_experiment_closes_loops():
    """Test that run_experiment properly closes both train and eval loops."""
    env = gymnasium.make("CartPole-v1")
    action_space = env_info_from_gymnasium(env, 1).action_space

    agent = RandomAgent(action_space.sample, 1)
    experiment = FakeExperimentConfig(
        env_obj=env,
        agent_obj=agent,
        env=None,
        num_eval_cycles=2,  # Make sure we have eval cycles
        num_train_cycles=10,
        random_seed=42,
        num_envs=1,
        steps_per_cycle=2,
    )

    train_loop = None
    eval_loop = None

    # Override _new_gymnasium_loop to use our mock loop
    def mock_new_gymnasium_loop(
        env,
        env_params,
        agent,
        num_envs,
        key,
        metric_writer,
        observe_cycle,
        inference: bool = False,
        assert_no_recompile: bool = True,
    ):
        nonlocal train_loop, eval_loop
        loop = MockGymnasiumLoop(
            env, agent, num_envs, key, metric_writer, observe_cycle, inference, assert_no_recompile
        )
        if inference:
            eval_loop = loop
        else:
            train_loop = loop
        return loop

    with patch("earl.experiments.run_experiment._new_gymnasium_loop", mock_new_gymnasium_loop):
        _ = run_experiment(experiment)

    assert train_loop is not None
    assert train_loop.close_called

    assert eval_loop is not None
    assert eval_loop.close_called


def test_config_to_dict_flat():
    @dataclasses.dataclass
    class FlatConfig:
        str_value: str = "test"
        int_value: int = 42
        float_value: float = 3.14
        bool_value: bool = True
        none_value: None = None
        _private: str = "private"  # Should be excluded

    config = FlatConfig()
    result = _config_to_dict(config)

    assert result == {
        "str_value": "test",
        "int_value": 42,
        "float_value": 3.14,
        "bool_value": True,
        "none_value": None,
    }


def test_config_to_dict_nested():
    @dataclasses.dataclass
    class InnerConfig:
        value: int = 1

    @dataclasses.dataclass
    class OuterConfig:
        inner: InnerConfig
        value: str = "outer"

    config = OuterConfig(inner=InnerConfig())
    result = _config_to_dict(config)

    assert result == {"inner.value": 1, "value": "outer"}


def test_config_to_dict_non_config_value_type():
    @dataclasses.dataclass
    class ConfigWithArray:
        array: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.array([1, 2, 3]))
        value: int = 42

    config = ConfigWithArray()
    result = _config_to_dict(config)

    assert result == {"array": str(jnp.array([1, 2, 3])), "value": 42}


def test_config_to_dict_invalid_type():
    class NonDataclass:
        value: int = 42

    config = NonDataclass()
    with pytest.raises(ValueError, match="parameter obj must be a dict or dataclass but has type.*"):
        _config_to_dict(config)
