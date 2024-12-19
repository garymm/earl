import abc
import enum
import pathlib
from dataclasses import dataclass

import orbax.checkpoint as ocp
from gymnax import EnvParams
from gymnax.environments.environment import Environment
from jax_loop_utils.metric_writers.interface import MetricWriter
from jax_loop_utils.metric_writers.noop_writer import NoOpWriter
from jaxtyping import PyTree

from research.earl.core import Agent
from research.earl.environment_loop import ObserveCycle, no_op_observe_cycle


class CheckpointRestoreMode(enum.StrEnum):
    """Specifies which checkpoint to restore.

    LATEST: Use orbax.checkpoint.CheckpointManager.latest_step().
    BEST: Use orbax.checkpoint.CheckpointManager.best_step().
      Relies on CheckpointConfig.manager_options.best_fn.
    """

    LATEST = enum.auto()
    BEST = enum.auto()


@dataclass
class CheckpointConfig:
    directory: str | pathlib.Path
    manager_options: ocp.CheckpointManagerOptions
    restore_from_checkpoint: CheckpointRestoreMode | int | None = None


CheckpointConfig.__init__.__doc__ = """
Checkpoints

Args:
    directory: Directory to save checkpoints to.
    checkpoint_manager_options: Options for the checkpoint manager.
      If best_fn is not set, will use MetricKey.REWARD_MEAN.
    restore_from_checkpoint: If not None, the experiment will restore from this checkpoint.
      If int, the experiment will be interpreted as a (training) step number.
      Logging will use step numbers starting from the cycle after the restored cycle.
"""


@dataclass
class CycleObservers:
    train: ObserveCycle = no_op_observe_cycle
    eval: ObserveCycle = no_op_observe_cycle


@dataclass
class MetricWriters:
    train: MetricWriter = NoOpWriter()
    eval: MetricWriter = NoOpWriter()


@dataclass
class ExperimentConfig(abc.ABC):
    """Configures an experiment.

    In general anything that is a factory rather than just the object is so that
    the caller can easily run multiple experiments in parallel with minor changes
    (e.g., just changing random_seed).
    """

    @abc.abstractmethod
    def new_agent(self) -> Agent: ...

    @abc.abstractmethod
    def new_env(self) -> Environment: ...

    @abc.abstractmethod
    def new_networks(self) -> PyTree: ...

    def new_cycle_observers(self) -> CycleObservers:
        return CycleObservers()

    def new_metric_writers(self) -> MetricWriters:
        return MetricWriters()

    env: EnvParams
    num_eval_cycles: int
    num_train_cycles: int
    num_envs: int
    random_seed: int
    steps_per_cycle: int
    checkpoint: CheckpointConfig | None = None

    def __post_init__(self):
        if self.num_eval_cycles < 0:
            raise ValueError(f"num_eval_cycles must be non-negative. Got {self.num_eval_cycles}")
        if self.num_train_cycles <= 0:
            raise ValueError(f"num_train_cycles must be positive. Got {self.num_train_cycles}")
        if self.num_eval_cycles > 0 and self.num_train_cycles % self.num_eval_cycles != 0:
            raise ValueError(
                f"num_train_cycles must be divisible by num_eval_cycles. Got {self.num_train_cycles} and "
                f"{self.num_eval_cycles}"
            )


ExperimentConfig.__init__.__doc__ = """
Args:
    env: Environment parameters.
    num_eval_cycles: Number of cycles to evaluate the agent for. AgentState.inference will be set to True,
      and no optimization will be performed. Must divide num_train_cycles.
    num_train_cycles: Number of cycles to train the agent for. Must be divisible by num_eval_cycles.
    num_envs: Number of environments to run in parallel.
    random_seed: Random seed to use for the experiment.
    steps_per_cycle: Number of steps to run in each cycle.
    checkpoint: If not None, the experiment will save and / or restore checkpoints.
      NOTE: if restore_from_checkpoint_num is not None, any configuration that was stored
      will override the configuration in this ExperimentConfig.
      The step number passed to CheckpointManager.save() is a number of environment steps
      during training cycles. It is only called after an eval cycle.
      Example: if steps_per_cycle=3, num_train_cycles=10 and num_eval_cycles=2, save(5*3) and save(10*3) will be called.
      If num_eval_cycles is 0, CheckpointManager.save(num_train_cycles*steps_per_cycle) will be called after training.
"""
