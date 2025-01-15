import abc
import enum
import pathlib
import typing
from dataclasses import dataclass

import draccus.choice_types
import jax
import orbax.checkpoint as ocp
from draccus.parsers.decoding import decode as draccus_decode
from gymnasium.core import Env as GymnasiumEnv
from gymnax.environments.environment import Environment as GymnaxEnv
from jax_loop_utils.metric_writers.interface import MetricWriter
from jax_loop_utils.metric_writers.noop_writer import NoOpWriter
from jaxtyping import PyTree

from research.earl.core import Agent
from research.earl.environment_loop import ObserveCycle, no_op_observe_cycle

# enable draccus to parse jax arrays from strings
draccus_decode.register(jax.Array, jax.numpy.asarray)  # pyright: ignore[reportFunctionMemberAccess]


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


class EnvConfig(draccus.choice_types.ChoiceRegistry):
    """Environment configuration

    This is a special "choice type", which means draccus will search
    for env.type and use that to decide the actual type to parse the
    rest of the env.* into.

    Users of this code should register their real configuration types
    with:

    EnvConfig.register_subclass("my_env_type", MyEnvConfig), or

    @EnvConfig.register_subclass("my_env_type")
    @dataclass
    class MyEnvConfig(EnvConfig):
        ...

    Note that the class passed to register_subclass() does not actually
    inherit from EnvConfig.

    Then when ExperimentConfig is parsed, the env.type field will be
    used to determine the actual type to parse the rest of the env.* into.

    You can use EnvConfig.get_choice_name(type(env)) to go from the type
    to the registered name.
    """

    pass


class AutoDeviceSelector(enum.StrEnum):
    ALL = enum.auto()
    """Use all of jax.local_devices()"""


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
    def new_env(self) -> GymnaxEnv | GymnasiumEnv: ...

    @abc.abstractmethod
    def new_networks(self) -> PyTree: ...

    def new_cycle_observers(self) -> CycleObservers:
        return CycleObservers()

    def new_metric_writers(self) -> MetricWriters:
        return MetricWriters()

    def jax_devices(self) -> list[jax.Device]:
        local_devices = jax.local_devices()
        if isinstance(self.devices, int):
            return local_devices[self.devices : self.devices + 1]
        elif isinstance(self.devices, list):
            return [local_devices[i] for i in self.devices]
        else:
            assert self.devices == AutoDeviceSelector.ALL
            return local_devices

    num_eval_cycles: int
    """Number of cycles to evaluate the agent for.

    AgentState.inference will be set to True,
    and no optimization will be performed. Must divide num_train_cycles."""
    num_train_cycles: int
    """Number of cycles to train the agent for. Must be divisible by num_eval_cycles."""
    num_envs: int
    """Number of environments to run in parallel."""
    random_seed: int
    """Random seed to use for the experiment."""
    steps_per_cycle: int
    """Number of steps to run in each cycle."""
    env: EnvConfig
    """Environment configuration.

    The type annotation is just to activate the
    choice type functionality in draccus, but nothing actually
    enforces that the field is a subclass of EnvConfig.
    """
    checkpoint: CheckpointConfig | None = None
    """If not None, the experiment will save and / or restore checkpoints.

    NOTE: if restore_from_checkpoint_num is not None, any configuration that was stored
    will override the configuration in this ExperimentConfig.
    The step number passed to CheckpointManager.save() is a number of environment steps
    during training cycles. It is only called after an eval cycle.
    Example: if steps_per_cycle=3, num_train_cycles=10 and num_eval_cycles=2, save(5*3) and save(10*3) will be called.
    If num_eval_cycles is 0, CheckpointManager.save(num_train_cycles*steps_per_cycle) will be called after training.
    """
    devices: int | list[int] | AutoDeviceSelector = 0
    """Which devices to use for data parallel training.

    int means the first <n> local devices.

    list[int] means the local devices with the given indices.

    "all" means all of jax.local_devices().
    """

    # explititly define this so we can change the type of env and
    # avoid every subclass of ExperimentConfig having to supress type
    # checking.
    def __init__(
        self,
        num_eval_cycles: int,
        num_train_cycles: int,
        num_envs: int,
        random_seed: int,
        steps_per_cycle: int,
        # diff type from what's in the dataclass field to reflect that
        # env is not required to be a subclass of EnvConfig.
        env: typing.Any,
        checkpoint: CheckpointConfig | None = None,
        devices: int | list[int] | AutoDeviceSelector = 0,
    ) -> None:
        self.num_eval_cycles = num_eval_cycles
        self.num_train_cycles = num_train_cycles
        self.num_envs = num_envs
        self.random_seed = random_seed
        self.steps_per_cycle = steps_per_cycle
        self.env = typing.cast(EnvConfig, env)
        self.checkpoint = checkpoint
        self.devices = devices
        super().__init__()
        self.__post_init__()

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
