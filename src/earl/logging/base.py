"""Base logger."""

import abc
import weakref
from typing import Any, Protocol

from research.earl.core import ConfigForLog, EnvStep, Metrics


class Closable(abc.ABC):
    def __init__(self):
        self._finalizer = weakref.finalize(self, self._close)

    def close(self) -> None:
        """Closes the logger.

        Sub-classes should override _close(). This exists to handle the finalizer.
        """
        # Calling the finalizer the first time calls self_close and marks the finalizer dead.
        # Subsequent calls to close do nothing.
        self._finalizer()

    @abc.abstractmethod
    def _close(self) -> None:
        """Closes the logger.

        Sub-classes should override this method.
        """


class MetricLogger(Closable):
    """A logger for metrics."""

    @abc.abstractmethod
    def write(self, metrics: Metrics) -> None:
        """Writes `metrics` to some destination."""


class MetricLoggerFactory(Protocol):
    def __call__(self, label: str) -> MetricLogger: ...


class NoOpMetricLogger(MetricLogger):
    """MetricLogger which does nothing."""

    def write(self, metrics: Metrics):
        pass

    def _close(self):
        pass


def no_op_metric_logger_factory(label: str) -> MetricLogger:
    return NoOpMetricLogger()


class ConfigLogger(Closable):
    """A logger for configs."""

    @abc.abstractmethod
    def write(self, config: ConfigForLog) -> None:
        """Writes `config` to some destination."""


class NoOpConfigLogger(ConfigLogger):
    """ConfigLogger which does nothing."""

    def write(self, config: ConfigForLog):
        pass

    def _close(self):
        pass


class ConfigLoggerFactory(Protocol):
    def __call__(self) -> ConfigLogger: ...


class ObserveTrajectory(Protocol):
    def __call__(self, env_steps: EnvStep, step_infos: dict[Any, Any], step_num: int) -> Metrics: ...

    """Args:
        env_steps: a trajectory of env timesteps where the shape of each field is
            (num_envs, num_steps, *).
        step_infos: the aggregated info returned from environment.step().
        step_num: number of steps taken prior to the trajectory.
    """
