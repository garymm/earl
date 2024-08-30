from collections.abc import Sequence

from research.earl import core
from research.earl.logging import base


class Dispatcher(base.MetricLogger):
    """Writes data to multiple `MetricLogger` objects."""

    def __init__(self, to: Sequence[base.MetricLogger]):
        """Initialize the logger."""
        super().__init__()

        self._to = to

    def write(self, metrics: core.Metrics):
        """Writes `values` to the underlying `Logger` objects."""
        for logger in self._to:
            logger.write(metrics)

    def _close(self):
        for logger in self._to:
            logger.close()
