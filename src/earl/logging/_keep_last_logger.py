from typing import Optional

from research.earl.core import Metrics
from research.earl.logging.base import MetricLogger


class KeepLastLogger(MetricLogger):
    """Forwards all calls to the provided inner `MetricLogger` while also storing the last written metrics."""

    _inner: MetricLogger
    last_metrics: Optional[Metrics]

    def __init__(self, inner: MetricLogger):
        self._inner = inner
        self.last_metrics = None

    def _close(self) -> None:
        self._inner.close()

    def write(self, metrics: Metrics) -> None:
        self.last_metrics = metrics
        self._inner.write(metrics)
