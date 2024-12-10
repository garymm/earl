from collections.abc import Mapping, Sequence

import jax

from research.earl.core import Image, Metrics
from research.earl.logging.base import MetricLogger


class MemoryLogger(MetricLogger):
    """Stores metrics by appending them to a list."""

    def __init__(self):
        super().__init__()
        self._metrics_list: list[Metrics] = []

    @property
    def metrics_list(self) -> Sequence[Metrics]:
        return self._metrics_list

    def metrics(self) -> Mapping[str, Sequence[int | float | Image | jax.Array]]:
        """Transposes the list of metrics assuming each metrics object contains the same set of keys"""
        return {k: list(m[k] for m in self._metrics_list) for k in set(k for m in self._metrics_list for k in m)}

    def write(self, metrics: Metrics):
        self._metrics_list.append(metrics)

    def _close(self):
        pass
