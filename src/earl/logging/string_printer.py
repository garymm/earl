import logging
import time
from collections.abc import Callable

from research.earl import core
from research.earl.logging import base


def to_str(metrics: core.Metrics) -> str:
    """Converts `metrics` to a pretty-printed string.

    Each [key, value] pair is separated by ' = ' and each entry is separated by ' | '.
    The keys are sorted alphabetically, and all values are formatted with 3 digits after the dot.

    For example:

        values = {"a": 1., "b" = 2.33333333}
        # Returns 'a = 1.000 | b = 2.333'
        values_string = to_str(values)

    Args:
      values: A dictionary with string keys.

    Returns:
      A formatted string.
    """
    return " | ".join(f"{k} = {v:0.3f}" for k, v in sorted(metrics.items()))


class StringPrinterMetricLogger(base.MetricLogger):
    """Converts metrics to a str and prints them."""

    def __init__(
        self,
        label: str = "",
        print_fn: Callable[[str], None] = logging.info,
        to_str_fn: Callable[[core.Metrics], str] = to_str,
        time_delta: float = 0.0,
    ):
        """Initializes the logger.

        Args:
          label: label string to use when logging.
          print_fn: function to call which acts like print.
          serialize_fn: function to call which transforms values into a str.
          time_delta: How often (in seconds) to write values. This can be used to
            minimize terminal spam, but is 0 by default---ie everything is written.
        """
        super().__init__()

        self._print_fn = print_fn
        self._to_str_fn = to_str_fn
        self._prefix = f"{label}/" if label else ""
        self._time = 0
        self._time_delta = time_delta

    def write(self, metrics: core.Metrics):
        should_print = True
        if self._time_delta > 0:
            now = time.monotonic()
            should_print = (now - self._time) > self._time_delta
            self._time = now
        if should_print:
            self._print_fn(f"{self._prefix}{self._to_str_fn(metrics)}")

    def _close(self):
        pass
