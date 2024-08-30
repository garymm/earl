import threading
import time

import mlflow
import mlflow.entities

from research.earl import core
from research.earl.logging import base
from research.earl.logging.metric_key import MetricKey

_RUN_ID = None
_RUN_ID_LOCK = threading.Lock()
_FLUSH_LOCK = threading.Lock()


class MlflowMetricLogger(base.MetricLogger):
    """Logs metrics to Mlflow."""

    def __init__(
        self,
        client: mlflow.MlflowClient,
        run_id: str,
        label: str = "",
    ):
        """Initializes the logger.

        Args:
          label: label string to use when logging. Metrics will be prefixed with this + "/".
        """
        super().__init__()

        self._client = client
        self._run_id = run_id
        self._prefix = f"{label}/" if label else ""

    def write(self, metrics: core.Metrics):
        """Removes MetricKey.STEP_NUM from the metrics and uses it as the step number."""
        step_num = int(metrics[MetricKey.STEP_NUM])
        timestamp = int(time.time() * 1000)
        metrics_list = []
        for k, v in metrics.items():
            if k == MetricKey.STEP_NUM:
                continue
            k = f"{self._prefix}{k}"
            metrics_list.append(mlflow.entities.Metric(k, float(v), timestamp, step_num))

        self._client.log_batch(self._run_id, metrics=metrics_list, synchronous=False)

    def _close(self):
        self._client.set_terminated(self._run_id)
        # It seems flush_async_logging() isn't thread safe.
        with _FLUSH_LOCK:
            mlflow.flush_async_logging()


class MlflowConfigLogger(base.ConfigLogger):
    def __init__(
        self,
        client: mlflow.MlflowClient,
        run_id: str,
    ):
        """Initializes the logger.

        Args:
          label: label string to use when logging. Metrics will be prefixed with this + "/".
        """
        super().__init__()

        self._client = client
        self._run_id = run_id

    def write(self, config: core.ConfigForLog):
        """Logs the config to Mlflow."""
        params = [mlflow.entities.Param(key, str(value)) for key, value in config.items()]
        self._client.log_batch(self._run_id, params=params, synchronous=False)

    def _close(self):
        # It seems flush_async_logging() isn't thread safe.
        with _FLUSH_LOCK:
            mlflow.flush_async_logging()


def _set_run_id(exp_name: str, tracking_uri: str):
    global _RUN_ID, _RUN_ID_LOCK
    with _RUN_ID_LOCK:
        if _RUN_ID is None:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(exp_name)
            run = mlflow.start_run()
            _RUN_ID = run.info.run_id


def make_metric_logger_factory(exp_name: str, tracking_uri: str) -> base.MetricLoggerFactory:
    _set_run_id(exp_name, tracking_uri)

    def _factory(label: str):
        assert _RUN_ID
        return MlflowMetricLogger(mlflow.MlflowClient(), _RUN_ID, label=label)

    return _factory


def make_config_logger_factory(exp_name: str, tracking_uri: str) -> base.ConfigLoggerFactory:
    _set_run_id(exp_name, tracking_uri)

    def _factory():
        assert _RUN_ID
        return MlflowConfigLogger(mlflow.MlflowClient(), _RUN_ID)

    return _factory


def _reset_run_id():
    """Resets the run ID to None. Only for testing."""
    global _RUN_ID, _RUN_ID_LOCK
    with _RUN_ID_LOCK:
        _RUN_ID = None
