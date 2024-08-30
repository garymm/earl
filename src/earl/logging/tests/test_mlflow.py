import functools
import threading
import unittest
from unittest.mock import Mock, call, patch

import mlflow
import pytest
from mlflow.entities import Metric, Param

from research.earl.logging.metric_key import MetricKey
from research.earl.logging.mlflow import (
    MlflowConfigLogger,
    MlflowMetricLogger,
    _reset_run_id,
    _set_run_id,
    make_config_logger_factory,
    make_metric_logger_factory,
)


class TestMlflowMetricLogger(unittest.TestCase):
    def setUp(self):
        self.mock_client = Mock(spec=mlflow.MlflowClient)
        self.run_id = "test_run_id"

    def test_write(self):
        label = "label"
        logger = MlflowMetricLogger(self.mock_client, self.run_id, label=label)
        self.addCleanup(logger.close)
        fake_time_s = 1000.1
        with pytest.raises(KeyError, match=MetricKey.STEP_NUM):
            logger.write({"x": 0.95})

        with patch("time.time", return_value=fake_time_s):
            logger.write({MetricKey.STEP_NUM: 0, "x": 0.95})

        with patch("time.time", return_value=fake_time_s):
            logger.write({MetricKey.STEP_NUM: 1, "y": 0.95})

        call_0 = call(self.run_id, metrics=[Metric(f"{label}/x", 0.95, int(fake_time_s * 1000), 0)], synchronous=False)
        # step num should be automatically incremented
        call_1 = call(self.run_id, metrics=[Metric(f"{label}/y", 0.95, int(fake_time_s * 1000), 1)], synchronous=False)
        self.mock_client.log_batch.assert_has_calls((call_0, call_1))

    def test_close(self):
        logger = MlflowMetricLogger(self.mock_client, self.run_id)
        with patch("mlflow.flush_async_logging") as mock_flush:
            logger.close()
            mock_flush.assert_called_once()

        self.mock_client.set_terminated.assert_called_once_with(self.run_id)

    @patch.object(mlflow, "MlflowClient")
    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.set_experiment")
    @patch("mlflow.start_run")
    def test_make_metric_logger_factory(self, mock_start_run, mock_set_experiment, mock_set_tracking_uri, mock_client):
        _reset_run_id()
        exp_name = "test_experiment"
        tracking_uri = "http://test-uri"
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value = mock_run

        factory = make_metric_logger_factory(exp_name, tracking_uri)

        mock_set_tracking_uri.assert_called_once_with(tracking_uri)
        mock_set_experiment.assert_called_once_with(exp_name)
        mock_start_run.assert_called_once()

        logger = factory("test_label")
        logger.close()
        assert isinstance(logger, MlflowMetricLogger)
        assert logger._prefix == "test_label/"

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.set_experiment")
    @patch("mlflow.start_run")
    def test_set_run_id_thread_safety(self, mock_start_run, mock_set_experiment, mock_set_tracking_uri):
        _reset_run_id()
        exp_name = "test_experiment"
        tracking_uri = "http://test-uri"
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value = mock_run

        threads = [threading.Thread(target=functools.partial(_set_run_id, exp_name, tracking_uri)) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert mock_start_run.call_count == 1


class TestMlflowConfigLogger(unittest.TestCase):
    def setUp(self):
        self.mock_client = Mock(spec=mlflow.MlflowClient)
        self.run_id = "test_run_id"

    def test_write(self):
        logger = MlflowConfigLogger(self.mock_client, self.run_id)
        self.addCleanup(logger.close)

        logger.write({"a": 1, "b": "Bee"})

        self.mock_client.log_batch.assert_called_once_with(
            self.run_id, params=[Param("a", "1"), Param("b", "Bee")], synchronous=False
        )

    def test_close(self):
        logger = MlflowConfigLogger(self.mock_client, self.run_id)
        with patch("mlflow.flush_async_logging") as mock_flush:
            logger.close()
            mock_flush.assert_called_once()

    @patch.object(mlflow, "MlflowClient")
    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.set_experiment")
    @patch("mlflow.start_run")
    def test_make_config_logger_factory(self, mock_start_run, mock_set_experiment, mock_set_tracking_uri, mock_client):
        _reset_run_id()
        exp_name = "test_experiment"
        tracking_uri = "http://test-uri"
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value = mock_run

        factory = make_config_logger_factory(exp_name, tracking_uri)

        mock_set_tracking_uri.assert_called_once_with(tracking_uri)
        mock_set_experiment.assert_called_once_with(exp_name)
        mock_start_run.assert_called_once()

        logger = factory()
        logger.close()
        assert isinstance(logger, MlflowConfigLogger)
