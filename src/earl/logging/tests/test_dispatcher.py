import unittest
from collections.abc import Sequence
from unittest.mock import Mock, call

from research.earl.logging import base
from research.earl.logging.dispatcher import Dispatcher


class TestDispatcher(unittest.TestCase):
    def setUp(self):
        self.mock_logger1 = Mock(spec=base.MetricLogger)
        self.mock_logger2 = Mock(spec=base.MetricLogger)
        loggers: Sequence[base.MetricLogger] = [self.mock_logger1, self.mock_logger2]
        self.dispatcher = Dispatcher(loggers)

    def test_write(self):
        metrics = {"x": 0.95}
        self.dispatcher.write(metrics)

        self.mock_logger1.write.assert_called_once_with(metrics)
        self.mock_logger2.write.assert_called_once_with(metrics)

    def test_close(self):
        self.dispatcher.close()

        self.mock_logger1.close.assert_called_once()
        self.mock_logger2.close.assert_called_once()

    def test_write_multiple_calls(self):
        metrics1 = {"x": 0.95}
        metrics2 = {"x": 0.975}

        self.dispatcher.write(metrics1)
        self.dispatcher.write(metrics2)

        assert self.mock_logger1.write.call_count == 2
        assert self.mock_logger2.write.call_count == 2
        self.mock_logger1.write.assert_has_calls([call(metrics1), call(metrics2)])
        self.mock_logger2.write.assert_has_calls([call(metrics1), call(metrics2)])

    def test_empty_logger_sequence(self):
        empty_dispatcher = Dispatcher([])
        metrics = {"x": 0.95}

        # Should not raise any exceptions
        empty_dispatcher.write(metrics)
        empty_dispatcher.close()
