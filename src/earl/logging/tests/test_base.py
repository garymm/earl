from research.earl.logging.base import Closable, NoOpConfigLogger, no_op_metric_logger_factory


class ConcreteClosable(Closable):
    def __init__(self):
        super().__init__()
        self.closed = False

    def _close(self):
        self.closed = True


def test_close():
    closable = ConcreteClosable()
    closable.close()
    assert closable.closed


# can't really test the finalizer works to auto-close, since
# the time at which the garbage collector runs is non-deterministic


def test_no_op():
    metric_logger = no_op_metric_logger_factory("foo")
    metrics = {"test": 1}
    metric_logger.write(metrics)
    metric_logger.close()

    config_logger = NoOpConfigLogger()
    config = {"test": "test"}
    config_logger.write(config)
    config_logger.close()
