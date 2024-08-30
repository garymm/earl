import unittest.mock

import jax.numpy as jnp

from research.earl.logging.string_printer import StringPrinterMetricLogger, to_str


def test_to_str():
    values = {"a": 1, "b": 2.33333333, "c": jnp.array(3)}
    assert to_str(values) == "a = 1.000 | b = 2.333 | c = 3.000"


def test_write():
    written = []

    def append_written(s: str):
        written.append(s)

    label = "label"
    logger = StringPrinterMetricLogger(label=label, print_fn=append_written)
    for i in range(3):
        logger.write({"a": i, "b": i + 0.5})
    assert written == [
        "label/a = 0.000 | b = 0.500",
        "label/a = 1.000 | b = 1.500",
        "label/a = 2.000 | b = 2.500",
    ]
    logger.close()


def test_time_delta():
    written = []

    def append_written(s: str):
        written.append(s)

    label = "label"
    time_delta = 1
    logger = StringPrinterMetricLogger(label=label, print_fn=append_written, time_delta=time_delta)
    with unittest.mock.patch("time.monotonic") as mock_time:
        mock_time.return_value = time_delta + 0.01
        for i in range(3):
            logger.write({"a": i, "b": i + 0.5})
    assert written == [
        "label/a = 0.000 | b = 0.500",
    ]
