from collections.abc import Generator

from jax.numpy import ndarray
from jax.random import split as split_key


def keygen(start_key: ndarray) -> Generator[ndarray, None, None]:
  key = start_key
  while True:
    key, subkey = split_key(key)
    yield subkey
