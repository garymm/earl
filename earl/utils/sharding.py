import typing
from collections.abc import Sequence

import jax
import jax.numpy as jnp


def shard_along_axis_0(x: jax.Array, devices: Sequence[jax.Device]) -> jax.Array:
  """Shards an array along axis 0.

  Args:
    x: The array to shard. x.shape[0] must be divisible by len(devices).
    devices: The devices to shard the array along.

  Returns:
    An array with shape (len(devices), x.shape[0] // len(devices), *x.shape[1:]).
  """
  n_devices = len(devices)
  if x.shape[0] % n_devices:
    raise ValueError(
      f"arr.shape[0] must be divisible by number of devices: {x.shape[0]} % {n_devices} != 0"
    )
  return jax.device_put_sharded(jnp.split(x, n_devices, axis=0), devices)


def pytree_get_index_0(pytree: typing.Any) -> typing.Any:
  """Gets the 0th index of a pytree.

  For each leaf in the pytree, if it is an array, return the 0th index.
  Otherwise, return the leaf unchanged.

  Args:
    pytree: The pytree to shard.
  """
  return jax.tree.map(lambda x: x[0] if isinstance(x, jax.Array) else x, pytree)
