import os

# this doesn't work with pytest discovery because test discovery may import
# jax before we set the XLA_FLAGS environment variable.
# to run this test, use bazel or pytest this file directly.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax
import jax.numpy as jnp
import pytest

from earl.utils.sharding import pytree_get_index_0, shard_along_axis_0


def test_shard_along_axis_0_correct_shape_and_contents():
  devices = jax.local_devices(backend="cpu")
  n_devices = len(devices)
  if n_devices < 2:
    pytest.skip("This test requires at least two devices.")
  num_rows = n_devices * 4  # arbitrary multiple of n_devices
  arr = jnp.arange(num_rows * 3).reshape((num_rows, 3))
  sharded = shard_along_axis_0(arr, devices)
  expected_shape = (n_devices, num_rows // n_devices, 3)
  assert sharded.shape == expected_shape, (
    f"Expected shape {expected_shape}, but got {sharded.shape}"
  )

  assert isinstance(sharded.sharding, jax.sharding.PmapSharding)
  assert list(sharded.sharding.devices) == devices
  reconstructed = jnp.reshape(sharded, (num_rows, 3))
  assert jnp.array_equal(reconstructed, arr), "Reconstructed array does not match the original."


def test_shard_along_axis_0_not_divisible_raises_error():
  devices = jax.local_devices(backend="cpu")
  n_devices = len(devices)
  if n_devices < 2:
    pytest.skip("This test requires at least two devices.")
  # Create an array whose first dimension is not divisible by n_devices.
  arr = jnp.arange((n_devices * 4) + 1)  # intentionally off by one
  with pytest.raises(ValueError):
    shard_along_axis_0(arr, devices)


def test_pytree_get_index_0():
  pytree = {"a": jnp.arange(4), "b": {"c": jnp.arange(4)}, "scalar": jnp.array(1)}
  assert pytree_get_index_0(pytree) == {"a": 0, "b": {"c": 0}, "scalar": jnp.array(1)}
