import functools
from collections.abc import Callable
from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

from earl.core import EnvStep, Metrics, Video

Array = jax.Array


# BEGIN copied from rlax
# Below copied from rlax because its package is unusable with Bazel
# due to https://github.com/google-deepmind/rlax/issues/133.
def n_step_bootstrapped_returns(
  r_t: Array,
  discount_t: Array,
  v_t: Array,
  n: int,
  lambda_t: Array | float = 1.0,
  stop_target_gradients: bool = False,
) -> Array:
  """Computes strided n-step bootstrapped return targets over a sequence.

  The returns are computed according to the below equation iterated `n` times:

     Gₜ = rₜ₊₁ + γₜ₊₁ [(1 - λₜ₊₁) vₜ₊₁ + λₜ₊₁ Gₜ₊₁].

  When lambda_t == 1. (default), this reduces to

     Gₜ = rₜ₊₁ + γₜ₊₁ * (rₜ₊₂ + γₜ₊₂ * (... * (rₜ₊ₙ + γₜ₊ₙ * vₜ₊ₙ ))).

  Args:
    r_t: rewards at times [1, ..., T].
    discount_t: discounts at times [1, ..., T].
    v_t: state or state-action values to bootstrap from at time [1, ...., T].
    n: number of steps over which to accumulate reward before bootstrapping.
    lambda_t: lambdas at times [1, ..., T]. Shape is [], or [T-1].
    stop_target_gradients: bool indicating whether or not to apply stop gradient
      to targets.

  Returns:
    estimated bootstrapped returns at times [0, ...., T-1]
  """
  chex.assert_rank([r_t, discount_t, v_t, lambda_t], [1, 1, 1, {0, 1}])
  chex.assert_type([r_t, discount_t, v_t, lambda_t], float)
  chex.assert_equal_shape([r_t, discount_t, v_t])
  seq_len = r_t.shape[0]

  # Maybe change scalar lambda to an array.
  lambda_t = jnp.ones_like(discount_t) * lambda_t

  # Shift bootstrap values by n and pad end of sequence with last value v_t[-1].
  pad_size = min(n - 1, seq_len)
  targets = jnp.concatenate([v_t[n - 1 :], jnp.array([v_t[-1]] * pad_size)])

  # Pad sequences. Shape is now (T + n - 1,).
  r_t = jnp.concatenate([r_t, jnp.zeros(n - 1)])
  discount_t = jnp.concatenate([discount_t, jnp.ones(n - 1)])
  lambda_t = jnp.concatenate([lambda_t, jnp.ones(n - 1)])
  v_t = jnp.concatenate([v_t, jnp.array([v_t[-1]] * (n - 1))])

  # Work backwards to compute n-step returns.
  for i in reversed(range(n)):
    r_ = r_t[i : i + seq_len]
    discount_ = discount_t[i : i + seq_len]
    lambda_ = lambda_t[i : i + seq_len]
    v_ = v_t[i : i + seq_len]
    targets = r_ + discount_ * ((1.0 - lambda_) * v_ + lambda_ * targets)

  return jax.lax.select(stop_target_gradients, jax.lax.stop_gradient(targets), targets)


def transform_values(build_targets, *value_argnums):
  """Decorator to convert targets to use transformed value function."""

  @functools.wraps(build_targets)
  def wrapped_build_targets(tx_pair, *args, **kwargs):
    tx_args = list(args)
    for index in value_argnums:
      tx_args[index] = tx_pair.apply_inv(tx_args[index])

    targets = build_targets(*tx_args, **kwargs)
    return tx_pair.apply(targets)

  return wrapped_build_targets


transformed_n_step_returns = transform_values(n_step_bootstrapped_returns, 2)


class TxPair(NamedTuple):
  apply: Callable
  apply_inv: Callable


def signed_hyperbolic(x: Array, eps: float = 1e-3) -> Array:
  """Signed hyperbolic transform, inverse of signed_parabolic."""
  chex.assert_type(x, float)
  return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + eps * x


def signed_parabolic(x: Array, eps: float = 1e-3) -> Array:
  """Signed parabolic transform, inverse of signed_hyperbolic."""
  chex.assert_type(x, float)
  z = jnp.sqrt(1 + 4 * eps * (eps + 1 + jnp.abs(x))) / 2 / eps - 1 / 2 / eps
  return jnp.sign(x) * (jnp.square(z) - 1)


IDENTITY_PAIR = TxPair(lambda x: x, lambda x: x)


def batched_index(values: Array, indices: Array, keepdims: bool = False) -> Array:
  """Index into the last dimension of a tensor, preserving all others dims.

  Args:
    values: a tensor of shape [..., D],
    indices: indices of shape [...].
    keepdims: whether to keep the final dimension.

  Returns:
    a tensor of shape [...] or [..., 1].
  """
  indexed = jnp.take_along_axis(values, indices[..., None], axis=-1)
  if not keepdims:
    indexed = jnp.squeeze(indexed, axis=-1)
  return indexed


def transformed_n_step_q_learning(
  q_tm1: Array,
  a_tm1: Array,
  target_q_t: Array,
  a_t: Array,
  r_t: Array,
  discount_t: Array,
  n: int,
  stop_target_gradients: bool = True,
  tx_pair: TxPair = IDENTITY_PAIR,
) -> Array:
  """Calculates transformed n-step TD errors.

  See "Recurrent Experience Replay in Distributed Reinforcement Learning" by
  Kapturowski et al. (https://openreview.net/pdf?id=r1lyTjAqYX).

  Args:
    q_tm1: Q-values at times [0, ..., T - 1].
    a_tm1: action index at times [0, ..., T - 1].
    target_q_t: target Q-values at time [1, ..., T].
    a_t: action index at times [[1, ... , T]] used to select target q-values to
      bootstrap from; max(target_q_t) for normal Q-learning, max(q_t) for double
      Q-learning.
    r_t: reward at times [1, ..., T].
    discount_t: discount at times [1, ..., T].
    n: number of steps over which to accumulate reward before bootstrapping.
    stop_target_gradients: bool indicating whether or not to apply stop gradient
      to targets.
    tx_pair: TxPair of value function transformation and its inverse.

  Returns:
    Transformed N-step TD error.
  """
  chex.assert_rank([q_tm1, target_q_t, a_tm1, a_t, r_t, discount_t], [2, 2, 1, 1, 1, 1])
  chex.assert_type(
    [q_tm1, target_q_t, a_tm1, a_t, r_t, discount_t], [float, float, int, int, float, float]
  )

  v_t = batched_index(target_q_t, a_t)
  target_tm1 = transformed_n_step_returns(
    tx_pair, r_t, discount_t, v_t, n, stop_target_gradients=stop_target_gradients
  )
  q_a_tm1 = batched_index(q_tm1, a_tm1)
  return target_tm1 - q_a_tm1


# END copied from rlax


def update_buffer_batch(buffer, pointer, data, debug=False):
  """
  Update buffer with data for all environments at once.

  Args:
    buffer: Array of shape (num_envs, buffer_capacity, ...)
    pointer: Scalar uint32 index where to insert data (same for all environments)
    data: Array of shape (num_envs, seq_length, ...)
    debug: Whether to print debug information

  Returns:
    Updated buffer of shape (num_envs, buffer_capacity, ...)
  """
  if debug:
    jax.debug.print(
      "update_buffer_batch: buffer.shape: {}, data.shape: {}", buffer.shape, data.shape
    )

  # Start indices for the update:
  # - First dimension: start at the first environment (index 0)
  # - Second dimension: start at the pointer
  # - Additional dimensions: start at index 0
  start_indices = (jnp.array(0, dtype=jnp.uint32), pointer) + tuple(
    jnp.array(0, dtype=jnp.uint32) for _ in range(len(buffer.shape) - 2)
  )

  if debug:
    jax.debug.print("start_indices: {}", start_indices)

  return jax.lax.dynamic_update_slice(buffer, data, start_indices)


def render_minatar(obs: Array) -> np.ndarray:
  n_channels = obs.shape[-1]
  numerical_state = np.amax(obs * np.reshape(np.arange(n_channels) + 1, (1, 1, -1)), 2) + 0.5

  # Create a simple color map (similar to cubehelix)
  # Add black as the first color (for value 0)
  colors = np.zeros((n_channels + 1, 3))

  # Generate colors for each channel (1 to n_channels)
  for i in range(1, n_channels + 1):
    # Create colors with increasing intensity and some variation
    # This is a simplified version of cubehelix - adjust as needed
    hue = (i / n_channels) * 0.8 + 0.1  # Hue varies from 0.1 to 0.9
    saturation = 0.7
    value = 0.5 + i / (2 * n_channels)  # Value increases with channel index

    # Simple HSV to RGB conversion
    h = hue * 6
    c = value * saturation
    x = c * (1 - abs(h % 2 - 1))
    m = value - c

    if h < 1:
      r, g, b = c, x, 0
    elif h < 2:
      r, g, b = x, c, 0
    elif h < 3:
      r, g, b = 0, c, x
    elif h < 4:
      r, g, b = 0, x, c
    elif h < 5:
      r, g, b = x, 0, c
    else:
      r, g, b = c, 0, x

    colors[i] = np.array([r + m, g + m, b + m])

  # Vectorized mapping of numerical_state to RGB colors
  # Convert numerical_state to integer indices and clip to valid range
  indices = np.clip(numerical_state.astype(np.int32), 0, n_channels).reshape(-1)

  # Use the indices to look up colors
  rgb_values = colors[indices]

  # Reshape back to image dimensions with RGB channels
  rgb_image = rgb_values.reshape(numerical_state.shape + (3,))

  # Convert from float (0-1) to uint8 (0-255) for PIL compatibility
  rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)

  # Resize to 64x64 using nearest neighbor interpolation
  height, width = rgb_image_uint8.shape[:2]
  scale_h = 64 // height
  scale_w = 64 // width

  # Use numpy's repeat for simple nearest-neighbor upscaling
  # First, repeat rows
  upscaled = np.repeat(rgb_image_uint8, scale_h, axis=0)
  # Then, repeat columns
  upscaled = np.repeat(upscaled, scale_w, axis=1)

  return upscaled


def render_minatar_cycle(trajectory: EnvStep, step_infos: dict[Any, Any]) -> Metrics:
  obs = trajectory.obs
  if len(obs.shape) != 5:
    raise ValueError(f"Expected trajectory.obs to have shape (B, T, H, W, C),got {obs.shape}")
  obs = obs[0]
  img_array = np.stack([render_minatar(obs[i]) for i in range(obs.shape[0])])

  return {"video": Video(img_array)}  # pyright: ignore


def render_atari_cycle(trajectory: EnvStep, step_infos: dict[Any, Any]) -> Metrics:
  obs = trajectory.obs
  if len(obs.shape) != 5:
    raise ValueError(
      f"Expected trajectory.obs to have shape (B, T, stack_size, H, W),got {obs.shape}"
    )
  obs = obs[0, :, 0, :, :]  # batch index 0, stack index 0
  obs = np.expand_dims(obs, axis=3)  # add channel dimension

  return {"video": Video(obs)}  # pyright: ignore
