import functools
from collections.abc import Callable
from typing import Any, NamedTuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

Array = jax.Array


# BEGIN copied from rlax
# Below copied from rlax because its package is unusable with Bazel
# due to https://github.com/google-deepmind/rlax/issues/133.
def double_q_learning(
  q_tm1: Array,
  a_tm1: Array,
  r_t: Array,
  discount_t: Array,
  q_t_value: Array,
  q_t_selector: Array,
  stop_target_gradients: bool = True,
) -> Array:
  """Calculates the double Q-learning temporal difference error.

  See "Double Q-learning" by van Hasselt.
  (https://papers.nips.cc/paper/3964-double-q-learning.pdf).

  Args:
    q_tm1: Q-values at time t-1.
    a_tm1: action index at time t-1.
    r_t: reward at time t.
    discount_t: discount at time t.
    q_t_value: Q-values at time t.
    q_t_selector: selector Q-values at time t.
    stop_target_gradients: bool indicating whether or not to apply stop gradient
      to targets.

  Returns:
    Double Q-learning temporal difference error.
  """
  chex.assert_rank([q_tm1, a_tm1, r_t, discount_t, q_t_value, q_t_selector], [1, 0, 0, 0, 1, 1])
  chex.assert_type(
    [q_tm1, a_tm1, r_t, discount_t, q_t_value, q_t_selector],
    [float, int, float, float, float, float],
  )

  target_tm1 = r_t + discount_t * q_t_value[q_t_selector.argmax()]
  target_tm1 = jax.lax.select(stop_target_gradients, jax.lax.stop_gradient(target_tm1), target_tm1)
  return target_tm1 - q_tm1[a_tm1]


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


def filter_incremental_update(new_tensors: Any, old_tensors: Any, step_size: float) -> Any:
  """Wrapper on top of optax.incremental_update that supports pytrees with non-array leaves."""
  new_tensors, _ = eqx.partition(new_tensors, eqx.is_array)
  old_tensors, static = eqx.partition(old_tensors, eqx.is_array)

  updated = optax.incremental_update(new_tensors, old_tensors, step_size)
  return eqx.combine(updated, static)


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
