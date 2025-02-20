from collections.abc import Sequence
from typing import Any

import chex
import equinox as eqx
import jax
import optax


# Below copied from rlax because its package is unusable with Bazel
# due to https://github.com/google-deepmind/rlax/issues/133.
def double_q_learning(
  q_tm1: jax.Array,
  a_tm1: jax.Array,
  r_t: jax.Array,
  discount_t: jax.Array,
  q_t_value: jax.Array,
  q_t_selector: jax.Array,
  stop_target_gradients: bool = True,
) -> jax.Array:
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


def filter_incremental_update(new_tensors: Any, old_tensors: Any, step_size: float) -> Any:
  """Wrapper on top of optax.incremental_update that supports pytrees with non-array leaves."""
  new_tensors, _ = eqx.partition(new_tensors, eqx.is_array)
  old_tensors, static = eqx.partition(old_tensors, eqx.is_array)

  updated = optax.incremental_update(new_tensors, old_tensors, step_size)
  return eqx.combine(updated, static)
