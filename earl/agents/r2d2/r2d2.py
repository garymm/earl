import dataclasses
import functools
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import gymnax.environments.spaces as gymnax_spaces
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PRNGKeyArray, PyTree, Scalar

from earl.agents.r2d2.networks import R2D2Networks
from earl.agents.r2d2.utils import (
  TxPair,
  filter_incremental_update,
  signed_hyperbolic,
  signed_parabolic,
  transformed_n_step_q_learning,
  update_buffer_batch,
)
from earl.core import ActionAndState, Agent, AgentState, EnvStep
from earl.utils.eqx_filter import filter_cond
from earl.utils.sharding import shard_along_axis_0


# ------------------------------------------------------------------
class R2D2OptState(eqx.Module):
  optax_state: optax.OptState
  target_update_count: jax.Array  # scalar; counts steps since last target network update
  key: PRNGKeyArray


# R2D2 Actor state holds the LSTMCell's recurrent state and a PRNG key.
class R2D2ActorState(eqx.Module):
  lstm_h_c: Any  # Tuple of (h, c); see Section 2.3.
  key: PRNGKeyArray


# R2D2 Experience state holds a replay buffer of fixed-length sequences.
# We store one hidden state per sequence.
class R2D2ExperienceState(eqx.Module):
  observation: jax.Array  # shape: (num_sequences, seq_length, *obs_shape)
  action: jax.Array  # shape: (num_sequences, seq_length)
  reward: jax.Array  # shape: (num_sequences, seq_length)
  new_episode: jax.Array  # shape: (num_sequences, seq_length); True indicates episode end
  pointer: jax.Array  # scalar int, current insertion index in the sequence buffer
  size: jax.Array  # scalar int, current number of stored sequences
  hidden_state_h: jax.Array  # shape: (num_sequences, hidden_size) if store_hidden_states is true
  hidden_state_c: jax.Array  # shape: (num_sequences, hidden_size) if store_hidden_states is true


# Configuration for R2D2 agent.
@dataclasses.dataclass(eq=True, frozen=True)
class R2D2Config:
  discount: float = 0.997  # Section 2.3: discount factor.
  q_learning_n_steps: int = 5  # Section 2.3: n-step return.
  burn_in: int = 40  # Section 3: burn-in length for LSTMCell.
  priority_exponent: float = 0.9  # Section 2.3: priority exponent.
  target_update_period: int = 2500  # Section 2.3: period to update target network.
  buffer_capacity: int = 10000  # Total number of time steps in replay.
  replay_seq_length: int = 80  # sequence length (m).
  store_hidden_states: bool = False  # If true, store hidden state for first time step per sequence.
  # Gradient clipping suggested by https://www.nature.com/articles/nature14236
  # TODO: learning rate schedule
  optimizer: optax.GradientTransformation = optax.chain(optax.clip(1.0), optax.adam(1e-4))
  num_optims_per_target_update: int = 1  # how frequently to update the target network.
  target_update_step_size: float = 0.01  # how much to update the target network.
  num_envs_per_learner: int = 0
  """Number of environments per learner. 0 means use env_info.num_envs."""
  value_rescaling_epsilon: float = 1e-3  # Epsilon parameter for h and h⁻¹.

  def __post_init__(self):
    if self.buffer_capacity % self.replay_seq_length:
      raise ValueError("buffer_capacity must be divisible by replay_seq_length.")
    if self.burn_in > self.replay_seq_length:
      raise ValueError("burn_in must be less than or equal to replay_seq_length.")


# Update the AgentState alias to use bundled networks.
R2D2AgentState = AgentState[R2D2Networks, R2D2OptState, R2D2ExperienceState, R2D2ActorState]


class R2D2(Agent[R2D2Networks, R2D2OptState, R2D2ExperienceState, R2D2ActorState]):
  """An implementation of R2D2, or Recurrent Replay Distributed DQN.

  As described in "Recurrent Experience Replay in Distributed Reinforcement Learning"
  https://openreview.net/forum?id=r1lyTjAqYX
  """

  config: R2D2Config

  def _new_actor_state(self, nets: R2D2Networks, key: PRNGKeyArray) -> R2D2ActorState:
    batch_size = self.env_info.num_envs
    init_hidden = self._init_lstm_hidden(batch_size, nets.online.lstm_cell)
    return R2D2ActorState(lstm_h_c=init_hidden, key=key)

  def _init_lstm_hidden(
    self, batch_size: int, lstm_cell: eqx.nn.LSTMCell
  ) -> tuple[jax.Array, jax.Array]:
    h = jnp.zeros((batch_size, lstm_cell.hidden_size))
    c = jnp.zeros((batch_size, lstm_cell.hidden_size))
    return (h, c)

  def _new_experience_state(self, nets: R2D2Networks, key: PRNGKeyArray) -> R2D2ExperienceState:
    buffer_capacity = self.config.buffer_capacity
    sequence_capacity = buffer_capacity // self.config.replay_seq_length
    num_envs = self.config.num_envs_per_learner or self.env_info.num_envs
    env_info = self.env_info
    assert isinstance(env_info.observation_space, gymnax_spaces.Box)
    obs_shape = env_info.observation_space.shape
    observations = jnp.zeros(
      (num_envs, buffer_capacity) + obs_shape, dtype=env_info.observation_space.dtype
    )
    action_space = env_info.action_space
    assert isinstance(action_space, gymnax_spaces.Discrete)
    action = jnp.zeros((num_envs, buffer_capacity), dtype=action_space.dtype)
    reward = jnp.zeros((num_envs, buffer_capacity))
    new_episode = jnp.zeros((num_envs, buffer_capacity), dtype=jnp.bool)
    pointer = jnp.zeros((), dtype=jnp.uint32)
    size = jnp.zeros((num_envs,), dtype=jnp.uint32)
    if self.config.store_hidden_states:
      hidden_state_h = jnp.zeros((num_envs, sequence_capacity, nets.online.lstm_cell.hidden_size))
      hidden_state_c = jnp.zeros((num_envs, sequence_capacity, nets.online.lstm_cell.hidden_size))
    else:
      hidden_state_h = jnp.zeros(())
      hidden_state_c = jnp.zeros(())
    return R2D2ExperienceState(
      observation=observations,
      action=action,
      reward=reward,
      new_episode=new_episode,
      pointer=pointer,
      size=size,
      hidden_state_h=hidden_state_h,
      hidden_state_c=hidden_state_c,
    )

  def _new_opt_state(self, nets: R2D2Networks, key: PRNGKeyArray) -> R2D2OptState:
    return R2D2OptState(
      optax_state=self.config.optimizer.init(eqx.filter(nets.online, eqx.is_array)),
      target_update_count=jnp.array(0, dtype=jnp.uint32),
      key=key,
    )

  def _actor_batch_size(self) -> int:
    return self.config.num_envs_per_learner or self.env_info.num_envs

  def _act(
    self, actor_state: R2D2ActorState, nets: R2D2Networks, env_step: EnvStep
  ) -> ActionAndState[R2D2ActorState]:
    q_values, new_h_c = eqx.filter_vmap(nets.online)(
      env_step.obs, env_step.prev_action, env_step.reward, actor_state.lstm_h_c
    )
    action = jnp.argmax(q_values, axis=-1)
    new_actor_state = R2D2ActorState(lstm_h_c=new_h_c, key=actor_state.key)
    return ActionAndState(action, new_actor_state)

  @functools.partial(jax.jit, static_argnums=(0, 3))
  def _slice_for_replay(self, data: jax.Array, start_idx: jax.Array, length: int) -> jax.Array:
    B = self._actor_batch_size()
    assert start_idx.shape == (B,)
    assert data.shape[0] == B

    # Create indices [B, length] for each sequence
    indices = start_idx[:, None] + jnp.arange(length)  # shape (B, length)

    # Expand indices to have the same number of dimensions as data.
    # data has shape (B, T, ...) and indices should broadcast to (B, length, 1, ..., 1)
    extra_dims = data.ndim - 2
    indices_expanded = indices.reshape(indices.shape + (1,) * extra_dims)

    # Use take_along_axis to extract values along the time axis.
    slices = jnp.take_along_axis(data, indices_expanded, axis=1)  # shape [B, length, ...]
    # Swap to time-major format [length, B, ...]
    return jnp.swapaxes(slices, 0, 1)

  def _sample_from_experience(
    self, nets: R2D2Networks, experience_state: R2D2ExperienceState, key: PRNGKeyArray
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    seq_length = self.config.replay_seq_length
    buffer_capacity_sequences = self.config.buffer_capacity // seq_length
    B = self._actor_batch_size()
    size_sequences = experience_state.size // seq_length
    # Avoid sampling indices that are beyond the end of the experience buffer.
    # Will only happen before the buffer is filled once.
    seq_idx = jnp.minimum(
      jax.random.randint(key, (B,), 0, buffer_capacity_sequences - 1), size_sequences - 1
    )
    obs_idx = seq_idx * seq_length
    obs_time = self._slice_for_replay(experience_state.observation, obs_idx, seq_length)
    action_time = self._slice_for_replay(experience_state.action, obs_idx, seq_length)
    reward_time = self._slice_for_replay(experience_state.reward, obs_idx, seq_length)
    dones_time = self._slice_for_replay(experience_state.new_episode, obs_idx, seq_length)
    if self.config.store_hidden_states:
      hidden_h_pre = experience_state.hidden_state_h[jnp.arange(B), seq_idx]
      hidden_c_pre = experience_state.hidden_state_c[jnp.arange(B), seq_idx]
    else:
      hidden_h_pre, hidden_c_pre = self._init_lstm_hidden(B, nets.online.lstm_cell)

    assert hidden_h_pre.shape == (B, nets.online.lstm_cell.hidden_size)
    assert hidden_c_pre.shape == (B, nets.online.lstm_cell.hidden_size)
    assert isinstance(self.env_info.observation_space, gymnax_spaces.Box)
    assert obs_time.shape == (seq_length, B, *self.env_info.observation_space.shape)
    assert action_time.shape == (seq_length, B)
    assert reward_time.shape == (seq_length, B)
    assert dones_time.shape == (seq_length, B)
    return obs_time, action_time, reward_time, dones_time, hidden_h_pre, hidden_c_pre

  def _loss(
    self, nets: R2D2Networks, opt_state: R2D2OptState, experience_state: R2D2ExperienceState
  ) -> tuple[Scalar, R2D2ExperienceState]:
    seq_length = self.config.replay_seq_length
    burn_in = self.config.burn_in
    q_learning_n_steps = self.config.q_learning_n_steps
    eps = self.config.value_rescaling_epsilon

    # Number of sequences (batch size).
    B = experience_state.observation.shape[0]
    assert self._actor_batch_size() == B

    # _time means time first, batch last.
    obs_time, action_time, reward_time, dones_time, hidden_h_pre, hidden_c_pre = (
      self._sample_from_experience(nets, experience_state, opt_state.key)
    )
    assert action_time.shape == (seq_length, B), action_time.shape
    online_hidden = hidden_h_pre, hidden_c_pre
    target_hidden = hidden_h_pre, hidden_c_pre

    def scan_fn(hidden, oar, network):
      obs, action, reward = oar
      q, new_hidden = jax.vmap(network)(obs, action, reward, hidden)
      return new_hidden, (q, new_hidden)

    scan_target_fn = functools.partial(scan_fn, network=nets.target)
    scan_online_fn = functools.partial(scan_fn, network=nets.online)

    if burn_in:
      burn_obs, burn_action, burn_reward = jax.tree.map(
        lambda x: x[:burn_in], (obs_time, action_time, reward_time)
      )
      online_hidden, _ = jax.lax.scan(
        scan_online_fn, online_hidden, (burn_obs, burn_action, burn_reward)
      )
      target_hidden, _ = jax.lax.scan(
        scan_target_fn, target_hidden, (burn_obs, burn_action, burn_reward)
      )
      obs_time, action_time, reward_time, dones_time = jax.tree.map(
        lambda x: x[burn_in:], (obs_time, action_time, reward_time, dones_time)
      )

    _, (online_q, _) = jax.lax.scan(
      scan_online_fn, online_hidden, (obs_time, action_time, reward_time)
    )
    _, (target_q, _) = jax.lax.scan(
      scan_target_fn, target_hidden, (obs_time, action_time, reward_time)
    )

    # Get value-selector actions from online Q-values for double Q-learning.
    selector_actions = jnp.argmax(online_q, axis=-1)
    # set discount to 0 for any time step past a done.
    mask_time = jnp.ones_like(dones_time, dtype=online_q.dtype) - jnp.cumsum(dones_time, axis=0)
    discount_time = (self.config.discount * mask_time).astype(online_q.dtype)

    tx_pair = TxPair(
      functools.partial(signed_hyperbolic, eps=eps), functools.partial(signed_parabolic, eps=eps)
    )

    batch_td_error_fn = jax.vmap(
      functools.partial(transformed_n_step_q_learning, n=q_learning_n_steps, tx_pair=tx_pair),
      in_axes=1,
      out_axes=1,
    )
    batch_td_error = batch_td_error_fn(
      online_q[:-1],
      action_time[:-1],
      target_q[1:],
      selector_actions[1:],
      reward_time[:-1],
      discount_time[:-1],
    )
    batch_loss = 0.5 * jnp.square(batch_td_error).sum(axis=0)
    # TODO: Add importance sampling to the experience state.
    mean_loss = jnp.mean(batch_loss)
    return mean_loss, experience_state

  def _update_experience(
    self,
    experience_state: R2D2ExperienceState,
    actor_state_pre: R2D2ActorState,
    actor_state_post: R2D2ActorState,
    trajectory: EnvStep,
  ) -> R2D2ExperienceState:
    B = self._actor_batch_size()
    if trajectory.reward.shape[0] != B:
      raise ValueError("trajectory.reward.shape[0] must be equal to num_envs_per_learner.")
    if trajectory.reward.shape[1] != self.config.replay_seq_length:
      raise ValueError("trajectory.reward.shape[1] must be equal to replay_seq_length.")
    buffer_capacity = self.config.buffer_capacity
    seq_length = self.config.replay_seq_length

    new_obs = update_buffer_batch(
      experience_state.observation,
      experience_state.pointer,
      trajectory.obs,
    )

    new_actions = update_buffer_batch(
      experience_state.action,
      experience_state.pointer,
      trajectory.prev_action,
    )

    new_rewards = update_buffer_batch(
      experience_state.reward,
      experience_state.pointer,
      trajectory.reward,
    )

    new_dones = update_buffer_batch(
      experience_state.new_episode,
      experience_state.pointer,
      trajectory.new_episode,
    )

    def update_hidden(hidden_buffer, pointer, init_hidden):
      seq_index = pointer // seq_length
      return hidden_buffer.at[seq_index].set(init_hidden)

    if self.config.store_hidden_states:
      new_hidden_h = jax.vmap(update_hidden)(
        experience_state.hidden_state_h,
        experience_state.pointer,
        actor_state_pre.lstm_h_c[0],
      )
      new_hidden_c = jax.vmap(update_hidden)(
        experience_state.hidden_state_c,
        experience_state.pointer,
        actor_state_pre.lstm_h_c[1],
      )
    else:
      new_hidden_h = experience_state.hidden_state_h
      new_hidden_c = experience_state.hidden_state_c
    new_ptr = (experience_state.pointer + seq_length) % buffer_capacity
    new_size = jnp.minimum(experience_state.size + seq_length, buffer_capacity)
    return R2D2ExperienceState(
      observation=new_obs,
      action=new_actions,
      reward=new_rewards,
      new_episode=new_dones,
      pointer=new_ptr,
      size=new_size,
      hidden_state_h=new_hidden_h,
      hidden_state_c=new_hidden_c,
    )

  def num_off_policy_optims_per_cycle(self) -> int:
    return 1

  def update_target(self, nets: R2D2Networks) -> R2D2Networks:
    return R2D2Networks(online=nets.online, target=nets.online)

  def _partition_for_grad(self, nets: R2D2Networks) -> tuple[R2D2Networks, R2D2Networks]:
    filter_spec = jax.tree.map(lambda _: eqx.is_array, nets)
    filter_spec = eqx.tree_at(lambda nets: nets.target, filter_spec, replace=False)
    return eqx.partition(nets, filter_spec)

  def _prepare_for_actor_cycle(self, actor_state: R2D2ActorState) -> R2D2ActorState:
    return actor_state

  def _optimize_from_grads(
    self, nets: R2D2Networks, opt_state: R2D2OptState, nets_grads: PyTree
  ) -> tuple[R2D2Networks, R2D2OptState]:
    updates, optax_state = self.config.optimizer.update(nets_grads.online, opt_state.optax_state)
    new_online = eqx.apply_updates(nets.online, updates)

    def update_target():
      return filter_incremental_update(
        new_online, nets.target, self.config.target_update_step_size
      ), jnp.zeros_like(opt_state.target_update_count)

    def keep_target():
      return nets.target, opt_state.target_update_count + 1

    new_target, new_count = filter_cond(
      opt_state.target_update_count + 1 >= self.config.target_update_period,
      update_target,
      keep_target,
    )
    new_nets = R2D2Networks(online=new_online, target=new_target)
    new_opt_state = R2D2OptState(
      optax_state=optax_state,
      target_update_count=new_count,
      key=jax.random.split(opt_state.key, 2)[1],
    )
    return new_nets, new_opt_state

  def shard_actor_state(
    self, actor_state: R2D2ActorState, learner_devices: Sequence[jax.Device]
  ) -> R2D2ActorState:
    return R2D2ActorState(
      lstm_h_c=(
        shard_along_axis_0(actor_state.lstm_h_c[0], learner_devices),
        shard_along_axis_0(actor_state.lstm_h_c[1], learner_devices),
      ),
      key=jax.device_put_replicated(actor_state.key, learner_devices),
    )
