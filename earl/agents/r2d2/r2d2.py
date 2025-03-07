"""An implementation of R2D2, or Recurrent Replay Distributed DQN.

NOTE: when changing, please run pytest earl/agents/r2d2/test_r2d2_learns.py to ensure it learns.
That does not run in CI because it takes too long since we don't have a GPU in CI.
"""

import dataclasses
import enum
import functools
from collections.abc import Sequence
from typing import Any

import distrax
import equinox as eqx
import gymnax.environments.spaces as gymnax_spaces
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PRNGKeyArray, PyTree, Scalar

from earl.agents.r2d2.networks import R2D2Network, R2D2Networks
from earl.agents.r2d2.utils import (
  IDENTITY_PAIR,
  TxPair,
  signed_hyperbolic,
  signed_parabolic,
  transformed_n_step_q_learning,
  update_buffer_batch,
)
from earl.core import ActionAndState, Agent, AgentState, EnvStep
from earl.utils.sharding import shard_along_axis_0


class ExplorationType(enum.Enum):
  RANDOM = enum.auto()
  STICKY = enum.auto()


class R2D2OptState(eqx.Module):
  optax_state: optax.OptState
  optimize_count: jax.Array  # scalar; counts steps since last target network update
  key: PRNGKeyArray


# R2D2 Actor state holds the LSTMCell's recurrent state and a PRNG key.
class R2D2ActorState(eqx.Module):
  lstm_h_c: Any  # Tuple of (h, c); see Section 2.3.
  key: PRNGKeyArray
  num_steps: jax.Array = eqx.field(default_factory=lambda: jnp.array(0, dtype=jnp.uint32))


# R2D2 Experience state holds a replay buffer of fixed-length sequences.
# We store one hidden state per sequence.
class R2D2ExperienceState(eqx.Module):
  observation: jax.Array  # shape: (num_envs, num_sequences, seq_length, *obs_shape)
  action: jax.Array  # shape: (num_envs, num_sequences, seq_length)
  reward: jax.Array  # shape: (num_envs, num_sequences, seq_length)
  new_episode: jax.Array  # shape: (num_envs, num_sequences, seq_length); True indicates episode end
  pointer: jax.Array  # scalar int, current insertion index in the sequence buffer
  size: jax.Array  # scalar int, current number of stored sequences
  priority: jax.Array  # shape: (num_envs, num_sequences)
  hidden_state_h: jax.Array  # shape: (num_envs, num_sequences, hidden_size) if store_hidden_states
  hidden_state_c: jax.Array  # shape: (num_envs, num_sequences, hidden_size) if store_hidden_states


# Configuration for R2D2 agent.
@dataclasses.dataclass(eq=True, frozen=True)
class R2D2Config:
  epsilon_greedy_schedule_args: dict[str, Any] = dataclasses.field(default_factory=dict)
  debug: bool = False
  discount: float = 0.997  # Section 2.3: discount factor. γ
  q_learning_n_steps: int = 5  # Section 2.3: n-step return.
  burn_in: int = 40  # Section 3: burn-in length for LSTMCell.
  target_update_period: int = 2500  # Section 2.3: period to update target network.
  buffer_capacity: int = 10000  # Total number of time steps in replay.
  replay_seq_length: int = 80  # sequence length (m).
  store_hidden_states: bool = False  # If true, store hidden state for first time step per sequence.
  num_optims_per_target_update: int = 1  # how frequently to update the target network.
  target_update_step_size: float = 0.001  # how much to update the target network.
  num_envs_per_learner: int = 0
  """Number of environments per learner. 0 means use env_info.num_envs."""
  importance_sampling_priority_exponent: float = 0.9  # section 2.3
  max_priority_weight: float = 0.9  # section 2.3, η
  gradient_clipping_max_delta: float = 1.0
  learning_rate_schedule_name: str = "constant_schedule"
  learning_rate_schedule_args: dict[str, Any] = dataclasses.field(default_factory=dict)
  value_rescaling_epsilon: float = 1e-3  # Epsilon parameter for h and h⁻¹.
  num_off_policy_optims_per_cycle: int = 10  # Number of off-policy optims per cycle.
  exploration_type: ExplorationType = ExplorationType.RANDOM
  replay_batch_size: int = (
    0  # Number of sequences to sample per cycle. If 0, use num_envs_per_learner.
  )

  def __post_init__(self):
    if self.buffer_capacity % self.replay_seq_length:
      raise ValueError("buffer_capacity must be divisible by replay_seq_length.")
    if self.burn_in > self.replay_seq_length:
      raise ValueError("burn_in must be less than or equal to replay_seq_length.")
    if self.target_update_step_size <= 0 and not self.target_update_period:
      raise ValueError(
        "target_update_step_size must be greater than 0 or target_update_period must be set."
      )

  @property
  def epsilon_greedy_schedule(self) -> optax.Schedule:
    return optax.linear_schedule(**self.epsilon_greedy_schedule_args)

  @property
  def optimizer(self) -> optax.GradientTransformation:
    learning_rate_schedule = getattr(optax, self.learning_rate_schedule_name)(
      **self.learning_rate_schedule_args
    )
    return optax.chain(
      optax.clip(self.gradient_clipping_max_delta), optax.adam(learning_rate_schedule, eps=1e-3)
    )


# Update the AgentState alias to use bundled networks.
R2D2AgentState = AgentState[R2D2Networks, R2D2OptState, R2D2ExperienceState, R2D2ActorState]


class R2D2(Agent[R2D2Networks, R2D2OptState, R2D2ExperienceState, R2D2ActorState]):
  """An implementation of R2D2, or Recurrent Replay Distributed DQN.

  As described in "Recurrent Experience Replay in Distributed Reinforcement Learning"
  https://openreview.net/forum?id=r1lyTjAqYX
  """

  config: R2D2Config = eqx.field(static=True)

  def _new_actor_state(self, nets: R2D2Networks, key: PRNGKeyArray) -> R2D2ActorState:
    batch_size = self.env_info.num_envs
    init_hidden = self._init_lstm_hidden(batch_size, nets.online)
    return R2D2ActorState(lstm_h_c=init_hidden, key=key, num_steps=jnp.array(0, dtype=jnp.uint32))

  def _init_lstm_hidden(self, batch_size: int, network: R2D2Network) -> tuple[jax.Array, jax.Array]:
    if network.lstm_cell is None:
      key = jax.random.PRNGKey(0)
      sample_input = jnp.zeros((batch_size, *self.env_info.observation_space.sample(key).shape))
      init_hidden = eqx.filter_vmap(network.embed)(
        sample_input,
        jnp.zeros((batch_size,), dtype=self.env_info.action_space.sample(key).dtype),
        jnp.zeros((batch_size,), dtype=jnp.float32),
      )
      return init_hidden, init_hidden
    h = jnp.zeros((batch_size, network.lstm_cell.hidden_size))
    c = jnp.zeros((batch_size, network.lstm_cell.hidden_size))
    return h, c

  def _new_experience_state(self, nets: R2D2Networks, key: PRNGKeyArray) -> R2D2ExperienceState:
    if (
      self.config.replay_batch_size > 0
      and self.config.replay_batch_size % self._experience_state_batch_size()
    ):
      raise ValueError("replay_batch_size must be divisible by _experience_state_batch_size.")
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
    priority = jnp.zeros((num_envs, sequence_capacity), dtype=jnp.float32)
    if self.config.store_hidden_states:
      assert nets.online.lstm_cell is not None
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
      priority=priority,
      hidden_state_h=hidden_state_h,
      hidden_state_c=hidden_state_c,
    )

  def _new_opt_state(self, nets: R2D2Networks, key: PRNGKeyArray) -> R2D2OptState:
    return R2D2OptState(
      optax_state=self.config.optimizer.init(eqx.filter(nets.online, eqx.is_array)),
      optimize_count=jnp.array(0, dtype=jnp.uint32),
      key=key,
    )

  def _experience_state_batch_size(self) -> int:
    return self.config.num_envs_per_learner or self.env_info.num_envs

  def _replay_batch_size(self) -> int:
    return self.config.replay_batch_size or self._experience_state_batch_size()

  def _act(
    self, actor_state: R2D2ActorState, nets: R2D2Networks, env_step: EnvStep
  ) -> ActionAndState[R2D2ActorState]:
    q_values, new_h_c = eqx.filter_vmap(nets.online)(
      env_step.obs, env_step.prev_action, env_step.reward, actor_state.lstm_h_c
    )
    key = jax.random.split(actor_state.key, 2)[1]
    epsilon = self.config.epsilon_greedy_schedule(actor_state.num_steps)
    if self.config.exploration_type == ExplorationType.RANDOM:
      action = distrax.EpsilonGreedy(
        q_values,
        epsilon,  # pyright: ignore[reportArgumentType]
      ).sample(seed=key)
    else:
      assert self.config.exploration_type == ExplorationType.STICKY
      greedy_action = jnp.argmax(q_values, axis=-1)
      sticky_action = env_step.prev_action
      action = jnp.where(
        jax.random.uniform(key, shape=(env_step.reward.shape[0],)) < epsilon,
        greedy_action,
        sticky_action,
      )
    new_actor_state = R2D2ActorState(lstm_h_c=new_h_c, key=key, num_steps=actor_state.num_steps + 1)
    if self.config.debug:
      jax.debug.print("Q-values before action: {}", q_values)
      q_diff = q_values[:, 1] - q_values[:, 0]
      jax.debug.print(
        "Q-value difference (1-0): mean={}, min={}, max={}",
        jnp.mean(q_diff),
        jnp.min(q_diff),
        jnp.max(q_diff),
      )
      lstm_h, lstm_c = actor_state.lstm_h_c
      jax.debug.print(
        "LSTM h,c stats - h_mean: {}, h_std: {}, c_mean: {}, c_std: {}",
        jnp.mean(lstm_h),
        jnp.std(lstm_h),
        jnp.mean(lstm_c),
        jnp.std(lstm_c),
      )
    return ActionAndState(action, new_actor_state)

  @functools.partial(jax.jit, static_argnums=(0, 3))
  def _slice_for_replay(self, data: jax.Array, start_idx: jax.Array, length: int) -> jax.Array:
    B = self._experience_state_batch_size()
    assert start_idx.shape == (B,)
    assert data.shape[0] == B

    # Create indices [B, length] for each sequence
    indices = start_idx[:, None] + jnp.arange(length)  # shape (B, length)

    # Expand indices to have the same number of dimensions as data.
    # data has shape (B, T, ...) and indices should broadcast to (B, length, 1, ..., 1)
    extra_dims = data.ndim - 2
    indices_expanded = indices.reshape(indices.shape + (1,) * extra_dims)

    # Use take_along_axis to extract values along the time axis.
    return jnp.take_along_axis(data, indices_expanded, axis=1)  # shape [B, length, ...]

  def _sample_one_experience_batch(
    self, seq_idx: jax.Array, nets: R2D2Networks, experience_state: R2D2ExperienceState
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Samples _experience_state_batch_size() sequences from experience."""
    seq_length = self.config.replay_seq_length
    B = self._experience_state_batch_size()
    obs_idx = seq_idx * seq_length
    obs = self._slice_for_replay(experience_state.observation, obs_idx, seq_length)
    action = self._slice_for_replay(experience_state.action, obs_idx, seq_length)
    reward = self._slice_for_replay(experience_state.reward, obs_idx, seq_length)
    dones = self._slice_for_replay(experience_state.new_episode, obs_idx, seq_length)
    if self.config.store_hidden_states:
      hidden_h_pre = experience_state.hidden_state_h[jnp.arange(B), seq_idx]
      hidden_c_pre = experience_state.hidden_state_c[jnp.arange(B), seq_idx]
    else:
      hidden_h_pre, hidden_c_pre = self._init_lstm_hidden(B, nets.online)

    if nets.online.lstm_cell is not None:
      assert hidden_h_pre.shape == (B, nets.online.lstm_cell.hidden_size)
      assert hidden_c_pre.shape == (B, nets.online.lstm_cell.hidden_size)
    assert isinstance(self.env_info.observation_space, gymnax_spaces.Box)
    assert obs.shape == (B, seq_length, *self.env_info.observation_space.shape)
    assert action.shape == (B, seq_length)
    assert reward.shape == (B, seq_length)
    assert dones.shape == (B, seq_length)
    return obs, action, reward, dones, hidden_h_pre, hidden_c_pre

  def _sample_from_experience(
    self, nets: R2D2Networks, key: PRNGKeyArray, experience_state: R2D2ExperienceState
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    # Number of sequences (batch size).
    B = self._replay_batch_size()
    batch_size_multiplier = B // self._experience_state_batch_size()
    categorical_ax1 = functools.partial(jax.random.categorical, axis=1)
    random_idx = jax.vmap(categorical_ax1, in_axes=(0, None))(
      jax.random.split(key, batch_size_multiplier), experience_state.priority
    )
    assert random_idx.shape == (batch_size_multiplier, self.config.num_envs_per_learner)
    size_sequences = experience_state.size // self.config.replay_seq_length
    # Avoid sampling indices that are beyond the end of the experience buffer.
    # Will only happen before the buffer is filled once.
    sampled_seq_idx = jnp.minimum(random_idx, size_sequences - 1)
    obs, action, reward, dones, hidden_h_pre, hidden_c_pre = jax.vmap(
      self._sample_one_experience_batch, in_axes=(0, None, None)
    )(sampled_seq_idx, nets, experience_state)
    # squash the batch_size_multiplier dimension and the experience_state_batch_size dimension
    sampled_seq_idx, obs, action, reward, dones, hidden_h_pre, hidden_c_pre = [
      jnp.reshape(x, (B, *x.shape[2:]))
      for x in (sampled_seq_idx, obs, action, reward, dones, hidden_h_pre, hidden_c_pre)
    ]
    # swap to time first, then batch.
    obs_time, action_time, reward_time, dones_time = [
      jnp.swapaxes(x, 1, 0) for x in (obs, action, reward, dones)
    ]
    return (
      sampled_seq_idx,
      obs_time,
      action_time,
      reward_time,
      dones_time,
      hidden_h_pre,
      hidden_c_pre,
    )

  def _loss(
    self, nets: R2D2Networks, opt_state: R2D2OptState, experience_state: R2D2ExperienceState
  ) -> tuple[Scalar, R2D2ExperienceState]:
    seq_length = self.config.replay_seq_length
    burn_in = self.config.burn_in
    q_learning_n_steps = self.config.q_learning_n_steps
    eps = self.config.value_rescaling_epsilon

    sampled_seq_idx, obs_time, action_time, reward_time, dones_time, hidden_h_pre, hidden_c_pre = (
      self._sample_from_experience(nets, opt_state.key, experience_state)
    )
    # # Debug the sampled experience data
    # jax.debug.print("Sampled dones_time shape: {}", dones_time.shape)
    if self.config.debug:
      jax.debug.print("Sampled dones_time content: {}", dones_time)
    B = self._replay_batch_size()
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
      # Debug after burn-in
      # jax.debug.print("Dones after burn-in: {}", dones_time)
      # TODO: probably need to stop grad?

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

    # Debug prints for Q-values and masking
    if self.config.debug:
      jax.debug.print("Online Q-values mean: {}", jnp.mean(online_q))
      jax.debug.print("Online Q-values action 0 mean: {}", jnp.mean(online_q[..., 0]))
      jax.debug.print("Online Q-values action 1 mean: {}", jnp.mean(online_q[..., 1]))
      jax.debug.print("Target Q-values mean: {}", jnp.mean(target_q))
      jax.debug.print("Q-value difference: {}", jnp.mean(online_q[..., 1] - online_q[..., 0]))
      jax.debug.print("Q-values variance: {}", jnp.var(online_q))
      jax.debug.print("dones_time sum: {}", jnp.sum(dones_time))
      jax.debug.print("mask_time min: {}, max: {}", jnp.min(mask_time), jnp.max(mask_time))
      jax.debug.print(
        "discount_time min: {}, max: {}", jnp.min(discount_time), jnp.max(discount_time)
      )

    tx_pair = IDENTITY_PAIR
    if self.config.value_rescaling_epsilon > 0:
      tx_pair = TxPair(
        functools.partial(signed_hyperbolic, eps=eps), functools.partial(signed_parabolic, eps=eps)
      )

    batch_td_error_fn = jax.vmap(
      functools.partial(transformed_n_step_q_learning, n=q_learning_n_steps, tx_pair=tx_pair),
      in_axes=1,
      out_axes=1,
    )
    # the vmap and partial seem to have confused pyright here.
    batch_td_error = batch_td_error_fn(
      online_q[:-1],
      action_time[:-1],
      target_q[1:],
      selector_actions[1:],
      reward_time[:-1],  # pyright: ignore[reportCallIssue]
      discount_time[:-1],
    )
    batch_loss = 0.5 * jnp.square(batch_td_error).sum(axis=0)
    # Debug TD errors
    if self.config.debug:
      jax.debug.print(
        "TD error mean: {}, min: {}, max: {}",
        jnp.mean(batch_td_error),
        jnp.min(batch_td_error),
        jnp.max(batch_td_error),
      )
      jax.debug.print("Batch loss mean: {}", jnp.mean(batch_loss))

    # Importance weighting.
    probs = jax.nn.softmax(experience_state.priority, axis=1)[jnp.arange(B), sampled_seq_idx]
    importance_weights = (1.0 / (probs + 1e-6)).astype(online_q.dtype)
    importance_weights **= self.config.importance_sampling_priority_exponent
    importance_weights /= jnp.max(importance_weights)
    mean_loss = jnp.mean(importance_weights * batch_loss)

    # Calculate priorities as a mixture of max and mean sequence errors.
    abs_td_error = jnp.abs(batch_td_error).astype(online_q.dtype)
    max_priority = self.config.max_priority_weight * jnp.max(abs_td_error, axis=0)
    mean_priority = (1 - self.config.max_priority_weight) * jnp.mean(abs_td_error, axis=0)
    priorities = max_priority + mean_priority
    new_priority = experience_state.priority.at[jnp.arange(B), sampled_seq_idx].set(priorities)

    new_experience_state = dataclasses.replace(experience_state, priority=new_priority)

    mean_loss = jnp.mean(batch_loss)
    return mean_loss, new_experience_state

  def _update_experience(
    self,
    experience_state: R2D2ExperienceState,
    actor_state_pre: R2D2ActorState,
    actor_state_post: R2D2ActorState,
    trajectory: EnvStep,
  ) -> R2D2ExperienceState:
    B = self._experience_state_batch_size()
    if trajectory.reward.shape[0] != B:
      raise ValueError(
        f"trajectory.reward.shape[0] must be equal to num_envs_per_learner, got "
        f"{trajectory.reward.shape[0]} and {B}"
      )
    if trajectory.reward.shape[1] != self.config.replay_seq_length:
      raise ValueError(
        f"trajectory.reward.shape[1] must be equal to replay_seq_length, got "
        f"{trajectory.reward.shape[1]} and {self.config.replay_seq_length}"
      )
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

    def update_seq_val(buffer, pointer, new_val):
      seq_index = pointer // seq_length
      return buffer.at[jnp.arange(B), seq_index].set(new_val)

    if self.config.store_hidden_states:
      new_hidden_h = update_seq_val(
        experience_state.hidden_state_h,
        experience_state.pointer,
        actor_state_pre.lstm_h_c[0],
      )
      new_hidden_c = update_seq_val(
        experience_state.hidden_state_c,
        experience_state.pointer,
        actor_state_pre.lstm_h_c[1],
      )
    else:
      new_hidden_h = experience_state.hidden_state_h
      new_hidden_c = experience_state.hidden_state_c
    new_ptr = (experience_state.pointer + seq_length) % buffer_capacity
    new_size = jnp.minimum(experience_state.size + seq_length, buffer_capacity)
    new_priority = update_seq_val(
      experience_state.priority, experience_state.pointer, jnp.ones((B,), dtype=jnp.float32)
    )
    return R2D2ExperienceState(
      observation=new_obs,
      action=new_actions,
      reward=new_rewards,
      new_episode=new_dones,
      pointer=new_ptr,
      size=new_size,
      priority=new_priority,
      hidden_state_h=new_hidden_h,
      hidden_state_c=new_hidden_c,
    )

  def num_off_policy_optims_per_cycle(self) -> int:
    return self.config.num_off_policy_optims_per_cycle

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

    # Soft update target network with a smaller step size
    # This increases stability compared to hard/periodic updates
    target_arrays, target_other = eqx.partition(nets.target, eqx.is_array)
    online_arrays, _ = eqx.partition(new_online, eqx.is_array)

    if self.config.target_update_step_size > 0:
      target_arrays = optax.incremental_update(
        online_arrays, target_arrays, self.config.target_update_step_size
      )
    else:
      assert self.config.target_update_period
      target_arrays = optax.periodic_update(
        online_arrays,
        target_arrays,
        opt_state.optimize_count,
        self.config.target_update_period,
      )

    assert isinstance(target_arrays, R2D2Network)
    new_nets = R2D2Networks(online=new_online, target=eqx.combine(target_arrays, target_other))

    new_opt_state = R2D2OptState(
      optax_state=optax_state,
      optimize_count=opt_state.optimize_count + 1,
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
      num_steps=jax.device_put_replicated(actor_state.num_steps, learner_devices),
    )
