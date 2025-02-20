import dataclasses
from collections.abc import Sequence
from typing import Any

import chex
import equinox as eqx
import gymnax.environments.spaces as gymnax_spaces
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PRNGKeyArray, PyTree, Scalar

from earl.agents.r2d2.networks import R2D2Networks
from earl.core import ActionAndState, Agent, AgentState, EnvStep, Metrics
from earl.utils.sharding import shard_along_axis_0


def _value_rescale(x: jnp.ndarray, eps: float) -> jnp.ndarray:
  """Value rescaling function.

  From section 2.3:  h(x) = sign(x) * (√(|x|+1) - 1) + ε*x
  """
  return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + eps * x


def _inverse_value_rescale(y: jnp.ndarray, eps: float) -> jnp.ndarray:
  """Inverse of _value_rescale."""
  return jnp.sign(y) * (((jnp.sqrt(4 * eps * (jnp.abs(y) + 1 + eps) + 1) - 1) / (2 * eps)) ** 2 - 1)


# ------------------------------------------------------------------
class R2D2OptState(eqx.Module):
  optax_state: optax.OptState
  target_update_count: jax.Array  # scalar; counts steps since last target network update


# R2D2 Actor state holds the LSTMCell's recurrent state and a PRNG key.
class R2D2ActorState(eqx.Module):
  lstm_h_c: Any  # Tuple of (h, c); see Section 2.3.
  key: PRNGKeyArray


# R2D2 Experience state holds a replay buffer of fixed-length sequences.
# We store one hidden state per sequence.
class R2D2ExperienceState(eqx.Module):
  observations: jax.Array  # shape: (num_sequences, seq_length, *obs_shape)
  actions: jax.Array  # shape: (num_sequences, seq_length)
  rewards: jax.Array  # shape: (num_sequences, seq_length)
  dones: jax.Array  # shape: (num_sequences, seq_length); True indicates episode end
  pointer: jax.Array  # scalar int, current insertion index in the sequence buffer
  size: jax.Array  # scalar int, current number of stored sequences
  hidden_states_h: jax.Array  # shape: (num_sequences, hidden_size) if store_hidden_states is true
  hidden_states_c: jax.Array  # shape: (num_sequences, hidden_size) if store_hidden_states is true


# Configuration for R2D2 agent.
@dataclasses.dataclass(eq=True, frozen=True)
class R2D2Config:
  discount: float = 0.997  # Section 2.3: discount factor.
  n_step: int = 5  # Section 2.3: n-step return.
  burn_in: int = 40  # Section 3: burn-in length for LSTMCell.
  priority_exponent: float = 0.9  # Section 2.3: priority exponent.
  target_update_period: int = 2500  # Section 2.3: period to update target network.
  buffer_capacity: int = 10000  # Total number of time steps in replay.
  update_experience_trajectory_length: int = 80  # trajectory length for each update.
  replay_seq_length: int = 80  # sequence length (m).
  store_hidden_states: bool = False  # If true, store hidden state for first time step per sequence.
  # Gradient clipping suggested by https://www.nature.com/articles/nature14236
  optimizer: optax.GradientTransformation = optax.chain(optax.clip(1.0), optax.adam(1e-4))
  td_lambda: float = 1.0  # Hyperparameter for lambda returns.
  num_optims_per_target_update: int = 1  # how frequently to update the target network.
  target_update_step_size: float = 0.01  # how much to update the target network.
  num_envs_per_learner: int = 0
  """Number of environments per learner. 0 means use env_info.num_envs."""
  value_rescaling_epsilon: float = 1e-3  # Epsilon parameter for h and h⁻¹.

  def __post_init__(self):
    if self.buffer_capacity % self.replay_seq_length:
      raise ValueError("buffer_size must be divisible by replay_seq_length.")
    if self.update_experience_trajectory_length % self.replay_seq_length:
      raise ValueError(
        "update_experience_trajectory_length must be divisible by replay_seq_length."
      )
    if self.buffer_capacity % self.update_experience_trajectory_length:
      raise ValueError("buffer_size must be divisible by update_experience_trajectory_length.")


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

  def _init_lstm_hidden(self, batch_size: int, lstm_cell: eqx.nn.LSTMCell) -> Any:
    h = jnp.zeros((batch_size, lstm_cell.hidden_size))
    c = jnp.zeros((batch_size, lstm_cell.hidden_size))
    return (h, c)

  def _new_experience_state(self, nets: R2D2Networks, key: PRNGKeyArray) -> R2D2ExperienceState:
    buffer_capacity = self.config.buffer_capacity
    num_trajectories = buffer_capacity // self.config.update_experience_trajectory_length
    num_envs = self.config.num_envs_per_learner or self.env_info.num_envs
    env_info = self.env_info
    assert isinstance(env_info.observation_space, gymnax_spaces.Box)
    obs_shape = env_info.observation_space.shape
    observations = jnp.zeros(
      (num_envs, buffer_capacity) + obs_shape, dtype=env_info.observation_space.dtype
    )
    action_space = env_info.action_space
    assert isinstance(action_space, gymnax_spaces.Discrete)
    actions = jnp.zeros((num_envs, buffer_capacity), dtype=action_space.dtype)
    rewards = jnp.zeros((num_envs, buffer_capacity))
    dones = jnp.zeros((num_envs, buffer_capacity), dtype=jnp.bool)
    pointer = jnp.zeros((num_envs,), dtype=jnp.uint32)
    size = jnp.zeros((num_envs,), dtype=jnp.uint32)
    if self.config.store_hidden_states:
      hidden_states_h = jnp.zeros((num_envs, num_trajectories, nets.online.lstm_cell.hidden_size))
      hidden_states_c = jnp.zeros((num_envs, num_trajectories, nets.online.lstm_cell.hidden_size))
    else:
      hidden_states_h = jnp.zeros(())
      hidden_states_c = jnp.zeros(())
    return R2D2ExperienceState(
      observations=observations,
      actions=actions,
      rewards=rewards,
      dones=dones,
      pointer=pointer,
      size=size,
      hidden_states_h=hidden_states_h,
      hidden_states_c=hidden_states_c,
    )

  def _new_opt_state(self, nets: R2D2Networks, key: PRNGKeyArray) -> R2D2OptState:
    return R2D2OptState(
      optax_state=self.config.optimizer.init(eqx.filter(nets.online, eqx.is_array)),
      target_update_count=jnp.array(0, dtype=jnp.uint32),
    )

  def _act(
    self, actor_state: R2D2ActorState, nets: R2D2Networks, env_step: EnvStep
  ) -> ActionAndState[R2D2ActorState]:
    q_values, new_h_c = eqx.filter_vmap(nets.online)(
      env_step.obs, env_step.prev_action, env_step.reward, actor_state.lstm_h_c
    )
    action = jnp.argmax(q_values, axis=-1)
    new_actor_state = R2D2ActorState(lstm_h_c=new_h_c, key=actor_state.key)
    return ActionAndState(action, new_actor_state)

  def _loss(
    self, nets: R2D2Networks, opt_state: R2D2OptState, experience_state: R2D2ExperienceState
  ) -> tuple[Scalar, Metrics]:
    seq_length = self.config.replay_seq_length
    burn_in = self.config.burn_in
    n_step = self.config.n_step
    discount = self.config.discount
    eps = self.config.value_rescaling_epsilon

    # Number of sequences (batch size).
    B = experience_state.observations.shape[0]

    if self.config.store_hidden_states:
      init_hidden = (experience_state.hidden_states_h, experience_state.hidden_states_c)
    else:
      init_hidden = self._init_lstm_hidden(B, nets.online.lstm_cell)

    # Unroll the online network over the sequence.
    obs_time = jnp.swapaxes(experience_state.observations, 0, 1)
    action_time = jnp.swapaxes(experience_state.actions, 0, 1)
    reward_time = jnp.swapaxes(experience_state.rewards, 0, 1)

    def scan_fn(hidden, oar):
      obs, action, reward = oar
      q, new_hidden = jax.vmap(nets.online)(obs, action, reward, hidden)
      return new_hidden, (q, new_hidden)

    final_hidden, (q_seq, hidden_seq) = jax.lax.scan(
      scan_fn, init_hidden, (obs_time, action_time, reward_time)
    )
    q_seq = jnp.swapaxes(q_seq, 0, 1)  # shape: (B, seq_length, num_actions)
    h_seq, c_seq = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), hidden_seq)

    actions = experience_state.actions  # shape: (B, seq_length)
    rewards = experience_state.rewards  # shape: (B, seq_length)
    dones = experience_state.dones  # shape: (B, seq_length)

    valid_ts = jnp.arange(burn_in, seq_length - n_step + 1)

    def td_error_for_t(t):
      discounts = discount ** jnp.arange(n_step)
      r_slice = jax.lax.dynamic_slice(
        rewards, (0, t), (rewards.shape[0], n_step)
      )  # shape: (B, n_step)
      r_cum = jnp.sum(r_slice * discounts, axis=-1)  # shape: (B,)

      dones_slice = jax.lax.dynamic_slice(
        dones, (0, t), (dones.shape[0], n_step)
      )  # shape: (B, n_step)
      mask = jnp.cumprod(1 - dones_slice.astype(jnp.float32), axis=-1)  # shape: (B,)

      q_online_tn = jax.lax.dynamic_slice(
        q_seq, (0, t + n_step, 0), (q_seq.shape[0], 1, q_seq.shape[2])
      )
      q_online_tn = q_online_tn.squeeze(1)  # shape: (B, num_actions)
      a_max = jnp.argmax(q_online_tn, axis=-1)  # shape: (B,)

      # Dynamically construct start indices and slice sizes for observations
      obs_start_indices = (0, t + n_step) + (0,) * (len(experience_state.observations.shape) - 2)
      obs_slice_sizes = (
        experience_state.observations.shape[0],
        1,
      ) + experience_state.observations.shape[2:]
      obs_tn = jax.lax.dynamic_slice(
        experience_state.observations,
        obs_start_indices,
        obs_slice_sizes,
      )
      obs_tn = obs_tn.squeeze(1)  # Remove the singleton time dimension
      h_tn = jax.lax.dynamic_slice(
        h_seq, (0, t + n_step, 0), (h_seq.shape[0], 1, h_seq.shape[2])
      ).squeeze(1)
      c_tn = jax.lax.dynamic_slice(
        c_seq, (0, t + n_step, 0), (c_seq.shape[0], 1, c_seq.shape[2])
      ).squeeze(1)
      hidden_tn = (h_tn, c_tn)
      a_tn = jax.lax.dynamic_slice(actions, (0, t + n_step), (actions.shape[0], 1)).squeeze(1)
      r_tn = jax.lax.dynamic_slice(rewards, (0, t + n_step), (rewards.shape[0], 1)).squeeze(1)
      q_target, _ = jax.vmap(nets.target)(obs_tn, a_tn, r_tn, hidden_tn)  # shape: (B, num_actions)
      q_target_val = jnp.take_along_axis(q_target, a_max[:, None], axis=-1).squeeze(
        -1
      )  # shape: (B,)

      unscaled_q_target = _inverse_value_rescale(q_target_val, eps)
      unscaled_target = r_cum + (discount**n_step) * mask * unscaled_q_target
      target = _value_rescale(unscaled_target, eps)

      q_t = jax.lax.dynamic_slice(q_seq, (0, t, 0), (q_seq.shape[0], 1, q_seq.shape[2]))
      q_t = q_t.squeeze(1)  # shape: (B, num_actions)
      actions_t = jax.lax.dynamic_slice(actions, (0, t), (actions.shape[0], 1))
      q_taken = jnp.take_along_axis(q_t, actions_t, axis=-1).squeeze(axis=1)
      return q_taken - target

    td_errors = jax.vmap(td_error_for_t)(valid_ts)  # shape: (T, B)
    loss = jnp.mean(jnp.square(td_errors))
    return loss, {}

  def _update_experience(
    self,
    experience_state: R2D2ExperienceState,
    actor_state_pre: R2D2ActorState,
    actor_state_post: R2D2ActorState,
    trajectory: EnvStep,
  ) -> R2D2ExperienceState:
    num_envs = self.config.num_envs_per_learner or self.env_info.num_envs
    if trajectory.reward.shape[0] != num_envs:
      raise ValueError("trajectory.reward.shape[0] must be equal to num_envs_per_learner.")
    if trajectory.reward.shape[1] != self.config.update_experience_trajectory_length:
      raise ValueError(
        "trajectory.reward.shape[1] must be equal to update_experience_trajectory_length."
      )
    buffer_capacity = self.config.buffer_capacity
    traj_len = self.config.update_experience_trajectory_length

    def update_buffer(buffer, pointer, data):
      chex.assert_shape(data, (traj_len, ...))
      chex.assert_shape(buffer, (buffer_capacity, ...))
      start_indices = (pointer,) + (jnp.array(0, dtype=jnp.uint32),) * (len(buffer.shape) - 1)
      return jax.lax.dynamic_update_slice(buffer, data, start_indices)

    new_obs = jax.vmap(update_buffer)(
      experience_state.observations,
      experience_state.pointer,
      trajectory.obs,
    )

    new_actions = jax.vmap(update_buffer)(
      experience_state.actions,
      experience_state.pointer,
      trajectory.prev_action,
    )

    new_rewards = jax.vmap(update_buffer)(
      experience_state.rewards,
      experience_state.pointer,
      trajectory.reward,
    )

    new_dones = jax.vmap(update_buffer)(
      experience_state.dones,
      experience_state.pointer,
      trajectory.new_episode,
    )

    def update_hidden(hidden_buffer, pointer, init_hidden):
      seq_index = pointer // traj_len
      return hidden_buffer.at[seq_index].set(init_hidden)

    if self.config.store_hidden_states:
      new_hidden_h = jax.vmap(update_hidden)(
        experience_state.hidden_states_h,
        experience_state.pointer,
        actor_state_pre.lstm_h_c[0],
      )
      new_hidden_c = jax.vmap(update_hidden)(
        experience_state.hidden_states_c,
        experience_state.pointer,
        actor_state_pre.lstm_h_c[1],
      )
    else:
      new_hidden_h = experience_state.hidden_states_h
      new_hidden_c = experience_state.hidden_states_c
    new_ptr = (experience_state.pointer + traj_len) % buffer_capacity
    new_size = jnp.minimum(experience_state.size + traj_len, buffer_capacity)
    return R2D2ExperienceState(
      observations=new_obs,
      actions=new_actions,
      rewards=new_rewards,
      dones=new_dones,
      pointer=new_ptr,
      size=new_size,
      hidden_states_h=new_hidden_h,
      hidden_states_c=new_hidden_c,
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
    new_target_update_count = opt_state.target_update_count + 1

    def update_target():
      return new_online, jnp.array(0, dtype=jnp.uint32)

    def keep_target():
      return nets.target, new_target_update_count

    new_target, final_count = jax.lax.cond(
      new_target_update_count >= self.config.target_update_period, update_target, keep_target
    )
    new_nets = R2D2Networks(online=new_online, target=new_target)
    new_opt_state = R2D2OptState(optax_state=optax_state, target_update_count=final_count)
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
