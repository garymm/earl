import dataclasses
from typing import Any

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PRNGKeyArray, PyTree, Scalar

from earl.core import ActionAndState, Agent, AgentState, EnvStep, Metrics


# Custom OptState containing the optax optimizer state and a target_update_count.
class OptState(eqx.Module):
  opt_state: optax.OptState
  target_update_count: jax.Array  # scalar; counts steps since last target network update


# Bundled networks: online and target.
class R2D2Networks(eqx.Module):
  online: "R2D2Network"
  target: "R2D2Network"


# R2D2 Actor state holds the LSTMCell’s recurrent state and a PRNG key.
class R2D2ActorState(eqx.Module):
  lstm_hidden: Any  # Tuple of (h, c); see Section 2.3.
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

  def __post_init__(self):
    num_sequences = self.observations.shape[0]
    if num_sequences <= 0:
      raise ValueError("Invalid buffer dimensions.")


# Configuration for R2D2 agent.
@dataclasses.dataclass(eq=True, frozen=True)
class Config:
  discount: float = 0.997  # Section 2.3: discount factor.
  n_step: int = 5  # Section 2.3: n-step return.
  burn_in: int = 40  # Section 3: burn-in length for LSTMCell.
  priority_exponent: float = 0.9  # Section 2.3: priority exponent.
  target_update_period: int = 2500  # Section 2.3: period to update target network.
  buffer_size: int = 10000  # Total number of time steps in replay.
  seq_length: int = 80  # Fixed sequence length (m).
  store_hidden_states: bool = False  # If true, store hidden state for first time step per sequence.
  optimizer: optax.GradientTransformation = optax.adam(1e-4)
  td_lambda: float = 1.0  # Hyperparameter for lambda returns.


# The R2D2 network: a convolutional feature extractor, an LSTMCell, and a dueling head.
class R2D2Network(eqx.Module):
  conv: eqx.nn.Sequential  # Section 2.3: convolutional feature extractor.
  lstm_cell: eqx.nn.LSTMCell  # Section 2.3 & 3: recurrent cell.
  dueling_value: eqx.nn.Linear  # Section 2.3: value branch.
  dueling_advantage: eqx.nn.Linear  # Section 2.3: advantage branch.

  def __call__(self, x: jax.Array, hidden: Any) -> tuple[jax.Array, Any]:
    features = self.conv(x)  # Section 2.3.
    features = features.reshape((features.shape[0], -1))
    new_hidden, lstm_out = self.lstm_cell(hidden, features)  # Section 3.
    value = self.dueling_value(lstm_out)  # Section 2.3.
    advantage = self.dueling_advantage(lstm_out)  # Section 2.3.
    q_values = value + (advantage - jnp.mean(advantage, axis=-1, keepdims=True))
    return q_values, new_hidden


# Update the AgentState alias to use bundled networks.
R2D2AgentState = AgentState[R2D2Networks, OptState, R2D2ExperienceState, R2D2ActorState]


# Below copied from rlax because it's package is unusable with Bazel.
# https://github.com/google-deepmind/rlax/issues/133
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


class R2D2(Agent[R2D2Networks, OptState, R2D2ExperienceState, R2D2ActorState]):
  config: Config

  def _new_actor_state(self, nets: R2D2Networks, key: PRNGKeyArray) -> R2D2ActorState:
    batch_size = self.env_info.num_envs
    init_hidden = self._init_lstm_hidden(batch_size, nets.online.lstm_cell)
    return R2D2ActorState(lstm_hidden=init_hidden, key=key)

  def _init_lstm_hidden(self, batch_size: int, lstm_cell: eqx.nn.LSTMCell) -> Any:
    h = jnp.zeros((batch_size, lstm_cell.hidden_size))
    c = jnp.zeros((batch_size, lstm_cell.hidden_size))
    return (h, c)

  def _new_experience_state(self, nets: R2D2Networks, key: PRNGKeyArray) -> R2D2ExperienceState:
    total_steps = self.config.buffer_size
    seq_length = self.config.seq_length
    if total_steps % seq_length != 0:
      raise ValueError("buffer_size must be divisible by seq_length.")
    num_sequences = total_steps // seq_length
    obs_shape = self.env_info.observation_space.shape
    observations = jnp.zeros((num_sequences, seq_length) + obs_shape)
    actions = jnp.zeros((num_sequences, seq_length), dtype=jnp.int32)
    rewards = jnp.zeros((num_sequences, seq_length))
    dones = jnp.zeros((num_sequences, seq_length), dtype=jnp.bool_)
    pointer = jnp.array(0, dtype=jnp.int32)
    size = jnp.array(0, dtype=jnp.int32)
    if self.config.store_hidden_states:
      hidden_states_h = jnp.zeros((num_sequences, nets.online.lstm_cell.hidden_size))
      hidden_states_c = jnp.zeros((num_sequences, nets.online.lstm_cell.hidden_size))
    else:
      hidden_states_h = jnp.zeros((num_sequences, 0))
      hidden_states_c = jnp.zeros((num_sequences, 0))
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

  def _new_opt_state(self, nets: R2D2Networks, key: PRNGKeyArray) -> OptState:
    return OptState(
      opt_state=self.config.optimizer.init(eqx.filter(nets.online, eqx.is_array)),
      target_update_count=jnp.array(0, dtype=jnp.uint32),
    )

  def _act(
    self, actor_state: R2D2ActorState, nets: R2D2Networks, env_step: EnvStep
  ) -> ActionAndState[R2D2ActorState]:
    q_values, new_hidden = nets.online(env_step.obs, actor_state.lstm_hidden)
    action = jnp.argmax(q_values, axis=-1)
    new_actor_state = R2D2ActorState(lstm_hidden=new_hidden, key=actor_state.key)
    return ActionAndState(action, new_actor_state)

  def _loss(
    self, nets: R2D2Networks, opt_state: OptState, experience_state: R2D2ExperienceState
  ) -> tuple[Scalar, Metrics]:
    seq_length = self.config.seq_length
    burn_in = self.config.burn_in
    n_step = self.config.n_step
    discount = self.config.discount

    # Number of sequences (batch size).
    B = experience_state.observations.shape[0]

    # Use stored hidden state if available; otherwise, initialize using online network.
    if self.config.store_hidden_states:
      init_hidden = (experience_state.hidden_states_h, experience_state.hidden_states_c)
    else:
      init_hidden = self._init_lstm_hidden(B, nets.online.lstm_cell)

    # Unroll the online network over the sequence.
    obs_time = jnp.swapaxes(experience_state.observations, 0, 1)

    def scan_fn(hidden, x):
      new_hidden, q = nets.online(x, hidden)
      return new_hidden, (q, new_hidden)

    final_hidden, (q_seq, hidden_seq) = jax.lax.scan(scan_fn, init_hidden, obs_time)
    q_seq = jnp.swapaxes(q_seq, 0, 1)  # shape: (B, seq_length, num_actions)
    h_seq, c_seq = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), hidden_seq)

    actions = experience_state.actions  # shape: (B, seq_length)
    rewards = experience_state.rewards  # shape: (B, seq_length)
    dones = experience_state.dones  # shape: (B, seq_length)

    valid_ts = jnp.arange(burn_in, seq_length - n_step + 1)

    def td_error_for_t(t):
      # Compute n-step cumulative reward.
      discounts = discount ** jnp.arange(n_step)
      r_slice = rewards[:, t : t + n_step]
      r_cum = jnp.sum(r_slice * discounts, axis=-1)  # shape: (B,)
      # Compute mask to zero out bootstrapped value if any done.
      mask = jnp.prod(1 - dones[:, t : t + n_step].astype(jnp.float32), axis=-1)  # shape: (B,)
      # Online network's Q-values at time t+n_step.
      q_online_tn = q_seq[:, t + n_step, :]  # shape: (B, num_actions)
      # Use target network to evaluate Q-values at time t+n_step.
      obs_tn = experience_state.observations[:, t + n_step]
      hidden_tn = (h_seq[:, t + n_step, :], c_seq[:, t + n_step, :])
      q_target, _ = nets.target(obs_tn, hidden_tn)  # shape: (B, num_actions)
      # Q-value at time t for taken action.
      q_taken = jnp.take_along_axis(q_seq[:, t, :], actions[:, t : t + 1], axis=-1).squeeze(axis=1)
      error = double_q_learning(
        q_tm1=q_taken,
        a_tm1=actions[:, t],
        r_t=r_cum,
        discount_t=(discount**n_step) * mask,
        q_t_value=q_target,
        q_t_selector=q_online_tn,
        stop_target_gradients=True,
      )
      return error

    td_errors = jax.vmap(td_error_for_t)(valid_ts)  # shape: (T, B)
    loss = jnp.mean(jnp.square(td_errors))
    return loss, {}

  def _optimize_from_grads(
    self, nets: R2D2Networks, opt_state: OptState, nets_grads: PyTree
  ) -> tuple[R2D2Networks, OptState]:
    updates, new_inner_opt_state = self.config.optimizer.update(nets_grads, opt_state.opt_state)
    new_online = eqx.apply_updates(nets.online, updates)
    new_target_update_count = opt_state.target_update_count + 1

    # Periodically update target network (Section 2.3, double Q-learning).
    def update_target():
      return new_online, jnp.array(0, dtype=jnp.uint32)

    def keep_target():
      return nets.target, new_target_update_count

    new_target, final_count = jax.lax.cond(
      new_target_update_count >= self.config.target_update_period, update_target, keep_target
    )
    new_nets = R2D2Networks(online=new_online, target=new_target)
    new_opt_state = OptState(opt_state=new_inner_opt_state, target_update_count=final_count)
    return new_nets, new_opt_state

  def _update_experience(
    self,
    experience_state: R2D2ExperienceState,
    actor_state_pre: R2D2ActorState,
    actor_state_post: R2D2ActorState,
    trajectory: EnvStep,
  ) -> R2D2ExperienceState:
    total_steps = self.config.buffer_size
    seq_length = self.config.seq_length
    num_sequences = total_steps // seq_length
    idx = experience_state.pointer
    new_obs = experience_state.observations.at[idx].set(trajectory.obs)
    new_actions = experience_state.actions.at[idx].set(trajectory.prev_action)
    new_rewards = experience_state.rewards.at[idx].set(trajectory.reward)
    new_dones = experience_state.dones.at[idx].set(trajectory.new_episode)
    if self.config.store_hidden_states:
      new_hidden_h = experience_state.hidden_states_h.at[idx].set(actor_state_pre.lstm_hidden[0])
      new_hidden_c = experience_state.hidden_states_c.at[idx].set(actor_state_pre.lstm_hidden[1])
    else:
      new_hidden_h = experience_state.hidden_states_h
      new_hidden_c = experience_state.hidden_states_c
    new_pointer = (idx + 1) % num_sequences
    new_size = jnp.minimum(experience_state.size + 1, num_sequences)
    return R2D2ExperienceState(
      observations=new_obs,
      actions=new_actions,
      rewards=new_rewards,
      dones=new_dones,
      pointer=new_pointer,
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
