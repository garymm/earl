import math

import ale_py
import chex
import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnax.environments.classic_control import CartPole
from gymnax.environments.spaces import Box, Discrete
from jax_loop_utils.metric_writers import MemoryWriter

import earl.agents.r2d2.networks as r2d2_networks
from earl.agents.r2d2.r2d2 import R2D2, R2D2Config
from earl.agents.r2d2.utils import update_buffer_batch
from earl.core import EnvInfo, env_info_from_gymnax
from earl.environment_loop.gymnax_loop import GymnaxLoop
from earl.experiments.run_experiment import _config_to_dict
from earl.metric_key import MetricKey

gymnasium.register_envs(ale_py)


def test_r2d2_accepts_atari_input():
  env = gymnasium.make("BreakoutNoFrameskip-v4")
  env = gymnasium.wrappers.AtariPreprocessing(env, noop_max=0)
  stack_size = 4
  env = gymnasium.wrappers.FrameStackObservation(env, stack_size=stack_size)
  obs = env.observation_space.sample()
  key = jax.random.PRNGKey(0)
  env.close()
  assert isinstance(env.action_space, gymnasium.spaces.Discrete)
  num_actions = int(env.action_space.n)
  action = jax.random.randint(key, (1,), 0, num_actions)
  reward = jax.random.uniform(key, ())
  hidden_size = 512
  hidden = (jnp.zeros((hidden_size,)), jnp.zeros((hidden_size,)))
  networks = r2d2_networks.make_networks_resnet(
    num_actions=num_actions,
    in_channels=stack_size,
    dtype=jnp.float32,
    hidden_size=hidden_size,
    key=key,
  )
  q_values, hiddens = networks.online(obs, action, reward, hidden)
  assert q_values.shape == (num_actions,)
  assert len(hiddens) == 2
  assert hiddens[0].shape == (hidden_size,)
  assert hiddens[1].shape == (hidden_size,)


def test_learns_cartpole():
  # NOTE: this test is pretty fragile. I had to search for good hyperparameters to get it to learn.
  # I think if it were run for many more cycles it would learn, but that's not appropriate for CI.
  env = CartPole()
  env_params = env.default_params
  action_space = env.action_space(env_params)
  assert isinstance(action_space, Discrete), action_space
  num_actions = int(action_space.n)
  observation_space = env.observation_space(env_params)
  assert isinstance(observation_space, Box), observation_space

  hidden_size = 32
  key = jax.random.PRNGKey(1)
  networks_key, loop_key, agent_key = jax.random.split(key, 3)
  networks = r2d2_networks.make_networks_mlp(
    num_actions=num_actions,
    input_size=int(math.prod(observation_space.shape)),
    dtype=jnp.float32,
    hidden_size=hidden_size,
    use_lstm=True,
    key=networks_key,
  )

  num_envs = 512
  burn_in = 10
  steps_per_cycle = 80 + burn_in
  num_cycles = 1000
  env_info = env_info_from_gymnax(env, env_params, num_envs)
  num_off_policy_optims_per_cycle = 1
  config = R2D2Config(
    epsilon_greedy_schedule_args=dict(
      init_value=0.5, end_value=0.01, transition_steps=steps_per_cycle * num_cycles
    ),
    q_learning_n_steps=2,
    debug=False,
    buffer_capacity=steps_per_cycle * 10,
    num_envs_per_learner=num_envs,
    replay_seq_length=steps_per_cycle,
    burn_in=burn_in,
    value_rescaling_epsilon=0.0,
    num_off_policy_optims_per_cycle=num_off_policy_optims_per_cycle,
    gradient_clipping_max_delta=1.0,
    learning_rate_schedule_name="cosine_onecycle_schedule",
    learning_rate_schedule_args=dict(
      transition_steps=steps_per_cycle * num_cycles // 2,
      peak_value=5e-4,
    ),
    target_update_step_size=0.001,
  )
  agent = R2D2(env_info, config)
  memory_writer = MemoryWriter()
  # For tweaking of hyperparameters, you can use the mlflow writer
  # and view metrics with `uv run --with mlflow mlflow server`
  # metric_writer = MultiWriter(
  #   (memory_writer, MlflowMetricWriter(experiment_name=env.name)),
  # )
  metric_writer = memory_writer
  config_dict = _config_to_dict(config)
  metric_writer.write_hparams(config_dict)
  loop = GymnaxLoop(env, env.default_params, agent, num_envs, loop_key, metric_writer=metric_writer)
  agent_state = agent.new_state(networks, agent_key)
  _ = loop.run(agent_state, num_cycles, steps_per_cycle)
  del agent_state
  metrics = memory_writer.scalars
  metric_writer.close()

  episode_lengths = np.array(
    [step_metrics[MetricKey.COMPLETE_EPISODE_LENGTH_MEAN] for step_metrics in metrics.values()]
  )

  assert len(metrics) == num_cycles
  # Due to auto-resets, the reward is always constant, but it's a survival task
  # so longer episodes are better.
  mean_over_cycles = 20
  first_mean = float(np.mean(episode_lengths[:mean_over_cycles]))
  assert first_mean > 0
  last_mean = float(np.mean(episode_lengths[-mean_over_cycles:]))
  assert last_mean > 1.5 * first_mean


_dummy_env_info = EnvInfo(
  num_envs=2,
  observation_space=Box(low=0, high=1, shape=(4,)),
  action_space=Discrete(num_categories=2),
  name="dummy",
)


@pytest.fixture
def mlp_agent_and_networks():
  key = jax.random.PRNGKey(0)
  env_info = _dummy_env_info
  config = R2D2Config(
    epsilon_greedy_schedule_args=dict(init_value=0.5, end_value=0.0001, transition_steps=10000),
    discount=0.99,
    q_learning_n_steps=3,
    burn_in=2,
    priority_exponent=0.9,
    target_update_period=10,
    buffer_capacity=8,  # must be divisible by replay_seq_length
    replay_seq_length=4,
    store_hidden_states=True,
    num_envs_per_learner=env_info.num_envs,
  )
  # Create dummy networks using the MLP builder
  networks = r2d2_networks.make_networks_mlp(
    num_actions=2, input_size=4, dtype=jnp.float32, hidden_size=32, key=key
  )
  agent = R2D2(env_info, config)
  return agent, networks


def test_slice_for_replay(mlp_agent_and_networks):
  agent, _ = mlp_agent_and_networks
  B = agent.env_info.num_envs
  T = 8
  dummy_data = jnp.arange(B * T).reshape(B, T, 1)  # shape (B, T, 1)
  start_idx = jnp.array([0, 4])  # one index per environment
  assert start_idx.shape == (B,)
  length = 4
  sliced = agent._slice_for_replay(dummy_data, start_idx, length)
  # Expected shape: (length, B, 1)
  assert sliced.shape == (length, B, 1)
  # Verify a couple of values
  chex.assert_equal(sliced[0, 0, 0], dummy_data[0, 0, 0])
  chex.assert_equal(sliced[0, 1, 0], dummy_data[1, 4, 0])


def test_sample_from_experience(mlp_agent_and_networks):
  agent, networks = mlp_agent_and_networks
  key = jax.random.PRNGKey(0)
  exp_state = agent._new_experience_state(networks, key)
  seq_idx = jnp.zeros(agent.env_info.num_envs, dtype=jnp.uint32)
  outputs = agent._sample_from_experience(seq_idx, networks, exp_state)
  obs_time, action_time, reward_time, dones_time, hidden_h_pre, hidden_c_pre = outputs
  assert obs_time.shape == (
    agent.config.replay_seq_length,
    agent.env_info.num_envs,
    agent.env_info.observation_space.shape[0],
  )
  assert action_time.shape == (agent.config.replay_seq_length, agent.env_info.num_envs)
  assert reward_time.shape == (agent.config.replay_seq_length, agent.env_info.num_envs)
  assert dones_time.shape == (agent.config.replay_seq_length, agent.env_info.num_envs)
  # Hidden state shapes
  assert hidden_h_pre.shape == (agent.env_info.num_envs, networks.online.lstm_cell.hidden_size)
  assert hidden_c_pre.shape == (agent.env_info.num_envs, networks.online.lstm_cell.hidden_size)


def test_update_buffer_batch():
  """Test the update_buffer_batch function with various scenarios."""
  # Set up parameters
  seq_length = 4
  buffer_capacity = 8
  num_envs = 2

  # Test with pointer at beginning
  buffer = jnp.zeros((num_envs, buffer_capacity), dtype=jnp.bool)
  pointer = jnp.array(0, dtype=jnp.uint32)
  data = jnp.array(
    [[True, False, True, False], [True, False, True, False]]
  )  # Shape is (num_envs, seq_length)
  updated_buffer = update_buffer_batch(buffer, pointer, data, debug=True)

  # Check buffer has been updated correctly
  expected_buffer = jnp.zeros((num_envs, buffer_capacity), dtype=jnp.bool)
  expected_buffer = expected_buffer.at[:, 0:4].set(data)
  chex.assert_trees_all_close(updated_buffer, expected_buffer)

  # Test with pointer in middle
  buffer = jnp.zeros((num_envs, buffer_capacity), dtype=jnp.bool)
  pointer = jnp.array(2, dtype=jnp.uint32)
  updated_buffer = update_buffer_batch(buffer, pointer, data, debug=True)

  # Check buffer has been updated correctly
  expected_buffer = jnp.zeros((num_envs, buffer_capacity), dtype=jnp.bool)
  expected_buffer = expected_buffer.at[:, 2:6].set(data)
  chex.assert_trees_all_close(updated_buffer, expected_buffer)

  # Test with nested dimensions
  hidden_size = 3
  buffer = jnp.zeros((num_envs, buffer_capacity, hidden_size), dtype=jnp.float32)
  data = jnp.ones((num_envs, seq_length, hidden_size), dtype=jnp.float32)
  data = data * jnp.arange(1, num_envs + 1).reshape(num_envs, 1, 1)  # Different values per env

  pointer = jnp.array(1, dtype=jnp.uint32)
  updated_buffer = update_buffer_batch(buffer, pointer, data, debug=True)

  # Check buffer shape and updated values
  chex.assert_shape(updated_buffer, (num_envs, buffer_capacity, hidden_size))

  # Verify values in updated region
  expected_buffer = jnp.zeros((num_envs, buffer_capacity, hidden_size), dtype=jnp.float32)
  expected_buffer = expected_buffer.at[:, 1:5].set(data)
  chex.assert_trees_all_close(updated_buffer, expected_buffer)

  # Also verify specific values to ensure environment-specific data was preserved
  for env_idx in range(num_envs):
    # Check that the updated region has values equal to env_idx + 1
    chex.assert_trees_all_close(
      updated_buffer[env_idx, 1:5], jnp.ones((4, hidden_size)) * (env_idx + 1)
    )
    # Check that areas outside the updated region remain zeros
    chex.assert_trees_all_close(updated_buffer[env_idx, 0], jnp.zeros(hidden_size))
    chex.assert_trees_all_close(updated_buffer[env_idx, 5:], jnp.zeros((3, hidden_size)))
