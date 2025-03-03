import dataclasses
import functools
import io
import math

import chex
import envpool
import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnax.environments.classic_control import CartPole
from gymnax.environments.spaces import Box, Discrete
from jax_loop_utils.metric_writers import MemoryWriter
from jax_loop_utils.metric_writers._audio_video import encode_video_to_gif

import earl.agents.r2d2.networks as r2d2_networks
from earl.agents.r2d2.r2d2 import R2D2, R2D2Config
from earl.agents.r2d2.utils import render_atari_cycle, update_buffer_batch
from earl.core import EnvInfo, env_info_from_gymnasium, env_info_from_gymnax
from earl.environment_loop.gymnasium_loop import GymnasiumLoop
from earl.environment_loop.gymnax_loop import GymnaxLoop
from earl.metric_key import MetricKey


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
      peak_value=2e-4,
    ),
    target_update_step_size=0.00,
    target_update_period=100,
  )
  agent = R2D2(env_info, config)
  memory_writer = MemoryWriter()
  # For tweaking of hyperparameters, you can use the mlflow writer
  # and view metrics with `uv run --with mlflow mlflow server`
  # metric_writer = MultiWriter(
  #   (memory_writer, MlflowMetricWriter(experiment_name=env.name)),
  # )
  metric_writer = memory_writer
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
  assert last_mean > 1.4 * first_mean


# triggered by envpool
def test_train_atari():
  stack_num = 4
  num_envs = 6
  input_size = (84, 84)
  env_factory = functools.partial(
    envpool.make_gymnasium,
    "Asterix-v5",
    num_envs=num_envs,
    stack_num=stack_num,
    img_height=input_size[0],
    img_width=input_size[1],
  )
  with env_factory() as env:
    assert isinstance(env.action_space, gymnasium.spaces.Discrete)
    num_actions = int(env.action_space.n)
  devices = jax.local_devices()
  if len(devices) > 1:
    actor_devices = devices[:1]
    learner_devices = devices[1:]
  else:
    actor_devices = devices
    learner_devices = devices
  print(f"running on {len(actor_devices)} actor devices and {len(learner_devices)} learner devices")
  if actor_devices == learner_devices:
    print("WARNING: actor and learner devices are the same. They will compete for the devices.")
  env_info = env_info_from_gymnasium(env, num_envs)
  hidden_size = 512
  key = jax.random.PRNGKey(0)
  networks_key, loop_key, agent_key = jax.random.split(key, 3)
  networks = r2d2_networks.make_networks_resnet(
    num_actions=num_actions,
    in_channels=stack_num,
    hidden_size=hidden_size,
    input_size=input_size,
    key=networks_key,
  )
  steps_per_cycle = 80

  config = R2D2Config(
    epsilon_greedy_schedule_args=dict(
      init_value=0.9, end_value=0.01, transition_steps=steps_per_cycle * 1_000
    ),
    num_envs_per_learner=num_envs,
    replay_seq_length=steps_per_cycle,
    buffer_capacity=steps_per_cycle * 10,
    burn_in=40,
    learning_rate_schedule_name="cosine_onecycle_schedule",
    learning_rate_schedule_args=dict(
      transition_steps=steps_per_cycle * 2_500,
      # NOTE: more devices effectively means a larger batch size, so we
      # scale the learning rate up to train faster!
      peak_value=5e-5 * len(devices),
    ),
    target_update_step_size=0.00,
    target_update_period=500,
  )
  agent = R2D2(env_info, config)
  loop_state = agent.new_state(networks, agent_key)
  metric_writer = MemoryWriter()
  train_loop = GymnasiumLoop(
    env_factory,
    agent,
    num_envs,
    loop_key,
    observe_cycle=render_atari_cycle,
    metric_writer=metric_writer,
    actor_devices=actor_devices,
    learner_devices=learner_devices,
    vectorization_mode="none",
  )
  # just one cycle, make sure it runs
  loop_state = train_loop.run(loop_state, 1, steps_per_cycle)
  train_loop.close()
  # make sure we can render the video
  video_buf = io.BytesIO()
  video_array = next(iter(metric_writer.videos.values()))["video"]
  encode_video_to_gif(video_array, video_buf)
  # for manual inspection, uncomment
  # with open("asterix_initial.gif", "wb") as f:
  #   f.write(video_buf.getvalue())


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
    importance_sampling_priority_exponent=0.9,
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
  assert sliced.shape == (B, length, 1)
  # Verify a couple of values
  chex.assert_equal(sliced[0, 0, 0], dummy_data[0, 0, 0])
  chex.assert_equal(sliced[1, 0, 0], dummy_data[1, 4, 0])


def test_sample_from_experience(mlp_agent_and_networks):
  agent, networks = mlp_agent_and_networks
  agent = dataclasses.replace(
    agent, config=dataclasses.replace(agent.config, replay_batch_size=2 * agent.env_info.num_envs)
  )
  key = jax.random.PRNGKey(0)
  exp_state = agent._new_experience_state(networks, key)
  outputs = agent._sample_from_experience(networks, key, exp_state)
  sampled_seq_idx, obs_time, action_time, reward_time, dones_time, hidden_h_pre, hidden_c_pre = (
    outputs
  )
  assert sampled_seq_idx.shape == (agent.config.replay_batch_size,)
  assert obs_time.shape == (
    agent.config.replay_seq_length,
    agent.config.replay_batch_size,
    agent.env_info.observation_space.shape[0],
  )
  assert action_time.shape == (agent.config.replay_seq_length, agent.config.replay_batch_size)
  assert reward_time.shape == (agent.config.replay_seq_length, agent.config.replay_batch_size)
  assert dones_time.shape == (agent.config.replay_seq_length, agent.config.replay_batch_size)
  # Hidden state shapes
  assert hidden_h_pre.shape == (
    agent.config.replay_batch_size,
    networks.online.lstm_cell.hidden_size,
  )
  assert hidden_c_pre.shape == (
    agent.config.replay_batch_size,
    networks.online.lstm_cell.hidden_size,
  )


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
