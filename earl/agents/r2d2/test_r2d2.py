import dataclasses
import io
import os

import ale_py
import chex
import gymnasium
import jax
import jax.numpy as jnp
import pytest
from gymnax.environments.spaces import Box, Discrete
from jax_loop_utils.metric_writers import MemoryWriter
from jax_loop_utils.metric_writers._audio_video import encode_video_to_gif

import earl.agents.r2d2.networks as r2d2_networks
from earl.agents.r2d2.r2d2 import R2D2, R2D2Config
from earl.agents.r2d2.utils import render_atari_cycle, update_buffer_batch
from earl.core import EnvInfo, env_info_from_gymnasium
from earl.environment_loop.gymnasium_loop import GymnasiumLoop

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


def test_train_atari():
  env = gymnasium.make("AsterixNoFrameskip-v4")
  env = gymnasium.wrappers.AtariPreprocessing(env, noop_max=0)
  stack_size = 4
  env = gymnasium.wrappers.FrameStackObservation(env, stack_size=stack_size)
  assert isinstance(env.action_space, gymnasium.spaces.Discrete)
  num_actions = int(env.action_space.n)
  devices = jax.local_devices()
  if len(devices) > 1:
    actor_devices = devices[: max(1, len(devices) // 3)]
    learner_devices = devices[len(actor_devices) :]
  else:
    actor_devices = devices
    learner_devices = devices
  print(f"running on {len(actor_devices)} actor devices and {len(learner_devices)} learner devices")
  if actor_devices == learner_devices:
    print("WARNING: actor and learner devices are the same. They will compete for the devices.")
  cpu_count = os.cpu_count() or 1
  num_envs = min(32, max(1, cpu_count // len(actor_devices)))
  env_info = env_info_from_gymnasium(env, num_envs)
  hidden_size = 512
  key = jax.random.PRNGKey(0)
  networks_key, loop_key, agent_key = jax.random.split(key, 3)
  networks = r2d2_networks.make_networks_resnet(
    num_actions=num_actions,
    in_channels=stack_size,
    hidden_size=hidden_size,
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
    env,
    agent,
    num_envs,
    loop_key,
    observe_cycle=render_atari_cycle,
    metric_writer=metric_writer,
    actor_devices=actor_devices,
    learner_devices=learner_devices,
  )
  # just one cycle, make sure it runs
  loop_state = train_loop.run(loop_state, 1, steps_per_cycle)
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
  # Expected shape: (length, B, 1)
  assert sliced.shape == (length, B, 1)
  # Verify a couple of values
  chex.assert_equal(sliced[0, 0, 0], dummy_data[0, 0, 0])
  chex.assert_equal(sliced[0, 1, 0], dummy_data[1, 4, 0])


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
