import ale_py
import gymnasium
import jax
import jax.numpy as jnp
from jax_loop_utils.metric_writers import MemoryWriter

import earl.agents.r2d2.networks as r2d2_networks
from earl.agents.r2d2.r2d2 import R2D2, R2D2Config
from earl.core import env_info_from_gymnasium
from earl.environment_loop.gymnasium_loop import GymnasiumLoop

gymnasium.register_envs(ale_py)


def test_r2d2_accepts_atari_input():
  env = gymnasium.make("BreakoutNoFrameskip-v4")
  env = gymnasium.wrappers.AtariPreprocessing(env, noop_max=0)
  stack_size = 4
  env = gymnasium.wrappers.FrameStackObservation(env, stack_size=stack_size)
  obs = env.observation_space.sample()
  key = jax.random.PRNGKey(0)
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


def test_r2d2_atari_training():
  env = gymnasium.make("BreakoutNoFrameskip-v4")
  env = gymnasium.wrappers.AtariPreprocessing(env, noop_max=0)
  stack_size = 4
  env = gymnasium.wrappers.FrameStackObservation(env, stack_size=stack_size)
  assert isinstance(env.action_space, gymnasium.spaces.Discrete)
  num_actions = int(env.action_space.n)
  hidden_size = 512
  key = jax.random.PRNGKey(0)
  networks_key, loop_key, agent_key = jax.random.split(key, 3)
  networks = r2d2_networks.make_networks_resnet(
    num_actions=num_actions,
    in_channels=stack_size,
    dtype=jnp.float32,
    hidden_size=hidden_size,
    key=networks_key,
  )
  num_envs = 2
  env_info = env_info_from_gymnasium(env, num_envs)
  config = R2D2Config(num_envs_per_learner=num_envs)
  agent = R2D2(env_info, config)
  agent_state = agent.new_state(networks, agent_key)
  num_cycles = 2
  steps_per_cycle = 10
  metric_writer = MemoryWriter()
  loop = GymnasiumLoop(env, agent, num_envs, loop_key, metric_writer=metric_writer)

  _ = loop.run(agent_state, num_cycles, steps_per_cycle)
  del agent_state
  metrics = metric_writer.scalars
  print(metrics)
