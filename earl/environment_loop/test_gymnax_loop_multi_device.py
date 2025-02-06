import os
from unittest import mock

import gymnax
import gymnax.environments.spaces

# this doesn't work with pytest discovery because test discovery may import
# jax before we set the XLA_FLAGS environment variable.
# to run this test, use bazel or pytest this file directly.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
import jax
import jax.numpy as jnp
from jax_loop_utils.metric_writers.noop_writer import NoOpWriter

from earl.agents.random_agent.random_agent import RandomAgent
from earl.core import EnvStep, env_info_from_gymnax
from earl.environment_loop.gymnax_loop import GymnaxLoop, State
from earl.utils.prng import keygen


def test_run_with_state():
  env, env_params = gymnax.make("CartPole-v1")
  num_envs = 2
  obs, env_state = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), num_envs))
  key_gen = keygen(jax.random.PRNGKey(0))
  agent = RandomAgent(env.action_space().sample, 1)
  env.reset = mock.Mock(spec=env.reset)
  devices = jax.local_devices()
  loop = GymnaxLoop(
    env, env_params, agent, num_envs, next(key_gen), metric_writer=NoOpWriter(), devices=devices
  )
  agent_state = agent.new_state(
    None, env_info_from_gymnax(env, env_params, num_envs), jax.random.PRNGKey(0)
  )
  initial_step_num = 2
  prev_action = jax.vmap(env.action_space(None).sample)(
    jax.random.split(jax.random.PRNGKey(0), num_envs)
  )
  assert isinstance(prev_action, jax.Array)
  env_step = EnvStep(
    new_episode=jnp.zeros((num_envs,), dtype=jnp.bool),
    obs=obs,
    prev_action=prev_action,
    reward=jnp.zeros((num_envs,)),
  )
  state = State(
    agent_state=jax.device_put_replicated(agent_state, devices=devices),
    env_state=jax.device_put_replicated(env_state, devices=devices),
    env_step=jax.device_put_replicated(env_step, devices=devices),
    step_num=initial_step_num,
  )
  steps_per_cycle = 10
  _result = loop.run(state, 1, steps_per_cycle)
  assert env.reset.call_count == 0
