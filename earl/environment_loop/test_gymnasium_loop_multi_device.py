import dataclasses
import os
import typing

import gymnasium

# this doesn't work with pytest discovery because test discovery may import
# jax before we set the XLA_FLAGS environment variable.
# to run this test, use bazel or pytest this file directly.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
import jax
import numpy as np
import pytest
from jax_loop_utils.metric_writers.memory_writer import MemoryWriter

from earl.agents.random_agent.random_agent import RandomAgent
from earl.core import env_info_from_gymnasium
from earl.environment_loop.gymnasium_loop import GymnasiumLoop
from earl.utils.prng import keygen


class NoOpEnv(gymnasium.Env[np.int64, np.int64]):
  def __init__(self):
    self.action_space = gymnasium.spaces.Discrete(2)
    self.observation_space = gymnasium.spaces.Discrete(1)

  def reset(
    self, *, seed: int | None = None, options: dict[str, typing.Any] | None = None
  ) -> tuple[np.int64, dict[str, typing.Any]]:
    super().reset(seed=seed, options=options)
    return np.int64(0), {}

  def step(
    self, action: np.int64
  ) -> tuple[np.int64, typing.SupportsFloat, bool, bool, dict[str, typing.Any]]:
    return np.int64(0), 0.0, False, False, {}


def test_actor_learner_different_devices():
  # This test can run on two CPU devices using
  # XLA_FLAGS="--xla_force_host_platform_device_count=2"
  # See the BUILD.bazel file for details.
  cpu_devices = jax.devices("cpu")
  if len(cpu_devices) >= 2:
    devices = cpu_devices[:2]
  else:
    try:
      devices = (cpu_devices[0], jax.devices("gpu")[0])
    except RuntimeError:
      pytest.skip("requires at least 2 devices")

  num_envs = 2
  env = NoOpEnv()
  env_factory = NoOpEnv
  env_info = env_info_from_gymnasium(env, num_envs)
  key_gen = keygen(jax.random.PRNGKey(0))
  agent = RandomAgent(env_info, env_info.action_space.sample, 1)
  metric_writer = MemoryWriter()

  loop = GymnasiumLoop(
    env_factory,
    agent,
    num_envs,
    next(key_gen),
    metric_writer=metric_writer,
    actor_devices=devices[:1],
    learner_devices=devices[1:],
  )
  num_cycles = 2
  steps_per_cycle = 5
  agent_state = agent.new_state(None, jax.random.PRNGKey(0))
  # the default agent_state has nets=None.
  # We set it to an array to check for use-after-donation bugs.
  agent_state = dataclasses.replace(agent_state, nets=jax.numpy.ones((1,)))
  loop.run(agent_state, num_cycles, steps_per_cycle)
