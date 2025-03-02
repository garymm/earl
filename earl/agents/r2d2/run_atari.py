import functools
import os

import envpool
import gymnasium
import jax
from jax_loop_utils.metric_writers.torch import TensorboardWriter

import earl.agents.r2d2.networks as r2d2_networks
from earl.agents.r2d2.r2d2 import R2D2, R2D2Config
from earl.agents.r2d2.utils import render_atari_cycle
from earl.core import env_info_from_gymnasium
from earl.environment_loop.gymnasium_loop import GymnasiumLoop

if __name__ == "__main__":
  devices = jax.local_devices()
  if len(devices) > 1:
    actor_devices = devices[: max(1, len(devices) // 3)]
    learner_devices = devices[len(actor_devices) :]
  else:
    actor_devices = devices
    learner_devices = devices
  cpu_count = os.cpu_count() or 1
  num_envs = max(1, cpu_count // len(actor_devices))
  assert num_envs % len(learner_devices) == 0
  print(f"num_envs: {num_envs}")
  env_factory = functools.partial(envpool.make_gymnasium, "Asterix-v5", num_envs=num_envs)
  stack_size = 4
  env = env_factory()
  assert isinstance(env.action_space, gymnasium.spaces.Discrete)
  num_actions = int(env.action_space.n)
  env.close()
  print(f"running on {len(actor_devices)} actor devices and {len(learner_devices)} learner devices")
  if actor_devices == learner_devices:
    print("WARNING: actor and learner devices are the same. They will compete for the devices.")
  env_info = env_info_from_gymnasium(env, num_envs)
  del env
  hidden_size = 512
  key = jax.random.PRNGKey(0)
  networks_key, loop_key, agent_key = jax.random.split(key, 3)
  networks = r2d2_networks.make_networks_resnet(
    num_actions=num_actions,
    in_channels=stack_size,
    hidden_size=hidden_size,
    key=networks_key,
    dtype=jax.numpy.bfloat16,
  )
  steps_per_cycle = 80
  num_cycles = 10_000

  config = R2D2Config(
    epsilon_greedy_schedule_args=dict(
      init_value=0.9, end_value=0.01, transition_steps=steps_per_cycle * 1_000
    ),
    num_envs_per_learner=num_envs // len(learner_devices),
    replay_seq_length=steps_per_cycle,
    buffer_capacity=steps_per_cycle * 10,
    replay_batch_size=num_envs * 2,
    burn_in=40,
    learning_rate_schedule_name="cosine_onecycle_schedule",
    learning_rate_schedule_args=dict(
      transition_steps=steps_per_cycle * 2_500,
      # NOTE: more devices effectively means a larger batch size, so we
      # scale the learning rate up to train faster!
      peak_value=5e-5 * len(learner_devices),
    ),
    target_update_step_size=0.00,
    target_update_period=500,
    num_off_policy_optims_per_cycle=1,
  )
  agent = R2D2(env_info, config)
  loop_state = agent.new_state(networks, agent_key)
  metric_writer = TensorboardWriter(logdir="logs/Asterix")
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
  loop_state = train_loop.run(loop_state, num_cycles, steps_per_cycle)
