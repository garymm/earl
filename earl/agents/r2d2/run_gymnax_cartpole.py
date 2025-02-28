import math
import os

import jax
import jax.numpy as jnp
from gymnax.environments.classic_control import CartPole
from gymnax.environments.spaces import Box, Discrete
from jax_loop_utils.metric_writers.async_writer import AsyncWriter
from jax_loop_utils.metric_writers.torch.tensorboard_writer import TensorboardWriter

import earl.agents.r2d2.networks as r2d2_networks
from earl.agents.r2d2.r2d2 import R2D2, ExplorationType, R2D2Config
from earl.core import env_info_from_gymnax
from earl.environment_loop.gymnax_loop import GymnaxLoop
from earl.experiments.run_experiment import _config_to_dict

if __name__ == "__main__":
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
  num_cycles = 10_000
  env_info = env_info_from_gymnax(env, env_params, num_envs)
  num_off_policy_optims_per_cycle = 1
  devices = jax.local_devices()
  config = R2D2Config(
    epsilon_greedy_schedule_args=dict(
      init_value=0.5, end_value=0.01, transition_steps=steps_per_cycle * num_cycles
    ),
    q_learning_n_steps=2,
    debug=False,
    buffer_capacity=steps_per_cycle * 5,
    num_envs_per_learner=num_envs,
    replay_seq_length=steps_per_cycle,
    burn_in=burn_in,
    value_rescaling_epsilon=0.0,
    num_off_policy_optims_per_cycle=num_off_policy_optims_per_cycle,
    gradient_clipping_max_delta=1e-1,
    learning_rate_schedule_name="cosine_onecycle_schedule",
    learning_rate_schedule_args=dict(
      transition_steps=steps_per_cycle * num_cycles // 2,
      peak_value=1e-4 * len(devices),
    ),
    target_update_step_size=0.001,
    exploration_type=ExplorationType.STICKY,
  )
  agent = R2D2(env_info, config)

  metric_writer = AsyncWriter(
    TensorboardWriter(logdir=os.path.join("logs", "gymnax_cartpole")),
    num_workers=None,
  )
  config_dict = _config_to_dict(config)
  metric_writer.write_hparams(config_dict)
  loop = GymnaxLoop(
    env,
    env_params,
    agent,
    num_envs,
    loop_key,
    metric_writer=metric_writer,
    devices=devices,
  )
  agent_state = agent.new_state(networks, agent_key)

  loop_state = loop.run(agent_state, num_cycles, steps_per_cycle)
  del agent_state
  metric_writer.close()
