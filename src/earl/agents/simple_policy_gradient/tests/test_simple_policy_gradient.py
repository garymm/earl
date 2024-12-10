import jax
import numpy as np
import optax
from gymnax.environments import CartPole

from research.earl.agents import simple_policy_gradient
from research.earl.core import env_info_from_gymnax
from research.earl.environment_loop.gymnax_loop import GymnaxLoop, MetricKey
from research.earl.logging import MemoryLogger


def test_learns_cart_pole():
    env = CartPole()
    num_envs = 500
    steps_per_cycle = 80
    num_cycles = 35

    config = simple_policy_gradient.Config(max_step_state_history=steps_per_cycle, optimizer=optax.adam(5e-3))
    agent = simple_policy_gradient.SimplePolicyGradient(config)
    (input_shape,) = env.obs_shape

    nets_key, loop_key, agent_key = jax.random.split(jax.random.PRNGKey(0), 3)

    networks = simple_policy_gradient.make_networks([input_shape, 32, env.num_actions], nets_key)
    logger = MemoryLogger()
    loop = GymnaxLoop(env, env.default_params, agent, num_envs, loop_key, logger=logger)
    env_info = env_info_from_gymnax(env, env.default_params, num_envs)
    agent_state = agent.new_state(networks, env_info, agent_key)
    agent_state = loop.run(agent_state, num_cycles, steps_per_cycle)
    assert agent_state
    metrics = logger.metrics()
    assert len(metrics[MetricKey.COMPLETE_EPISODE_LENGTH_MEAN]) == num_cycles
    episode_length_arr = np.array(metrics[MetricKey.COMPLETE_EPISODE_LENGTH_MEAN])
    # Due to auto-resets, the reward is always constant, but it's a survival task
    # so longer episodes are better.
    first_five_mean = np.mean(episode_length_arr[:5])
    assert first_five_mean > 0
    last_five_mean = np.mean(episode_length_arr[-5:])
    assert last_five_mean > 1.5 * first_five_mean, (episode_length_arr[:5], episode_length_arr[-5:])
