import jax
import numpy as np
import optax
from gymnax.environments import CartPole
from jax_loop_utils.metric_writers.memory_writer import MemoryWriter

from research.earl.agents import simple_policy_gradient
from research.earl.core import env_info_from_gymnax
from research.earl.environment_loop.gymnax_loop import GymnaxLoop, MetricKey


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
    metric_writer = MemoryWriter()
    loop = GymnaxLoop(env, env.default_params, agent, num_envs, loop_key, metric_writer=metric_writer)
    env_info = env_info_from_gymnax(env, env.default_params, num_envs)
    agent_state = agent.new_state(networks, env_info, agent_key)
    agent_state = loop.run(agent_state, num_cycles, steps_per_cycle)
    assert agent_state
    metrics = metric_writer.scalars
    assert len(metrics) == num_cycles
    episode_lengths = np.array(
        [step_metrics[MetricKey.COMPLETE_EPISODE_LENGTH_MEAN] for step_metrics in metrics.values()]
    )
    # Due to auto-resets, the reward is always constant, but it's a survival task
    # so longer episodes are better.
    first_five_mean = np.mean(episode_lengths[:5])
    assert first_five_mean > 0
    last_five_mean = np.mean(episode_lengths[-5:])
    assert last_five_mean > 1.5 * first_five_mean, (episode_lengths[:5], episode_lengths[-5:])
