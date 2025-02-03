import dataclasses
import logging
import time
import typing

import gymnasium.core
import gymnax
import gymnax.environments.spaces
import jax
import numpy as np
from absl.testing import absltest, parameterized
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from gymnax.environments.spaces import Discrete
from jax_loop_utils.metric_writers.memory_writer import MemoryWriter

from earl.agents.random_agent.random_agent import RandomAgent
from earl.core import ConflictingMetricError, Metrics, env_info_from_gymnasium
from earl.environment_loop import CycleResult
from earl.environment_loop.gymnasium_loop import GymnasiumLoop
from earl.metric_key import MetricKey
from earl.utils.prng import keygen


class TestGymnasiumLoop(parameterized.TestCase):
    @parameterized.parameters(
        {"inference": True, "num_off_policy_updates": 0},
        {"inference": False, "num_off_policy_updates": 0},
        {"inference": False, "num_off_policy_updates": 2},
    )
    # setting default device speeds things up a little, but running without cuda enabled jaxlib is even faster
    @jax.default_device(jax.devices("cpu")[0])
    def test_gymnasium_loop(self, inference: bool, num_off_policy_updates: int):
        num_envs = 2
        env = CartPoleEnv()
        env_info = env_info_from_gymnasium(env, num_envs)
        networks = None
        key_gen = keygen(jax.random.PRNGKey(0))
        agent = RandomAgent(env_info.action_space.sample, num_off_policy_updates)
        metric_writer = MemoryWriter()
        if not inference and not num_off_policy_updates:
            with self.assertRaisesRegex(ValueError, "On-policy training is not supported in GymnasiumLoop."):
                loop = GymnasiumLoop(
                    env, agent, num_envs, next(key_gen), metric_writer=metric_writer, inference=inference
                )
            return

        loop = GymnasiumLoop(
            env,
            agent,
            num_envs,
            next(key_gen),
            metric_writer=metric_writer,
            inference=inference,
            devices=jax.devices("cpu")[:1],
        )
        num_cycles = 2
        steps_per_cycle = 10
        agent_state = agent.new_state(networks, env_info, next(key_gen))
        result = loop.run(agent_state, num_cycles, steps_per_cycle)
        del agent_state
        self.assertEqual(result.agent_state.step.t, num_cycles * steps_per_cycle)
        metrics = metric_writer.scalars
        self.assertEqual(len(metrics), num_cycles)
        first_step_num, last_step_num = None, None
        for step_num, metrics_for_step in metrics.items():
            if first_step_num is None:
                first_step_num = step_num
            last_step_num = step_num
            self.assertIn(MetricKey.DURATION_SEC, metrics_for_step)
            self.assertIn(MetricKey.REWARD_SUM, metrics_for_step)
            self.assertIn(MetricKey.REWARD_MEAN, metrics_for_step)
            self.assertIn(MetricKey.TOTAL_DONES, metrics_for_step)
            action_count_sum = 0
            self.assertIsInstance(env_info.action_space, Discrete)
            for i in range(env_info.action_space.n):
                action_count_i = metrics_for_step[f"action_counts/{i}"]
                action_count_sum += action_count_i
            self.assertGreater(action_count_sum, 0)
            if not inference:
                self.assertIn(MetricKey.LOSS, metrics_for_step)
                self.assertIn(agent._prng_metric_key, metrics_for_step)
        if inference:
            self.assertEqual(result.agent_state.opt.opt_count, 0)
        else:
            self.assertIsNotNone(first_step_num)
            self.assertIsNotNone(last_step_num)
            self.assertNotEqual(
                metrics[first_step_num][agent._prng_metric_key], metrics[last_step_num][agent._prng_metric_key]
            )
            expected_opt_count = num_cycles * (num_off_policy_updates or 1)
            self.assertEqual(result.agent_state.opt.opt_count, expected_opt_count)

        self.assertIsInstance(env_info.action_space, Discrete)
        self.assertGreater(env_info.action_space.n, 0)
        self.assertFalse(loop._env.closed)
        loop.close()
        self.assertTrue(loop._env.closed)

    def test_bad_args(self):
        num_envs = 2
        env = CartPoleEnv()
        env_info = env_info_from_gymnasium(env, num_envs)
        agent = RandomAgent(env_info.action_space.sample, 0)
        metric_writer = MemoryWriter()
        loop = GymnasiumLoop(env, agent, num_envs, jax.random.PRNGKey(0), metric_writer=metric_writer, inference=True)
        agent_state = agent.new_state(None, env_info, jax.random.PRNGKey(0))
        with self.assertRaisesRegex(ValueError, "num_cycles"):
            loop.run(agent_state, 0, 10)
        with self.assertRaisesRegex(ValueError, "steps_per_cycle"):
            loop.run(agent_state, 10, 0)

    def test_bad_metric_key(self):
        networks = None
        num_envs = 2
        env = CartPoleEnv()
        env_info = env_info_from_gymnasium(env, num_envs)
        key_gen = keygen(jax.random.PRNGKey(0))
        # make the agent return a metric with a key that conflicts with a built-in metric.
        agent = RandomAgent(env_info.action_space.sample, 1)
        agent = dataclasses.replace(agent, _prng_metric_key=MetricKey.DURATION_SEC)

        metric_writer = MemoryWriter()
        loop = GymnasiumLoop(env, agent, num_envs, next(key_gen), metric_writer=metric_writer)
        num_cycles = 1
        steps_per_cycle = 1
        agent_state = agent.new_state(networks, env_info, jax.random.PRNGKey(0))
        with self.assertRaises(ConflictingMetricError):
            loop.run(agent_state, num_cycles, steps_per_cycle)

    def test_continuous_action_space(self):
        num_envs = 2
        env = PendulumEnv()
        env_info = env_info_from_gymnasium(env, num_envs)
        networks = None
        key_gen = keygen(jax.random.PRNGKey(0))
        action_space = env_info.action_space
        self.assertIsInstance(action_space, gymnax.environments.spaces.Box)
        self.assertIsInstance(action_space.low, jax.Array)
        self.assertIsInstance(action_space.high, jax.Array)
        agent = RandomAgent(action_space.sample, 0)
        metric_writer = MemoryWriter()
        loop = GymnasiumLoop(env, agent, num_envs, next(key_gen), metric_writer=metric_writer, inference=True)
        num_cycles = 1
        steps_per_cycle = 1
        agent_state = agent.new_state(networks, env_info, jax.random.PRNGKey(0))
        loop.run(agent_state, num_cycles, steps_per_cycle)
        for _, v in metric_writer.scalars.items():
            for k in v:
                self.assertFalse(k.startswith("action_counts_"))

    def test_observe_cycle(self):
        num_envs = 2
        env = PendulumEnv()
        env_info = env_info_from_gymnasium(env, num_envs)
        networks = None
        key_gen = keygen(jax.random.PRNGKey(0))
        agent = RandomAgent(env_info.action_space.sample, 0)
        metric_writer = MemoryWriter()

        def observe_cycle(cycle_result: CycleResult) -> Metrics:
            self.assertEqual(cycle_result.trajectory.obs.shape[0], num_envs)
            self.assertEqual(cycle_result.trajectory.obs.shape[1], steps_per_cycle)
            return {"ran": True}

        loop = GymnasiumLoop(
            env,
            agent,
            num_envs,
            next(key_gen),
            metric_writer=metric_writer,
            inference=True,
            observe_cycle=observe_cycle,
        )
        num_cycles = 2
        steps_per_cycle = 3
        agent_state = agent.new_state(networks, env_info, jax.random.PRNGKey(0))

        loop.run(agent_state, num_cycles, steps_per_cycle)
        for _, v in metric_writer.scalars.items():
            self.assertTrue(v.get("ran", False))


class SleepEnv(gymnasium.core.Env[np.int64, np.int64]):
    def __init__(self, sleep_secs: float):
        self.sleep_secs = sleep_secs
        self.action_space = gymnasium.spaces.Discrete(2)
        self.observation_space = gymnasium.spaces.Discrete(1)

    def reset(
        self, *, seed: int | None = None, options: dict[str, typing.Any] | None = None
    ) -> tuple[np.int64, dict[str, typing.Any]]:
        super().reset(seed=seed, options=options)
        return np.int64(0), {}

    def step(self, action: np.int64) -> tuple[np.int64, typing.SupportsFloat, bool, bool, dict[str, typing.Any]]:
        time.sleep(self.sleep_secs)
        return np.int64(0), 0.0, False, False, {}


class TestInferenceUpdateDifferentDevices(absltest.TestCase):
    def test_inference_update_different_devices(self):
        # This test can run on two CPU devices using
        # XLA_FLAGS="--xla_force_host_platform_device_count=2"
        # but I can't figure out how to make that flag take effect
        # only for this test before the test starts when using pytest.
        cpu_devices = jax.devices("cpu")
        if len(cpu_devices) >= 2:
            devices = cpu_devices[:2]
        else:
            try:
                devices = (cpu_devices[0], jax.devices("gpu")[0])
            except RuntimeError:
                self.skipTest("requires at least 2 devices")

        num_envs = 2
        env = SleepEnv(sleep_secs=0.1)
        env_info = env_info_from_gymnasium(env, num_envs)
        key_gen = keygen(jax.random.PRNGKey(0))
        agent = RandomAgent(env_info.action_space.sample, 1)
        metric_writer = MemoryWriter()

        loop = GymnasiumLoop(env, agent, num_envs, next(key_gen), metric_writer=metric_writer, devices=devices)
        num_cycles = 2
        steps_per_cycle = 10
        agent_state = agent.new_state(None, env_info, jax.random.PRNGKey(0))
        # the default agent_state has nets=None.
        # We set it to an array to check for use-after-donation bugs.
        agent_state = dataclasses.replace(agent_state, nets=jax.numpy.ones((1,)))
        with self.assertLogs(level=logging.WARNING) as caplog:
            loop.run(agent_state, num_cycles, steps_per_cycle)
            self.assertIn("inference is much slower than update", caplog.output[0])


if __name__ == "__main__":
    absltest.main()
