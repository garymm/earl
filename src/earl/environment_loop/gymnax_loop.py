import collections
import time
import typing
from collections.abc import Mapping
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import optax
from gymnax import EnvState
from gymnax.environments.environment import Environment as GymnaxEnvironment
from gymnax.environments.spaces import Discrete
from jaxtyping import PRNGKeyArray, PyTree, Scalar
from tqdm import tqdm

from research.earl.core import Agent, AgentState, EnvTimestep, ObserveTrajectory
from research.earl.logging.base import MetricLogger
from research.earl.logging.metric_key import MetricKey
from research.utils.eqx_filter import filter_scan  # TODO: remove deps on research

_ALL_METRIC_KEYS = {str(k) for k in MetricKey}


class ConflictingMetricError(Exception):
    pass


@jdc.pytree_dataclass()
class _StepCarry:
    env_timestep: EnvTimestep
    env_state: EnvState
    agent_step_state: Any
    key: PRNGKeyArray
    total_reward: Scalar
    total_dones: Scalar
    """Number of steps in current episode for each environment."""
    episode_steps: jnp.ndarray
    complete_episode_length_sum: Scalar
    complete_episode_count: Scalar
    action_counts: jnp.ndarray


_ArrayMetrics = dict[str, jnp.ndarray]


@jdc.pytree_dataclass()
class _CycleResult:
    agent_state: PyTree
    env_state: EnvState
    env_timestep: EnvTimestep
    key: PRNGKeyArray
    metrics: _ArrayMetrics
    trajectory: EnvTimestep | None = None
    step_infos: dict[Any, Any] | None = None


def _raise_if_metric_conflicts(metrics: Mapping):
    conflicting_keys = [k for k in metrics if k in _ALL_METRIC_KEYS]
    if not conflicting_keys:
        return
    raise ConflictingMetricError(
        "The following metrics conflict with Earl's default MetricKey: " + ", ".join(conflicting_keys)
    )


def _pytree_leaf_means(pytree: PyTree, prefix: str) -> dict[str, jax.Array]:
    """Returns a dict with key = path to the array in pytree, val = mean of array."""

    def traverse(obj, current_path=""):
        if isinstance(obj, jax.Array):
            return {f"{prefix}/{current_path}": jnp.mean(obj)}

        if isinstance(obj, tuple | list):
            return {k: v for i, item in enumerate(obj) for k, v in traverse(item, f"{current_path}[{i}]").items()}

        if hasattr(obj, "__dict__"):
            return {
                k: v
                for attr, value in obj.__dict__.items()
                for k, v in traverse(value, f"{current_path}.{attr}" if current_path else attr).items()
            }

        return {}

    return traverse(pytree)


class GymnaxLoop:
    """Runs an Agent in a Gymnax environment.

    Runs an agent and a Gymnax environment for a certain number of cycles.
    Each cycle is some (caller-specified) number of environment steps. It supports three modes:
    * training=False: just environment steps, no agent updates. Useful for evaluation.
    * training=True, agent.num_off_policy_updates_per_cycle() == 0: one on-policy update per cycle.
      The entire cycle of environment interaction is used to calculate the gradient.
    * training=True, agent.num_off_policy_updates_per_cycle() > 0: the specified number
      of off-policy updates per cycle. The gradient is calculated only for the Agent.loss() call,
      not the interaction with the environment.
    """

    def __init__(
        self,
        env: GymnaxEnvironment,
        env_params: Any,
        agent: Agent,
        num_envs: int,
        key: PRNGKeyArray,
        logger: MetricLogger | None = None,
        inference: bool = False,
        assert_no_recompile: bool = True,
    ):
        """Initializes the GymnaxLoop.

        Args:
            env: The environment.
            env_params: The environment's parameters.
            agent: The agent.
            num_envs: The number of environments to run in parallel.
            num_off_policy_updates_per_cycle: The number of off-policy updates to perform in each cycle.
              If zero and training is True, there will be 1 on-policy update per cycle.
            key: The PRNG key. Must not be shared with the agent state, as this will cause jax buffer donation
                errors.
            logger: The logger to write metrics to.
            inference: If False, agent.update_for_cycle() will not be called.
            assert_no_recompile: Whether to fail if the inner loop gets compiled more than once.
        """
        self._env = env
        sample_key, key = jax.random.split(key)
        sample_key = jax.random.split(sample_key, num_envs)
        self._action_space = self._env.action_space(env_params)
        self._example_action = jax.vmap(self._action_space.sample)(sample_key)
        self._example_obs = jax.vmap(self._env.observation_space(env_params).sample)(sample_key)
        self._env_reset = partial(self._env.reset, params=env_params)
        self._env_step = partial(self._env.step, params=env_params)
        self._agent = agent
        self._num_envs = num_envs
        self._key = key
        self._logger = logger
        self._inference = inference
        _run_cycle_and_update = partial(self._run_cycle_and_update)
        if assert_no_recompile:
            _run_cycle_and_update = eqx.debug.assert_max_traces(_run_cycle_and_update, max_traces=1)
        self._run_cycle_and_update = eqx.filter_jit(_run_cycle_and_update, donate="warn")

    def example_batched_obs(self) -> jnp.ndarray:
        return self._example_obs

    def run(
        self,
        agent_state: AgentState,
        num_cycles: int,
        steps_per_cycle: int,
        print_progress=True,
        step_num_metric_start: int = 0,
        observe_trajectory: ObserveTrajectory | None = None,
    ) -> tuple[AgentState, dict[str, list[float | int]]]:
        """Runs the agent for num_cycles cycles, each with steps_per_cycle steps.

        Args:
            agent_state: The initial agent state. Donated, meaning callers should not access it
                after calling this function. They can instead use the returned agent state.
                Will be replaced with `equinox.nn.inference_mode(agent_state, value=inference)`
                before running, where `inference` is the value that was passed into
                GymnaxLoop.__init__().
            num_cycles: The number of cycles to run.
            steps_per_cycle: The number of steps to run in each cycle.
            print_progress: Whether to print progress to std out.
            step_num_metric_start: The value to start the step number metric at. This is useful
                when calling this function multiple times. Typically should be incremented by
                num_cycles * steps_per_cycle between calls.
            observe_trajectory: A function that takes an EnvTimestep representing a trajectory of
                length steps_per_cycle and runs any custom logic on it.

        Returns:
            The final agent state and a dictionary of metrics.
            Each metric is a list of length num_cycles. All returned metrics are per-cycle.
        """
        if num_cycles <= 0:
            raise ValueError("num_cycles must be positive.")
        if steps_per_cycle <= 0:
            raise ValueError("steps_per_cycle must be positive.")

        agent_state = eqx.nn.inference_mode(agent_state, value=self._inference)

        all_metrics = collections.defaultdict(list)

        env_key, self._key = jax.random.split(self._key, 2)
        env_keys = jax.random.split(env_key, self._num_envs)
        obs, env_state = jax.vmap(self._env_reset)(env_keys)

        env_timestep = EnvTimestep(
            new_episode=jnp.ones((self._num_envs,), dtype=jnp.bool),
            obs=obs,
            prev_action=jnp.zeros_like(self._example_action),
            reward=jnp.zeros((self._num_envs,)),
        )

        logger = self._logger
        cycles_iter = range(num_cycles)
        if print_progress:
            cycles_iter = tqdm(cycles_iter, desc="cycles", unit="cycle", leave=False)

        keep_observations = observe_trajectory is not None

        if observe_trajectory is None:

            def noop(env_timesteps: EnvTimestep, step_infos: dict[Any, Any], step_num: int):
                return None

            observe_trajectory = noop  # So that I don't need to add an assertion in the inner loop for pyright

        for cycle_num in cycles_iter:
            cycle_start = time.monotonic()
            # We saw some environments inconsistently returning weak_type, which triggers recompilation
            # when the previous cycle's output is passed back in to _run_cycle, so strip all weak_type.
            # The astype looks like a no-op but it strips the weak_type.
            env_timestep = jax.tree.map(lambda x: x.astype(x.dtype) if isinstance(x, jax.Array) else x, env_timestep)
            env_state = jax.tree.map(lambda x: x.astype(x.dtype) if isinstance(x, jax.Array) else x, env_state)

            cycle_result = self._run_cycle_and_update(
                agent_state, env_state, env_timestep, self._key, steps_per_cycle, keep_observations
            )

            if (
                cycle_result.trajectory is not None and cycle_result.step_infos is not None
            ):  # Need this condition to appease pyright
                observe_trajectory(
                    cycle_result.trajectory,
                    cycle_result.step_infos,
                    step_num_metric_start + cycle_num * steps_per_cycle,
                )

            agent_state, env_state, env_timestep, self._key = (
                cycle_result.agent_state,
                cycle_result.env_state,
                cycle_result.env_timestep,
                cycle_result.key,
            )

            # convert arrays to python types
            py_metrics: dict[str, float | int] = {}
            py_metrics[MetricKey.DURATION_SEC] = time.monotonic() - cycle_start
            reward_mean = cycle_result.metrics[MetricKey.TOTAL_REWARD] / self._num_envs
            if MetricKey.REWARD_MEAN_SMOOTH not in all_metrics:
                reward_mean_smooth = reward_mean
            else:
                reward_mean_smooth = optax.incremental_update(
                    jnp.array(reward_mean), all_metrics[MetricKey.REWARD_MEAN_SMOOTH][-1], 0.01
                )
            assert isinstance(reward_mean_smooth, jnp.ndarray)
            py_metrics[MetricKey.REWARD_MEAN_SMOOTH] = float(reward_mean_smooth)

            py_metrics[MetricKey.STEP_NUM] = step_num_metric_start + (cycle_num + 1) * steps_per_cycle

            action_counts = cycle_result.metrics.pop(MetricKey.ACTION_COUNTS)
            assert isinstance(action_counts, jnp.ndarray)
            if action_counts.shape:  # will be empty tuple if not discrete action space
                for i in range(action_counts.shape[0]):
                    py_metrics[f"action_counts/{i}"] = int(action_counts[i])

            for k, v in cycle_result.metrics.items():
                assert v.shape == (), f"Expected scalar metric {k} to be scalar, got {v.shape}"
                if jnp.isdtype(v.dtype, "integral"):
                    v = int(v)
                else:
                    assert jnp.isdtype(v.dtype, "real floating")
                    v = float(v)
                py_metrics[k] = v

            for k, v in py_metrics.items():
                all_metrics[k].append(v)

            if logger:
                logger.write(py_metrics)

        return agent_state, all_metrics

    def _run_cycle_and_update(
        self,
        agent_state: AgentState,
        env_state: EnvState,
        env_timestep: EnvTimestep,
        key: PRNGKeyArray,
        steps_per_cycle: int,
        keep_observations: bool,
    ) -> _CycleResult:
        @eqx.filter_grad(has_aux=True)
        def _run_cycle_and_loss_grad(
            nets_yes_grad: PyTree,
            nets_no_grad: PyTree,
            other_agent_state: AgentState,
            env_state: EnvState,
            env_timestep: EnvTimestep,
            num_steps: int,
            key: PRNGKeyArray,
        ) -> tuple[Scalar, _CycleResult]:
            agent_state: AgentState = jdc.replace(other_agent_state, nets=eqx.combine(nets_yes_grad, nets_no_grad))
            cycle_result = self._run_cycle(agent_state, env_state, env_timestep, num_steps, key, keep_observations)
            loss, metrics = self._agent.loss(cycle_result.agent_state)
            _raise_if_metric_conflicts(metrics)
            # inside jit, return values are guaranteed to be arrays
            mutable_metrics: _ArrayMetrics = typing.cast(_ArrayMetrics, dict(metrics))
            mutable_metrics[MetricKey.LOSS] = loss
            mutable_metrics.update(cycle_result.metrics)
            return loss, jdc.replace(cycle_result, metrics=mutable_metrics)

        @eqx.filter_grad(has_aux=True)
        def _loss_for_cycle_grad(nets_yes_grad, nets_no_grad, other_agent_state) -> tuple[Scalar, _ArrayMetrics]:
            agent_state: AgentState = jdc.replace(other_agent_state, nets=eqx.combine(nets_yes_grad, nets_no_grad))
            loss, metrics = self._agent.loss(agent_state)
            _raise_if_metric_conflicts(metrics)
            # inside jit, return values are guaranteed to be arrays
            mutable_metrics: _ArrayMetrics = typing.cast(_ArrayMetrics, dict(metrics))
            mutable_metrics[MetricKey.LOSS] = loss
            return loss, mutable_metrics

        def _off_policy_update(agent_state: AgentState, _) -> tuple[AgentState, _ArrayMetrics]:
            nets_yes_grad, nets_no_grad = self._agent.partition_for_grad(agent_state.nets)
            grad, metrics = _loss_for_cycle_grad(nets_yes_grad, nets_no_grad, jdc.replace(agent_state, nets=None))
            grad_means = _pytree_leaf_means(grad, "grad_mean")
            metrics.update(grad_means)
            agent_state = self._agent.update_from_grads(agent_state, grad)
            return agent_state, metrics

        if not self._inference and not self._agent.num_off_policy_updates_per_cycle():
            # On-policy update. Calculate the gradient through the entire cycle.
            nets_yes_grad, nets_no_grad = self._agent.partition_for_grad(agent_state.nets)
            nets_grad, cycle_result = _run_cycle_and_loss_grad(
                nets_yes_grad,
                nets_no_grad,
                jdc.replace(agent_state, nets=None),
                env_state,
                env_timestep,
                steps_per_cycle,
                key,
            )
            grad_means = _pytree_leaf_means(nets_grad, "grad_mean")
            cycle_result.metrics.update(grad_means)
            agent_state = self._agent.update_from_grads(cycle_result.agent_state, nets_grad)
        else:
            cycle_result = self._run_cycle(
                agent_state, env_state, env_timestep, steps_per_cycle, key, keep_observations
            )
            agent_state = cycle_result.agent_state

        metrics = cycle_result.metrics
        if not self._inference and self._agent.num_off_policy_updates_per_cycle():
            agent_state, off_policy_metrics = filter_scan(
                _off_policy_update,
                init=agent_state,
                xs=None,
                length=self._agent.num_off_policy_updates_per_cycle(),
            )
            # Take mean of each metric.
            # This is potentially misleading, but not sure what else to do.
            metrics.update({k: jnp.mean(v) for k, v in off_policy_metrics.items()})

        return _CycleResult(
            agent_state,
            cycle_result.env_state,
            cycle_result.env_timestep,
            cycle_result.key,
            metrics,
            cycle_result.trajectory,
            cycle_result.step_infos,
        )

    def _run_cycle(
        self,
        agent_state: AgentState,
        env_state: EnvState,
        env_timestep: EnvTimestep,
        num_steps: int,
        key: PRNGKeyArray,
        keep_observations: bool,
    ) -> _CycleResult:
        """Runs self._agent and self._env for num_steps."""

        def scan_body(inp: _StepCarry, _) -> tuple[_StepCarry, tuple[EnvTimestep, dict[Any, Any]] | None]:
            agent_state_for_step = jdc.replace(agent_state, step=inp.agent_step_state)

            select_action_out = self._agent.select_action(agent_state_for_step, inp.env_timestep)
            action = select_action_out.action
            if isinstance(self._action_space, Discrete):
                one_hot_actions = jax.nn.one_hot(action, self._env.num_actions, dtype=inp.action_counts.dtype)
                action_counts = inp.action_counts + jnp.sum(one_hot_actions, axis=0)
            else:
                action_counts = inp.action_counts
            env_key, key = jax.random.split(inp.key)
            env_keys = jax.random.split(env_key, self._num_envs)
            obs, env_state, reward, done, info = jax.vmap(self._env_step)(env_keys, inp.env_state, action)
            next_timestep = EnvTimestep(done, obs, action, reward)

            episode_steps = inp.episode_steps + 1

            # Update episode statistics
            completed_episodes = next_timestep.new_episode
            episode_length_sum = inp.complete_episode_length_sum + jnp.sum(episode_steps * completed_episodes)
            episode_count = inp.complete_episode_count + jnp.sum(completed_episodes, dtype=jnp.uint32)

            # Reset steps for completed episodes
            episode_steps = jnp.where(completed_episodes, jnp.zeros_like(episode_steps), episode_steps)

            total_reward = inp.total_reward + jnp.sum(next_timestep.reward)
            total_dones = inp.total_dones + jnp.sum(next_timestep.new_episode, dtype=jnp.uint32)

            return (
                _StepCarry(
                    next_timestep,
                    env_state,
                    select_action_out.step_state,
                    key,
                    total_reward,
                    total_dones,
                    episode_steps,
                    episode_length_sum,
                    episode_count,
                    action_counts,
                ),
                (inp.env_timestep, info) if keep_observations else None,
            )

        if isinstance(self._action_space, Discrete):
            action_counts = jnp.zeros(self._env.num_actions, dtype=jnp.uint32)
        else:
            action_counts = jnp.array(0, dtype=jnp.uint32)

        final_carry, trajectory_and_info = filter_scan(
            scan_body,
            init=_StepCarry(
                env_timestep,
                env_state,
                agent_state.step,
                key,
                jnp.array(0.0),
                jnp.array(0, dtype=jnp.uint32),
                jnp.zeros(self._num_envs, dtype=jnp.uint32),
                jnp.array(0, dtype=jnp.uint32),
                jnp.array(0, dtype=jnp.uint32),
                action_counts,
            ),
            xs=None,
            length=num_steps,
        )

        agent_state = jdc.replace(agent_state, step=final_carry.agent_step_state)
        # mean across complete episodes
        complete_episode_length_mean = jnp.where(
            final_carry.complete_episode_count > 0,
            final_carry.complete_episode_length_sum / final_carry.complete_episode_count,
            0,
        )
        assert isinstance(complete_episode_length_mean, jnp.ndarray)

        metrics = {}
        # mean across environments
        metrics[MetricKey.COMPLETE_EPISODE_LENGTH_MEAN] = jnp.mean(complete_episode_length_mean)
        metrics[MetricKey.NUM_ENVS_THAT_DID_NOT_COMPLETE] = jnp.sum(final_carry.complete_episode_count == 0)
        metrics[MetricKey.TOTAL_DONES] = final_carry.total_dones
        metrics[MetricKey.TOTAL_REWARD] = final_carry.total_reward
        metrics[MetricKey.ACTION_COUNTS] = final_carry.action_counts

        trajectory = step_infos = None
        if keep_observations:
            assert isinstance(trajectory_and_info, tuple)
            trajectory, step_infos = trajectory_and_info
        return _CycleResult(
            agent_state,
            final_carry.env_state,
            final_carry.env_timestep,
            final_carry.key,
            metrics,
            trajectory,
            step_infos,
        )
