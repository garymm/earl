import dataclasses
import time
import typing
from collections.abc import Mapping
from functools import partial
from typing import Any, Generic

import equinox as eqx
import jax
import jax.numpy as jnp
from gymnax import EnvState
from gymnax.environments.environment import Environment as GymnaxEnvironment
from gymnax.environments.spaces import Discrete
from jaxtyping import PRNGKeyArray, PyTree, Scalar
from tqdm import tqdm

from research.earl.core import (
    Agent,
    AgentState,
    EnvStep,
    Metrics,
    _ExperienceState,
    _Networks,
    _OptState,
    _StepState,
)
from research.earl.logging.base import ArrayMetrics, CycleResult, MetricLogger, NoOpMetricLogger, ObserveCycle
from research.earl.logging.metric_key import MetricKey
from research.utils.eqx_filter import filter_scan  # TODO: remove deps on research

_ALL_METRIC_KEYS = {str(k) for k in MetricKey}


class ConflictingMetricError(Exception):
    pass


class _StepCarry(eqx.Module, Generic[_StepState]):
    env_step: EnvStep
    env_state: EnvState
    step_state: _StepState
    key: PRNGKeyArray
    total_reward: Scalar
    total_dones: Scalar
    """Number of steps in current episode for each environment."""
    episode_steps: jax.Array
    complete_episode_length_sum: Scalar
    complete_episode_count: Scalar
    action_counts: jax.Array


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


@dataclasses.dataclass(frozen=True)
class State(Generic[_Networks, _OptState, _ExperienceState, _StepState]):
    agent_state: AgentState[_Networks, _OptState, _ExperienceState, _StepState]
    env_state: EnvState | None = None
    env_step: EnvStep | None = None
    step_num: int = 0


State.__init__.__doc__ = """Initializes the State.

Args:
    agent_state: The agent state.
    env_state: The environment state. If None, the environment will be reset.
    env_step: The environment step. If None, the environment will be reset.
    step_num: The step number to start the metric logging at.
"""


@dataclasses.dataclass(frozen=True)
class Result(State[_Networks, _OptState, _ExperienceState, _StepState]):
    """A State but with all fields guaranteed to be not None."""

    # Note: defaults are just to please the type checker, since the base class has defaults.
    # We should not actually use the defaults.
    env_state: EnvState = EnvState(0)
    env_step: EnvStep = EnvStep(jnp.array(0), jnp.array(0), jnp.array(0), jnp.array(0))


def result_to_cycle_result(result: Result) -> CycleResult:
    """Converts a Result to a CycleResult.

    Result does not contain all the fields that CycleResult does, namely key, metrics, trajectory, and step_infos.
    This function fills in the missing fields with dummy values.
    """
    return CycleResult(
        agent_state=result.agent_state,
        env_state=result.env_state,
        env_step=result.env_step,
        key=jax.random.PRNGKey(0),
        metrics={},
        trajectory=EnvStep(jnp.array(0), jnp.array(0), jnp.array(0), jnp.array(0)),
        step_infos={},
    )


def _noop_observe_cycle(cycle_result: CycleResult) -> ArrayMetrics:
    return {}


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
        observe_cycle: ObserveCycle | None = None,
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
            observe_cycle: A function that takes a CycleResult representing a final environment state and a trajectory
                of length steps_per_cycle and runs any custom logic on it.
            inference: If False, agent.update_for_cycle() will not be called.
            assert_no_recompile: Whether to fail if the inner loop gets compiled more than once.
        """
        self._env = env
        sample_key, key = jax.random.split(key)
        sample_key = jax.random.split(sample_key, num_envs)
        self._action_space = self._env.action_space(env_params)
        self._example_action = jax.vmap(self._action_space.sample)(sample_key)
        self._env_reset = partial(self._env.reset, params=env_params)
        self._env_step = partial(self._env.step, params=env_params)
        self._agent = agent
        self._num_envs = num_envs
        self._key = key
        self._logger: MetricLogger = logger or NoOpMetricLogger()
        self._observe_cycle = observe_cycle or _noop_observe_cycle
        self._inference = inference
        _run_cycle_and_update = partial(self._run_cycle_and_update)
        if assert_no_recompile:
            _run_cycle_and_update = eqx.debug.assert_max_traces(_run_cycle_and_update, max_traces=1)
        self._run_cycle_and_update = eqx.filter_jit(_run_cycle_and_update, donate="warn")

    def reset_env(self) -> tuple[EnvState, EnvStep]:
        """Resets the environment.

        Should probably not be called by most users.
        Exposed so that callers can get the env_state and env_step to restore from a checkpoint.
        """
        env_key, self._key = jax.random.split(self._key, 2)
        env_keys = jax.random.split(env_key, self._num_envs)
        obs, env_state = jax.vmap(self._env_reset)(env_keys)

        env_step = EnvStep(
            new_episode=jnp.ones((self._num_envs,), dtype=jnp.bool),
            obs=obs,
            prev_action=jnp.zeros_like(self._example_action),
            reward=jnp.zeros((self._num_envs,)),
        )
        return env_state, env_step

    def run(
        self,
        state: State[_Networks, _OptState, _ExperienceState, _StepState]
        | AgentState[_Networks, _OptState, _ExperienceState, _StepState],
        num_cycles: int,
        steps_per_cycle: int,
        print_progress: bool = True,
    ) -> Result[_Networks, _OptState, _ExperienceState, _StepState]:
        """Runs the agent for num_cycles cycles, each with steps_per_cycle steps.

        Args:
            state: The initial state. Donated, meaning callers should not access it
                after calling this function. They can instead use the returned state.
                Callers can pass in an AgentState, which is equivalent to passing in a LoopState
                with the same agent_state and all other fields set to their default values.
                state.agent_state will be replaced with `equinox.nn.inference_mode(agent_state, value=inference)`
                before running, where `inference` is the value that was passed into
                GymnaxLoop.__init__().
            num_cycles: The number of cycles to run.
            steps_per_cycle: The number of steps to run in each cycle.
            print_progress: Whether to print progress to std out.
            step_num_metric_start: The value to start the step number metric at. This is useful
                when calling this function multiple times. Typically should be incremented by
                num_cycles * steps_per_cycle between calls.


        Returns:
            The final loop state and a dictionary of metrics.
            Each metric is a list of length num_cycles. All returned metrics are per-cycle.
        """
        if num_cycles <= 0:
            raise ValueError("num_cycles must be positive.")
        if steps_per_cycle <= 0:
            raise ValueError("steps_per_cycle must be positive.")

        if isinstance(state, AgentState):
            state = State(state)

        agent_state = eqx.nn.inference_mode(state.agent_state, value=self._inference)

        if state.env_state is None:
            assert state.env_step is None
            env_state, env_step = self.reset_env()
        else:
            env_state = state.env_state
            assert state.env_step is not None
            env_step = state.env_step

        step_num_metric_start = state.step_num
        del state

        cycles_iter = range(num_cycles)
        if print_progress:
            cycles_iter = tqdm(cycles_iter, desc="cycles", unit="cycle", leave=False)

        for cycle_num in cycles_iter:
            cycle_start = time.monotonic()
            # We saw some environments inconsistently returning weak_type, which triggers recompilation
            # when the previous cycle's output is passed back in to _run_cycle, so strip all weak_type.
            # The astype looks like a no-op but it strips the weak_type.
            env_step = jax.tree.map(lambda x: x.astype(x.dtype) if isinstance(x, jax.Array) else x, env_step)
            env_state = typing.cast(
                EnvState, jax.tree.map(lambda x: x.astype(x.dtype) if isinstance(x, jax.Array) else x, env_state)
            )
            cycle_result = self._run_cycle_and_update(agent_state, env_state, env_step, self._key, steps_per_cycle)

            cycle_metrics = self._observe_cycle(cycle_result)
            # Could potentially be very slow if there are lots
            # of metrics to compare
            _raise_if_metric_conflicts(cycle_metrics)

            agent_state, env_state, env_step, self._key = (
                cycle_result.agent_state,
                cycle_result.env_state,
                cycle_result.env_step,
                cycle_result.key,
            )

            # convert arrays to python types
            metrics: Metrics = {}
            metrics[MetricKey.DURATION_SEC] = time.monotonic() - cycle_start
            metrics[MetricKey.STEP_NUM] = step_num_metric_start + (cycle_num + 1) * steps_per_cycle

            action_counts = cycle_result.metrics.pop(MetricKey.ACTION_COUNTS)
            assert isinstance(action_counts, jax.Array)
            if action_counts.shape:  # will be empty tuple if not discrete action space
                for i in range(action_counts.shape[0]):
                    metrics[f"action_counts/{i}"] = int(action_counts[i])

            for k, v in cycle_result.metrics.items():
                assert v.shape == (), f"Expected scalar metric {k} to be scalar, got {v.shape}"
                metrics[k] = v.item()

            for k, v in cycle_metrics.items():
                if isinstance(v, jax.Array):
                    v = v.item()
                metrics[k] = v

            self._logger.write(metrics)

        return Result(agent_state, env_state, env_step, step_num_metric_start + num_cycles * steps_per_cycle)

    def _run_cycle_and_update(
        self,
        agent_state: AgentState[_Networks, _OptState, _ExperienceState, _StepState],
        env_state: EnvState,
        env_step: EnvStep,
        key: PRNGKeyArray,
        steps_per_cycle: int,
    ) -> CycleResult:
        @eqx.filter_grad(has_aux=True)
        def _run_cycle_and_loss_grad(
            nets_yes_grad: PyTree,
            nets_no_grad: PyTree,
            other_agent_state: AgentState[_Networks, _OptState, _ExperienceState, _StepState],
            env_state: EnvState,
            env_step: EnvStep,
            num_steps: int,
            key: PRNGKeyArray,
        ) -> tuple[Scalar, CycleResult]:
            agent_state = dataclasses.replace(other_agent_state, nets=eqx.combine(nets_yes_grad, nets_no_grad))
            cycle_result = self._run_cycle(agent_state, env_state, env_step, num_steps, key)
            experience_state = self._agent.update_experience(cycle_result.agent_state, cycle_result.trajectory)
            agent_state = dataclasses.replace(cycle_result.agent_state, experience=experience_state)
            loss, metrics = self._agent.loss(agent_state)
            _raise_if_metric_conflicts(metrics)
            # inside jit, return values are guaranteed to be arrays
            mutable_metrics: ArrayMetrics = typing.cast(ArrayMetrics, dict(metrics))
            mutable_metrics[MetricKey.LOSS] = loss
            mutable_metrics.update(cycle_result.metrics)
            return loss, dataclasses.replace(cycle_result, metrics=mutable_metrics, agent_state=agent_state)

        @eqx.filter_grad(has_aux=True)
        def _loss_for_cycle_grad(nets_yes_grad, nets_no_grad, other_agent_state) -> tuple[Scalar, ArrayMetrics]:
            agent_state = dataclasses.replace(other_agent_state, nets=eqx.combine(nets_yes_grad, nets_no_grad))
            loss, metrics = self._agent.loss(agent_state)
            _raise_if_metric_conflicts(metrics)
            # inside jit, return values are guaranteed to be arrays
            mutable_metrics: ArrayMetrics = typing.cast(ArrayMetrics, dict(metrics))
            mutable_metrics[MetricKey.LOSS] = loss
            return loss, mutable_metrics

        def _off_policy_update(
            agent_state: AgentState[_Networks, _OptState, _ExperienceState, _StepState], _
        ) -> tuple[AgentState[_Networks, _OptState, _ExperienceState, _StepState], ArrayMetrics]:
            nets_yes_grad, nets_no_grad = self._agent.partition_for_grad(agent_state.nets)
            grad, metrics = _loss_for_cycle_grad(
                nets_yes_grad, nets_no_grad, dataclasses.replace(agent_state, nets=None)
            )
            grad_means = _pytree_leaf_means(grad, "grad_mean")
            metrics.update(grad_means)
            agent_state = self._agent.optimize_from_grads(agent_state, grad)
            return agent_state, metrics

        if not self._inference and not self._agent.num_off_policy_optims_per_cycle():
            # On-policy update. Calculate the gradient through the entire cycle.
            nets_yes_grad, nets_no_grad = self._agent.partition_for_grad(agent_state.nets)
            nets_grad, cycle_result = _run_cycle_and_loss_grad(
                nets_yes_grad,
                nets_no_grad,
                dataclasses.replace(agent_state, nets=None),
                env_state,
                env_step,
                steps_per_cycle,
                key,
            )
            grad_means = _pytree_leaf_means(nets_grad, "grad_mean")
            cycle_result.metrics.update(grad_means)
            agent_state = self._agent.optimize_from_grads(cycle_result.agent_state, nets_grad)
        else:
            cycle_result = self._run_cycle(agent_state, env_state, env_step, steps_per_cycle, key)
            agent_state = cycle_result.agent_state
            if not self._inference:
                experience_state = self._agent.update_experience(cycle_result.agent_state, cycle_result.trajectory)
                agent_state = dataclasses.replace(agent_state, experience=experience_state)

        metrics = cycle_result.metrics
        if not self._inference and self._agent.num_off_policy_optims_per_cycle():
            agent_state, off_policy_metrics = filter_scan(
                _off_policy_update,
                init=agent_state,
                xs=None,
                length=self._agent.num_off_policy_optims_per_cycle(),
            )
            # Take mean of each metric.
            # This is potentially misleading, but not sure what else to do.
            metrics.update({k: jnp.mean(v) for k, v in off_policy_metrics.items()})

        return CycleResult(
            agent_state,
            cycle_result.env_state,
            cycle_result.env_step,
            cycle_result.key,
            metrics,
            cycle_result.trajectory,
            cycle_result.step_infos,
        )

    def _run_cycle(
        self,
        agent_state: AgentState[_Networks, _OptState, _ExperienceState, _StepState],
        env_state: EnvState,
        env_step: EnvStep,
        num_steps: int,
        key: PRNGKeyArray,
    ) -> CycleResult:
        """Runs self._agent and self._env for num_steps."""

        def scan_body(inp: _StepCarry[_StepState], _) -> tuple[_StepCarry[_StepState], tuple[EnvStep, dict[Any, Any]]]:
            agent_state_for_step = dataclasses.replace(agent_state, step=inp.step_state)

            agent_step = self._agent.step(agent_state_for_step, inp.env_step)
            action = agent_step.action
            if isinstance(self._action_space, Discrete):
                one_hot_actions = jax.nn.one_hot(action, self._action_space.n, dtype=inp.action_counts.dtype)
                action_counts = inp.action_counts + jnp.sum(one_hot_actions, axis=0)
            else:
                action_counts = inp.action_counts
            env_key, key = jax.random.split(inp.key)
            env_keys = jax.random.split(env_key, self._num_envs)
            obs, env_state, reward, done, info = jax.vmap(self._env_step)(env_keys, inp.env_state, action)
            next_timestep = EnvStep(done, obs, action, reward)

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
                    env_step=next_timestep,
                    env_state=env_state,
                    step_state=agent_step.state,
                    key=key,
                    total_reward=total_reward,
                    total_dones=total_dones,
                    episode_steps=episode_steps,
                    complete_episode_length_sum=episode_length_sum,
                    complete_episode_count=episode_count,
                    action_counts=action_counts,
                ),
                # NOTE: the very last env step in the last cycle is never returned in a trajectory.
                # I can't think of a clean way to do it, and losing a single step is unlikely to matter.
                (inp.env_step, info),
            )

        if isinstance(self._action_space, Discrete):
            action_counts = jnp.zeros(self._action_space.n, dtype=jnp.uint32)
        else:
            action_counts = jnp.array(0, dtype=jnp.uint32)

        final_carry, (trajectory, step_infos) = filter_scan(
            scan_body,
            init=_StepCarry(
                env_step=env_step,
                env_state=env_state,
                step_state=agent_state.step,
                key=key,
                total_reward=jnp.array(0.0),
                total_dones=jnp.array(0, dtype=jnp.uint32),
                episode_steps=jnp.zeros(self._num_envs, dtype=jnp.uint32),
                complete_episode_length_sum=jnp.array(0, dtype=jnp.uint32),
                complete_episode_count=jnp.array(0, dtype=jnp.uint32),
                action_counts=action_counts,
            ),
            xs=None,
            length=num_steps,
        )

        agent_state = dataclasses.replace(agent_state, step=final_carry.step_state)
        # mean across complete episodes
        complete_episode_length_mean = jnp.where(
            final_carry.complete_episode_count > 0,
            final_carry.complete_episode_length_sum / final_carry.complete_episode_count,
            0,
        )
        assert isinstance(complete_episode_length_mean, jax.Array)

        metrics = {}
        # mean across environments
        metrics[MetricKey.COMPLETE_EPISODE_LENGTH_MEAN] = jnp.mean(complete_episode_length_mean)
        metrics[MetricKey.NUM_ENVS_THAT_DID_NOT_COMPLETE] = jnp.sum(final_carry.complete_episode_count == 0)
        metrics[MetricKey.TOTAL_DONES] = final_carry.total_dones
        metrics[MetricKey.REWARD_SUM] = final_carry.total_reward
        metrics[MetricKey.REWARD_MEAN] = final_carry.total_reward / self._num_envs
        metrics[MetricKey.ACTION_COUNTS] = final_carry.action_counts

        # Somewhat arbitrary, but flashbax expects (num_envs, num_steps, ...)
        # so we make it easier on users by transposing here.
        def to_num_envs_first(x):
            if isinstance(x, jax.Array) and x.ndim > 1:
                return jnp.transpose(x, (1, 0, *range(2, x.ndim)))
            return x

        trajectory = jax.tree.map(to_num_envs_first, trajectory)

        return CycleResult(
            agent_state,
            final_carry.env_state,
            final_carry.env_step,
            final_carry.key,
            metrics,
            trajectory,
            step_infos,
        )
