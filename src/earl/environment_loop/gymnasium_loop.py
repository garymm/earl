import copy
import dataclasses
import multiprocessing
import multiprocessing.context
import time
import typing

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.core import Env as GymnasiumEnv
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import Autoreset
from gymnax import EnvState
from gymnax.environments.spaces import Discrete
from jax_loop_utils.metric_writers.interface import MetricWriter
from jaxtyping import PRNGKeyArray, PyTree, Scalar
from tqdm import tqdm

from research.earl.core import (
    Agent,
    AgentState,
    EnvStep,
    _ExperienceState,
    _Networks,
    _OptState,
    _StepState,
    env_info_from_gymnasium,
)
from research.earl.environment_loop import (
    ArrayMetrics,
    CycleResult,
    ObserveCycle,
    Result,
    State,
    StepCarry,
    no_op_observe_cycle,
)
from research.earl.environment_loop._common import (
    extract_metrics,
    pytree_leaf_means,
    raise_if_metric_conflicts,
    to_num_envs_first,
)
from research.earl.metric_key import MetricKey
from research.utils.eqx_filter import filter_scan


# Cannot donate all because when stacking leaves for the trajectory
# because we still need the final carry to be on the GPU.
@eqx.filter_jit(donate="all")
def _stack_leaves(pytree_list: list[PyTree]) -> PyTree:
    pytree = jax.tree.map(lambda *leaves: jnp.stack(leaves), *pytree_list)
    pytree = jax.tree.map(to_num_envs_first, pytree)
    return pytree


@eqx.filter_jit
def _copy_pytree(pytree: PyTree) -> PyTree:
    return jax.tree.map(lambda x: x.copy(), pytree)


@eqx.filter_grad(has_aux=True)
def _loss_for_cycle_grad(nets_yes_grad, nets_no_grad, other_agent_state, agent: Agent) -> tuple[Scalar, ArrayMetrics]:
    # this is a free function so we don't have to pass self as first arg, since filter_grad
    # takes gradient with respect to the first arg.
    agent_state = dataclasses.replace(other_agent_state, nets=eqx.combine(nets_yes_grad, nets_no_grad))
    loss, metrics = agent.loss(agent_state)
    raise_if_metric_conflicts(metrics)
    # inside jit, return values are guaranteed to be arrays
    mutable_metrics: ArrayMetrics = typing.cast(ArrayMetrics, dict(metrics))
    mutable_metrics[MetricKey.LOSS] = loss
    return loss, mutable_metrics


class GymnasiumLoop:
    """Runs an Agent in a Gymnasium environment.

    Runs an agent and a Gymnasium environment for a certain number of cycles.
    Each cycle is some (caller-specified) number of environment steps. It supports three modes:
    * training=False: just environment steps, no agent updates. Useful for evaluation.
    * training=True, agent.num_off_policy_updates_per_cycle() > 0: the specified number
      of off-policy updates per cycle. The gradient is calculated only for the Agent.loss() call,
      not the interaction with the environment.
    * On-policy training is not supported (training=True, agent.num_off_policy_updates_per_cycle() = 0)
    """

    def __init__(
        self,
        env: GymnasiumEnv,
        agent: Agent,
        num_envs: int,
        key: PRNGKeyArray,
        metric_writer: MetricWriter,
        observe_cycle: ObserveCycle = no_op_observe_cycle,
        inference: bool = False,
        assert_no_recompile: bool = True,
        vectorization_mode: typing.Literal["sync", "async"] = "async",
    ):
        """Initializes the GymnasiumLoop.

        Args:
            env: The environment.
            agent: The agent.
            num_envs: The number of environments to run in parallel.
            key: The PRNG key.
            metric_writer: The metric writer to write metrics to.
            observe_cycle: A function that takes a CycleResult representing a final environment state and a trajectory
                of length steps_per_cycle and runs any custom logic on it.
            inference: If False, agent.update_for_cycle() will not be called.
            assert_no_recompile: Whether to fail if the inner loop gets compiled more than once.
            vectorization_mode: Whether to create a synchronous or asynchronous vectorized environment from the provided
                Gymnasium environment.
        """
        env = Autoreset(env)  # run() assumes autoreset.

        def _env_factory() -> GymnasiumEnv:
            return copy.deepcopy(env)

        if vectorization_mode == "sync":
            self._env = SyncVectorEnv([_env_factory for _ in range(num_envs)])
        elif vectorization_mode == "async":
            # Cannot use forkserver because it is unsafe with multithreaded
            # jax
            multiprocessing.set_start_method("spawn", force=True)
            self._env = AsyncVectorEnv([_env_factory for _ in range(num_envs)])

        sample_key, key = jax.random.split(key)
        sample_key = jax.random.split(sample_key, num_envs)
        env_info = env_info_from_gymnasium(env, num_envs)
        self._action_space = env_info.action_space
        self._example_action = jax.vmap(self._action_space.sample)(sample_key)
        self._agent = agent
        self._num_envs = num_envs
        self._key = key
        self._metric_writer: MetricWriter = metric_writer
        self._observe_cycle = observe_cycle
        self._inference = inference
        update = self._update
        if assert_no_recompile:
            update = eqx.debug.assert_max_traces(update, max_traces=1)
        self._update = eqx.filter_jit(update, donate="warn")

        if not self._inference and not self._agent.num_off_policy_optims_per_cycle():
            raise ValueError("On-policy training is not supported in GymnasiumLoop.")

    def reset_env(self) -> EnvStep:
        """Resets the environment.

        Should probably not be called by most users.
        Exposed so that callers can get the env_step to restore from a checkpoint.
        """
        obs, _info = self._env.reset()

        return EnvStep(
            new_episode=jnp.ones((self._num_envs,), dtype=jnp.bool),
            obs=jnp.array(obs),
            prev_action=jnp.zeros_like(self._example_action),
            reward=jnp.zeros((self._num_envs,)),
        )

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
                GymnasiumLoop.__init__().
            num_cycles: The number of cycles to run.
            steps_per_cycle: The number of steps to run in each cycle.
            print_progress: Whether to print progress to std out.


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

        env_step = self.reset_env()

        step_num_metric_start = state.step_num
        del state

        cycles_iter = range(num_cycles)
        if print_progress:
            cycles_iter = tqdm(cycles_iter, desc="cycles", unit="cycle", leave=False)

        env_state = typing.cast(EnvState, None)
        for cycle_num in cycles_iter:
            cycle_start = time.monotonic()
            cycle_result = self._run_cycle(agent_state, env_step, steps_per_cycle, self._key)
            agent_state = cycle_result.agent_state

            if not self._inference:
                experience_state = self._agent.update_experience(cycle_result.agent_state, cycle_result.trajectory)
                agent_state = dataclasses.replace(agent_state, experience=experience_state)

            cycle_result = self._update(agent_state, cycle_result)

            observe_cycle_metrics = self._observe_cycle(cycle_result)
            # Could potentially be very slow if there are lots
            # of metrics to compare
            raise_if_metric_conflicts(observe_cycle_metrics)

            agent_state, env_state, env_step, self._key = (
                cycle_result.agent_state,
                cycle_result.env_state,
                cycle_result.env_step,
                cycle_result.key,
            )

            metrics_by_type = extract_metrics(cycle_result, observe_cycle_metrics)
            metrics_by_type.scalar[MetricKey.DURATION_SEC] = time.monotonic() - cycle_start

            step_num = step_num_metric_start + (cycle_num + 1) * steps_per_cycle
            self._metric_writer.write_scalars(step_num, metrics_by_type.scalar)
            self._metric_writer.write_images(step_num, metrics_by_type.image)
            self._metric_writer.write_videos(step_num, metrics_by_type.video)

        return Result(agent_state, env_state, env_step, step_num_metric_start + num_cycles * steps_per_cycle)

    def _off_policy_update(
        self, agent_state: AgentState[_Networks, _OptState, _ExperienceState, _StepState], _
    ) -> tuple[AgentState[_Networks, _OptState, _ExperienceState, _StepState], ArrayMetrics]:
        nets_yes_grad, nets_no_grad = self._agent.partition_for_grad(agent_state.nets)
        grad, metrics = _loss_for_cycle_grad(
            nets_yes_grad, nets_no_grad, dataclasses.replace(agent_state, nets=None), self._agent
        )
        grad_means = pytree_leaf_means(grad, "grad_mean")
        metrics.update(grad_means)
        agent_state = self._agent.optimize_from_grads(agent_state, grad)
        return agent_state, metrics

    def _update(
        self,
        agent_state: AgentState[_Networks, _OptState, _ExperienceState, _StepState],
        cycle_result: CycleResult,
    ) -> CycleResult:
        metrics = cycle_result.metrics
        if not self._inference:
            agent_state, off_policy_metrics = filter_scan(
                self._off_policy_update,
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
        env_step: EnvStep,
        num_steps: int,
        key: PRNGKeyArray,
    ) -> CycleResult:
        """Runs self._agent and self._env for num_steps."""

        if isinstance(self._action_space, Discrete):
            action_counts = jnp.zeros(self._action_space.n, dtype=jnp.uint32)
        else:
            action_counts = jnp.array(0, dtype=jnp.uint32)

        inp = StepCarry(
            env_step=env_step,
            env_state=typing.cast(EnvState, None),
            step_state=agent_state.step,
            key=key,
            total_reward=jnp.array(0.0),
            total_dones=jnp.array(0, dtype=jnp.uint32),
            episode_steps=jnp.zeros(self._num_envs, dtype=jnp.uint32),
            complete_episode_length_sum=jnp.array(0, dtype=jnp.uint32),
            complete_episode_count=jnp.array(0, dtype=jnp.uint32),
            action_counts=action_counts,
        )
        final_carry = inp
        trajectory = []
        step_infos = []
        for _ in range(num_steps):
            agent_state_for_step = dataclasses.replace(agent_state, step=inp.step_state)
            agent_step = self._agent.step(agent_state_for_step, inp.env_step)
            action = agent_step.action
            if isinstance(self._action_space, Discrete):
                one_hot_actions = jax.nn.one_hot(action, self._action_space.n, dtype=inp.action_counts.dtype)
                action_counts = inp.action_counts + jnp.sum(one_hot_actions, axis=0)
            else:
                action_counts = inp.action_counts

            # If action is on GPU, np.array(action) will move it to CPU.
            obs, reward, done, trunc, info = self._env.step(np.array(action))

            # FYI: This is expensive, moves data to GPU if we use
            # a GPU backend.
            obs = jnp.array(obs)
            reward = jnp.array(reward)
            done = jnp.array(done | trunc)

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

            inp = StepCarry(
                env_step=next_timestep,
                env_state=typing.cast(EnvState, None),
                step_state=agent_step.state,
                key=key,
                total_reward=total_reward,
                total_dones=total_dones,
                episode_steps=episode_steps,
                complete_episode_length_sum=episode_length_sum,
                complete_episode_count=episode_count,
                action_counts=action_counts,
            )
            trajectory.append(inp.env_step)
            step_infos.append(info)

        final_carry = inp
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

        # Need to copy final_carry.env_step because _stack_leaves has buffer
        # donation and we still need the final carry.
        final_carry_env_step = _copy_pytree(final_carry.env_step)
        trajectory = _stack_leaves(trajectory)
        with jax.default_device(jax.devices("cpu")[0]):
            step_infos = _stack_leaves(step_infos)

        return CycleResult(
            agent_state,
            # not really the env state in any meaningful way, but we need to pass something
            # valid to make this code more similar to GymnaxLoop.
            EnvState(time=num_steps),
            final_carry_env_step,
            final_carry.key,
            metrics,
            trajectory,
            step_infos,
        )

    def close(self):
        self._env.close()
