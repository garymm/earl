import jax.numpy as jnp
import pytest

from research.earl.core import EnvStep, Image, Video
from research.earl.environment_loop._common import (
    CycleResult,
    multi_observe_cycle,
    pixel_obs_to_video_observe_cycle,
)


def test_pixel_obs_to_video_observe_cycle():
    # Test with shape (T, H, W, C)
    obs = jnp.zeros((10, 64, 64, 3))
    cycle_result = CycleResult(
        agent_state=None,
        env_state=None,  # pyright: ignore[reportArgumentType]
        env_step=None,  # pyright: ignore[reportArgumentType]
        key=None,  # pyright: ignore[reportArgumentType]
        metrics={},
        trajectory=EnvStep(
            new_episode=jnp.array(0),
            obs=obs,
            prev_action=jnp.array(0),
            reward=jnp.array(0),
        ),
        step_infos={},
    )
    metrics = pixel_obs_to_video_observe_cycle(cycle_result)
    assert len(metrics) == 1
    assert isinstance(metrics["video"], Video)
    assert metrics["video"].data.shape == (10, 64, 64, 3)

    # Test with shape (B, T, H, W, C)
    obs = jnp.zeros((2, 10, 64, 64, 3))
    cycle_result = CycleResult(
        agent_state=None,
        env_state=None,  # pyright: ignore[reportArgumentType]
        env_step=None,  # pyright: ignore[reportArgumentType]
        key=None,  # pyright: ignore[reportArgumentType]
        metrics={},
        trajectory=EnvStep(
            new_episode=jnp.array(0),
            obs=obs,
            prev_action=jnp.array(0),
            reward=jnp.array(0),
        ),
        step_infos={},
    )
    metrics = pixel_obs_to_video_observe_cycle(cycle_result)
    assert len(metrics) == 2
    assert isinstance(metrics["video_0"], Video)
    assert isinstance(metrics["video_1"], Video)
    assert metrics["video_0"].data.shape == (10, 64, 64, 3)
    assert metrics["video_1"].data.shape == (10, 64, 64, 3)

    # Test invalid shape
    obs = jnp.zeros((10,))
    cycle_result = CycleResult(
        agent_state=None,
        env_state=None,  # pyright: ignore[reportArgumentType]
        env_step=None,  # pyright: ignore[reportArgumentType]
        key=None,  # pyright: ignore[reportArgumentType]
        metrics={},
        trajectory=EnvStep(
            new_episode=jnp.array(0),
            obs=obs,
            prev_action=jnp.array(0),
            reward=jnp.array(0),
        ),
        step_infos={},
    )
    with pytest.raises(ValueError, match="Expected trajectory.obs to have shape"):
        pixel_obs_to_video_observe_cycle(cycle_result)


def test_multi_observe_cycle():
    def observer1(cycle_result):
        return {"metric1": 1, "image1": Image(jnp.zeros((64, 64, 3)))}

    def observer2(cycle_result):
        return {"metric2": 2, "video2": Video(jnp.zeros((10, 64, 64, 3)))}

    combined_observer = multi_observe_cycle([observer1, observer2])
    cycle_result = CycleResult(
        agent_state=None,
        env_state=None,  # pyright: ignore[reportArgumentType]
        env_step=None,  # pyright: ignore[reportArgumentType]
        key=None,  # pyright: ignore[reportArgumentType]
        metrics={},
        trajectory=None,  # pyright: ignore[reportArgumentType]
        step_infos={},
    )

    metrics = combined_observer(cycle_result)
    assert len(metrics) == 4
    assert metrics["metric1"] == 1
    assert metrics["metric2"] == 2
    assert isinstance(metrics["image1"], Image)
    assert isinstance(metrics["video2"], Video)
