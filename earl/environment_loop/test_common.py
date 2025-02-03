import jax.numpy as jnp
from absl.testing import absltest

from earl.core import EnvStep, Image, Video
from earl.environment_loop._common import (
    CycleResult,
    multi_observe_cycle,
    pixel_obs_to_video_observe_cycle,
)


class TestCommon(absltest.TestCase):
    def test_pixel_obs_to_video_observe_cycle_single_video(self):
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
        self.assertEqual(len(metrics), 1)
        self.assertIsInstance(metrics["video"], Video)
        self.assertEqual(metrics["video"].data.shape, (10, 64, 64, 3))

    def test_pixel_obs_to_video_observe_cycle_multiple_videos(self):
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
        self.assertEqual(len(metrics), 2)
        self.assertIsInstance(metrics["video_0"], Video)
        self.assertIsInstance(metrics["video_1"], Video)
        self.assertEqual(metrics["video_0"].data.shape, (10, 64, 64, 3))
        self.assertEqual(metrics["video_1"].data.shape, (10, 64, 64, 3))

    def test_pixel_obs_to_video_observe_cycle_invalid_shape(self):
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
        with self.assertRaisesRegex(ValueError, "Expected trajectory.obs to have shape"):
            pixel_obs_to_video_observe_cycle(cycle_result)

    def test_multi_observe_cycle(self):
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
        self.assertEqual(len(metrics), 4)
        self.assertEqual(metrics["metric1"], 1)
        self.assertEqual(metrics["metric2"], 2)
        self.assertIsInstance(metrics["image1"], Image)
        self.assertIsInstance(metrics["video2"], Video)


if __name__ == "__main__":
    absltest.main()
