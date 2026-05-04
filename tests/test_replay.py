"""Tests for ReplayRunner and replay models."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rl_sandbox.envs.registry import SUPPORTED_ENVS, EnvSpec
from rl_sandbox.models.recording import EpisodeRecording
from rl_sandbox.models.replay import BatchReplay, EpisodeReplay
from rl_sandbox.replay import ReplayRunner

CARTPOLE_SPEC: EnvSpec = SUPPORTED_ENVS["CartPole-v1"]


class TestEpisodeReplay:
    """Tests for EpisodeReplay dataclass."""

    def test_holds_frames_and_metadata(self) -> None:
        frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)]
        metadata = EpisodeRecording(
            episode_index=0,
            policy_name="TestPolicy",
            timestamp=datetime.now(tz=timezone.utc),
            timesteps_reached=3,
            sim_speed_multiplier=1.0,
        )
        replay = EpisodeReplay(frames=frames, metadata=metadata)
        assert len(replay.frames) == 3
        assert replay.metadata.policy_name == "TestPolicy"


class TestBatchReplay:
    """Tests for BatchReplay dataclass."""

    def _make_episode(self, index: int, n_frames: int = 2) -> EpisodeReplay:
        return EpisodeReplay(
            frames=[np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(n_frames)],
            metadata=EpisodeRecording(
                episode_index=index,
                policy_name="TestPolicy",
                timestamp=datetime.now(tz=timezone.utc),
                timesteps_reached=n_frames,
                sim_speed_multiplier=1.0,
            ),
        )

    def test_holds_episodes_and_metadata(self) -> None:
        episodes = [self._make_episode(0), self._make_episode(5)]
        batch = BatchReplay(
            episodes=episodes,
            source_path="/tmp/test",
            total_source_episodes=10,
            replay_freq=5,
            env_spec=CARTPOLE_SPEC,
        )
        assert len(batch.episodes) == 2
        assert batch.total_source_episodes == 10
        assert batch.replay_freq == 5
        assert batch.env_spec.env_id == "CartPole-v1"


class TestReplayRunner:
    """Tests for ReplayRunner."""

    def _make_episode(
        self,
        index: int = 0,
        n_frames: int = 3,
        sim_speed: float = 1.0,
    ) -> EpisodeReplay:
        return EpisodeReplay(
            frames=[
                np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
                for _ in range(n_frames)
            ],
            metadata=EpisodeRecording(
                episode_index=index,
                policy_name="TestPolicy",
                timestamp=datetime.now(tz=timezone.utc),
                timesteps_reached=n_frames,
                sim_speed_multiplier=sim_speed,
            ),
        )

    def test_replay_episode_calls_display_frame_per_frame(self) -> None:
        mock_renderer = MagicMock()
        runner = ReplayRunner(renderer=mock_renderer)
        episode = self._make_episode(n_frames=5)

        asyncio.run(runner.replay_episode(episode, CARTPOLE_SPEC.physics_tick_s))

        mock_renderer.init_display.assert_called_once()
        assert mock_renderer.display_frame.call_count == 5

    def test_replay_batch_replays_all_episodes(self) -> None:
        mock_renderer = MagicMock()
        runner = ReplayRunner(renderer=mock_renderer)
        batch = BatchReplay(
            episodes=[self._make_episode(0, 2), self._make_episode(5, 3)],
            source_path="/tmp/test",
            total_source_episodes=10,
            replay_freq=5,
            env_spec=CARTPOLE_SPEC,
        )

        asyncio.run(runner.replay_batch(batch))

        # 2 episodes: init_display called once per episode
        assert mock_renderer.init_display.call_count == 2
        # total display_frame calls: 2 + 3 = 5
        assert mock_renderer.display_frame.call_count == 5

    def test_replay_episode_sleep_scales_with_speed(self) -> None:
        """Verify sleep duration = physics_tick / sim_speed_multiplier."""
        mock_renderer = MagicMock()
        runner = ReplayRunner(renderer=mock_renderer)
        episode = self._make_episode(n_frames=2, sim_speed=2.0)

        sleep_args: list[float] = []

        async def fake_sleep(duration: float) -> None:
            sleep_args.append(duration)

        async def run() -> None:
            with patch("rl_sandbox.replay.asyncio.sleep", side_effect=fake_sleep):
                await runner.replay_episode(episode, CARTPOLE_SPEC.physics_tick_s)

        asyncio.run(run())

        expected_tick = CARTPOLE_SPEC.physics_tick_s / 2.0
        assert len(sleep_args) == 2
        assert sleep_args[0] == pytest.approx(expected_tick)
        assert sleep_args[1] == pytest.approx(expected_tick)
