"""Tests for FileOps and recording models."""

import asyncio
from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from app.cartpole import CartPoleApp
from app.episode import EpisodeRunner
from app.file_ops import FileOps
from app.models.batch import BatchConfig
from app.models.episode import EpisodeConfig
from app.models.recording import BatchRecording, EpisodeRecording
from app.policy.random import RandomActionPolicy


class TestEpisodeRecording:
    """Tests for EpisodeRecording model."""

    def test_constructs_correctly(self) -> None:
        rec = EpisodeRecording(
            episode_index=5,
            policy_name="RandomActionPolicy",
            timestamp=datetime.now(tz=timezone.utc),
            timesteps_reached=42,
            sim_speed_multiplier=2.0,
        )
        assert rec.episode_index == 5
        assert rec.policy_name == "RandomActionPolicy"
        assert rec.timesteps_reached == 42
        assert rec.sim_speed_multiplier == 2.0

    def test_roundtrips_to_dict(self) -> None:
        rec = EpisodeRecording(
            episode_index=0,
            policy_name="TestPolicy",
            timestamp=datetime(2026, 4, 7, tzinfo=timezone.utc),
            timesteps_reached=10,
            sim_speed_multiplier=1.0,
        )
        data = rec.model_dump()
        restored = EpisodeRecording(**data)
        assert restored == rec


class TestBatchRecording:
    """Tests for BatchRecording model."""

    def test_stores_correct_counts(self) -> None:
        episodes = [
            EpisodeRecording(
                episode_index=i,
                policy_name="TestPolicy",
                timestamp=datetime.now(tz=timezone.utc),
                timesteps_reached=10,
                sim_speed_multiplier=1.0,
            )
            for i in [0, 5, 10]
        ]
        rec = BatchRecording(no_episodes=15, replay_freq=5, episodes=episodes)
        assert rec.no_episodes == 15
        assert rec.replay_freq == 5
        assert len(rec.episodes) == 3


class TestBatchConfigReplayFreq:
    """Tests for replay_freq on BatchConfig."""

    def test_defaults_to_one(self) -> None:
        config = BatchConfig(no_episodes=10, episode_config=EpisodeConfig())
        assert config.replay_freq == 1

    def test_accepts_custom_value(self) -> None:
        config = BatchConfig(
            no_episodes=10, episode_config=EpisodeConfig(), replay_freq=5
        )
        assert config.replay_freq == 5


class TestFileOpsSave:
    """Tests for FileOps.save."""

    def test_creates_npz_and_json(self, tmp_path: Path) -> None:
        frames: dict[int, list[np.ndarray]] = {
            0: [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)],
        }
        metadata = BatchRecording(
            no_episodes=10,
            replay_freq=5,
            episodes=[
                EpisodeRecording(
                    episode_index=0,
                    policy_name="TestPolicy",
                    timestamp=datetime.now(tz=timezone.utc),
                    timesteps_reached=3,
                    sim_speed_multiplier=1.0,
                )
            ],
        )
        FileOps.save(frames, metadata, tmp_path)
        assert (tmp_path / "frames.npz").exists()
        assert (tmp_path / "metadata.json").exists()

    def test_npz_contains_correct_keys(self, tmp_path: Path) -> None:
        frames: dict[int, list[np.ndarray]] = {
            0: [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(2)],
            5: [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(4)],
        }
        metadata = BatchRecording(
            no_episodes=10,
            replay_freq=5,
            episodes=[
                EpisodeRecording(
                    episode_index=i,
                    policy_name="TestPolicy",
                    timestamp=datetime.now(tz=timezone.utc),
                    timesteps_reached=n,
                    sim_speed_multiplier=1.0,
                )
                for i, n in [(0, 2), (5, 4)]
            ],
        )
        FileOps.save(frames, metadata, tmp_path)
        data = np.load(tmp_path / "frames.npz")
        assert "replay_episodes" in data
        np.testing.assert_array_equal(data["replay_episodes"], [0, 5])
        assert data["frames_0"].shape == (2, 10, 10, 3)
        assert data["frames_1"].shape == (4, 10, 10, 3)


class TestFileOpsRoundtrip:
    """Tests for FileOps save/load roundtrip."""

    def test_roundtrip_preserves_frames_and_metadata(self, tmp_path: Path) -> None:
        frame_a = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        frame_b = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        frames: dict[int, list[np.ndarray]] = {
            0: [frame_a, frame_b],
            3: [frame_a],
        }
        metadata = BatchRecording(
            no_episodes=6,
            replay_freq=3,
            episodes=[
                EpisodeRecording(
                    episode_index=0,
                    policy_name="PolicyA",
                    timestamp=datetime(2026, 4, 7, 12, 0, 0, tzinfo=timezone.utc),
                    timesteps_reached=2,
                    sim_speed_multiplier=2.0,
                ),
                EpisodeRecording(
                    episode_index=3,
                    policy_name="PolicyA",
                    timestamp=datetime(2026, 4, 7, 12, 0, 1, tzinfo=timezone.utc),
                    timesteps_reached=1,
                    sim_speed_multiplier=2.0,
                ),
            ],
        )
        FileOps.save(frames, metadata, tmp_path)
        replay = FileOps.load_batch_replay(tmp_path)

        assert replay.total_source_episodes == 6
        assert replay.replay_freq == 3
        assert len(replay.episodes) == 2

        # Check first episode frames
        np.testing.assert_array_equal(replay.episodes[0].frames[0], frame_a)
        np.testing.assert_array_equal(replay.episodes[0].frames[1], frame_b)
        assert replay.episodes[0].metadata.episode_index == 0
        assert replay.episodes[0].metadata.policy_name == "PolicyA"
        assert replay.episodes[0].metadata.sim_speed_multiplier == 2.0

        # Check second episode
        np.testing.assert_array_equal(replay.episodes[1].frames[0], frame_a)
        assert replay.episodes[1].metadata.episode_index == 3


class TestEpisodeRunnerFrameCapture:
    """Tests for frame capture in run_episode and run_batch."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> Generator[None, None, None]:
        self.app = CartPoleApp()
        self.runner = EpisodeRunner(app=self.app)
        self.policy = RandomActionPolicy()
        yield
        self.app.env.close()

    def test_run_episode_with_capture_returns_frames(self) -> None:
        config = EpisodeConfig(max_steps=5)
        result, frames = asyncio.run(
            self.runner.run_episode(self.policy, config, capture_frames=True)
        )
        assert frames is not None
        assert len(frames) > 0
        assert len(frames) == result.timesteps_reached
        # Each frame should be an RGB array
        assert frames[0].ndim == 3
        assert frames[0].shape[2] == 3

    def test_run_episode_without_capture_returns_none(self) -> None:
        config = EpisodeConfig(max_steps=5)
        result, frames = asyncio.run(
            self.runner.run_episode(self.policy, config, capture_frames=False)
        )
        assert frames is None
        assert isinstance(result.timesteps_reached, int)

    def test_run_episode_default_no_capture(self) -> None:
        config = EpisodeConfig(max_steps=5)
        result, frames = asyncio.run(self.runner.run_episode(self.policy, config))
        assert frames is None

    def test_run_batch_captures_by_replay_freq(self) -> None:
        batch_config = BatchConfig(
            no_episodes=10, episode_config=EpisodeConfig(max_steps=5), replay_freq=3
        )
        batch_result, captured = asyncio.run(
            self.runner.run_batch(self.policy, batch_config)
        )
        assert len(batch_result.episodes) == 10
        # replay_freq=3 on 10 episodes: indices 0, 3, 6, 9
        assert set(captured.keys()) == {0, 3, 6, 9}
        for idx, frames in captured.items():
            assert len(frames) > 0
            assert frames[0].ndim == 3

    def test_run_batch_replay_freq_1_captures_all(self) -> None:
        batch_config = BatchConfig(
            no_episodes=3, episode_config=EpisodeConfig(max_steps=5), replay_freq=1
        )
        batch_result, captured = asyncio.run(
            self.runner.run_batch(self.policy, batch_config)
        )
        assert set(captured.keys()) == {0, 1, 2}
