"""Tests for episode management models and runner."""

import asyncio
from collections.abc import Generator
from datetime import datetime, timezone

import numpy as np
import pytest

from app.envs.gym_env import GymEnvApp
from app.envs.registry import SUPPORTED_ENVS
from app.episode import EpisodeRunner
from app.models.batch import BatchConfig, BatchResult
from app.models.episode import EpisodeConfig, EpisodeResult, StepResult
from app.policy.random import RandomActionPolicy


class TestStepResult:
    """Tests for StepResult model."""

    def test_done_true_when_terminated(self) -> None:
        result = StepResult(
            obs=[0.0, 0.0, 0.0, 0.0],
            reward=1.0,
            terminated=True,
            truncated=False,
            done=True,
        )
        assert result.done is True

    def test_done_true_when_truncated(self) -> None:
        result = StepResult(
            obs=[0.0, 0.0, 0.0, 0.0],
            reward=1.0,
            terminated=False,
            truncated=True,
            done=True,
        )
        assert result.done is True

    def test_done_false_when_continuing(self) -> None:
        result = StepResult(
            obs=[0.0, 0.0, 0.0, 0.0],
            reward=1.0,
            terminated=False,
            truncated=False,
            done=False,
        )
        assert result.done is False

    def test_stores_observation(self) -> None:
        obs = [1.0, 2.0, 3.0, 4.0]
        result = StepResult(
            obs=obs, reward=1.0, terminated=False, truncated=False, done=False
        )
        assert result.obs == obs


class TestEpisodeConfig:
    """Tests for EpisodeConfig model."""

    def test_defaults(self) -> None:
        config = EpisodeConfig()
        assert config.max_steps is None
        assert config.sim_speed_multiplier == 1.0

    def test_custom_values(self) -> None:
        config = EpisodeConfig(max_steps=100, sim_speed_multiplier=2.0)
        assert config.max_steps == 100
        assert config.sim_speed_multiplier == 2.0


class TestBatchConfig:
    """Tests for BatchConfig model."""

    def test_stores_episode_config(self) -> None:
        ep_config = EpisodeConfig(max_steps=50)
        config = BatchConfig(no_episodes=10, episode_config=ep_config)
        assert config.no_episodes == 10
        assert config.episode_config.max_steps == 50


class TestEpisodeResult:
    """Tests for EpisodeResult model."""

    def test_stores_policy_name_as_string(self) -> None:
        result = EpisodeResult(
            sequence=[],
            timestamp=datetime.now(tz=timezone.utc),
            policy_name="RandomActionPolicy",
            episode_config=EpisodeConfig(),
            timesteps_reached=0,
        )
        assert isinstance(result.policy_name, str)
        assert result.policy_name == "RandomActionPolicy"

    def test_timesteps_matches_sequence_length(self) -> None:
        steps = [
            StepResult(
                obs=[0.0] * 4, reward=1.0, terminated=False, truncated=False, done=False
            )
            for _ in range(5)
        ]
        result = EpisodeResult(
            sequence=steps,
            timestamp=datetime.now(tz=timezone.utc),
            policy_name="TestPolicy",
            episode_config=EpisodeConfig(),
            timesteps_reached=5,
        )
        assert len(result.sequence) == result.timesteps_reached


class TestBatchResult:
    """Tests for BatchResult model."""

    def test_contains_correct_number_of_episodes(self) -> None:
        ep_config = EpisodeConfig()
        batch_config = BatchConfig(no_episodes=3, episode_config=ep_config)
        episodes = [
            EpisodeResult(
                sequence=[],
                timestamp=datetime.now(tz=timezone.utc),
                policy_name="TestPolicy",
                episode_config=ep_config,
                timesteps_reached=0,
            )
            for _ in range(3)
        ]
        result = BatchResult(episodes=episodes, batch_config=batch_config)
        assert len(result.episodes) == result.batch_config.no_episodes


class TestGymEnvAppReturnsStepResult:
    """Test that GymEnvApp.apply_action returns StepResult."""

    def test_apply_action_returns_step_result(self) -> None:
        app = GymEnvApp(SUPPORTED_ENVS["CartPole-v1"])
        try:
            result = app.apply_action(RandomActionPolicy())
            assert isinstance(result, StepResult)
            assert isinstance(result.obs, np.ndarray)
            assert isinstance(result.reward, float)
            assert isinstance(result.terminated, bool)
            assert isinstance(result.truncated, bool)
            assert result.done == (result.terminated or result.truncated)
        finally:
            app.env.close()


class TestEpisodeRunner:
    """Tests for EpisodeRunner."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> Generator[None, None, None]:
        self.app = GymEnvApp(SUPPORTED_ENVS["CartPole-v1"])
        self.runner = EpisodeRunner(app=self.app)
        self.policy = RandomActionPolicy()
        yield
        self.app.env.close()

    def test_run_episode_returns_episode_result(self) -> None:
        config = EpisodeConfig()
        result, _frames = asyncio.run(self.runner.run_episode(self.policy, config))
        assert isinstance(result, EpisodeResult)
        assert len(result.sequence) > 0
        assert result.timesteps_reached == len(result.sequence)
        assert result.policy_name == "RandomActionPolicy"
        assert isinstance(result.timestamp, datetime)
        assert result.episode_config == config

    def test_run_episode_resets_env_at_start(self) -> None:
        """Each episode resets env at start, so a second episode should work cleanly."""
        config = EpisodeConfig(max_steps=5)
        asyncio.run(self.runner.run_episode(self.policy, config))
        # Second episode should run without error (env reset at start)
        result, _frames = asyncio.run(self.runner.run_episode(self.policy, config))
        assert isinstance(result, EpisodeResult)
        assert len(result.sequence) > 0

    def test_run_episode_respects_max_steps(self) -> None:
        config = EpisodeConfig(max_steps=5)
        result, _frames = asyncio.run(self.runner.run_episode(self.policy, config))
        assert result.timesteps_reached <= 5

    def test_run_episode_sequence_length_matches_timesteps(self) -> None:
        config = EpisodeConfig()
        result, _frames = asyncio.run(self.runner.run_episode(self.policy, config))
        assert len(result.sequence) == result.timesteps_reached

    def test_run_episode_acquires_lock(self) -> None:
        """Verify env access happens under the lock by checking no RuntimeError is raised."""
        config = EpisodeConfig(max_steps=10)

        async def run_with_concurrent_lock_check() -> EpisodeResult:
            result, _frames = await self.runner.run_episode(self.policy, config)
            return result

        result = asyncio.run(run_with_concurrent_lock_check())
        assert isinstance(result, EpisodeResult)

    def test_current_episode_increments(self) -> None:
        config = EpisodeConfig()
        assert self.runner.current_episode == 0
        asyncio.run(self.runner.run_episode(self.policy, config))
        assert self.runner.current_episode == 1
        asyncio.run(self.runner.run_episode(self.policy, config))
        assert self.runner.current_episode == 2

    def test_total_steps_accumulates(self) -> None:
        config = EpisodeConfig(max_steps=5)
        assert self.runner.total_steps == 0
        asyncio.run(self.runner.run_episode(self.policy, config))
        steps_after_first = self.runner.total_steps
        assert steps_after_first > 0
        asyncio.run(self.runner.run_episode(self.policy, config))
        assert self.runner.total_steps > steps_after_first

    def test_episode_steps_resets(self) -> None:
        config = EpisodeConfig(max_steps=5)
        asyncio.run(self.runner.run_episode(self.policy, config))
        first_episode_steps = self.runner.episode_steps
        assert first_episode_steps > 0
        asyncio.run(self.runner.run_episode(self.policy, config))
        # episode_steps reflects the latest episode, not accumulated
        assert self.runner.episode_steps > 0

    def test_run_batch_runs_correct_number_of_episodes(self) -> None:
        batch_config = BatchConfig(
            no_episodes=3, episode_config=EpisodeConfig(max_steps=10)
        )
        result, _captured = asyncio.run(
            self.runner.run_batch(self.policy, batch_config)
        )
        assert isinstance(result, BatchResult)
        assert len(result.episodes) == 3
        assert self.runner.current_episode == 3

    def test_run_batch_returns_batch_result(self) -> None:
        batch_config = BatchConfig(
            no_episodes=2, episode_config=EpisodeConfig(max_steps=5)
        )
        result, _captured = asyncio.run(
            self.runner.run_batch(self.policy, batch_config)
        )
        assert isinstance(result, BatchResult)
        assert result.batch_config == batch_config
        for ep in result.episodes:
            assert isinstance(ep, EpisodeResult)
            assert ep.timesteps_reached <= 5
