"""Tests for training configuration and result models."""

from app.envs.registry import SUPPORTED_ENVS, EnvSpec
from app.models.training import (
    EpisodeMetrics,
    EvalResult,
    TrainingConfig,
    TrainingResult,
)

CARTPOLE_SPEC: EnvSpec = SUPPORTED_ENVS["CartPole-v1"]


class TestTrainingConfig:
    """Tests for TrainingConfig model."""

    def test_default_values(self) -> None:
        config: TrainingConfig = TrainingConfig()
        assert config.num_episodes == 1000
        assert config.gamma == 0.99
        assert config.learning_rate == 1e-3
        assert config.hidden_size == 128

    def test_custom_values(self) -> None:
        config: TrainingConfig = TrainingConfig(
            num_episodes=500,
            gamma=0.95,
            learning_rate=5e-4,
            hidden_size=64,
        )
        assert config.num_episodes == 500
        assert config.gamma == 0.95
        assert config.learning_rate == 5e-4
        assert config.hidden_size == 64


class TestEpisodeMetrics:
    """Tests for EpisodeMetrics model."""

    def test_constructs_correctly(self) -> None:
        metrics: EpisodeMetrics = EpisodeMetrics(
            episode=5, total_reward=200.0, steps=200, loss=0.5
        )
        assert metrics.episode == 5
        assert metrics.total_reward == 200.0
        assert metrics.steps == 200
        assert metrics.loss == 0.5

    def test_loss_can_be_none(self) -> None:
        metrics: EpisodeMetrics = EpisodeMetrics(
            episode=0, total_reward=10.0, steps=10, loss=None
        )
        assert metrics.loss is None


class TestEvalResult:
    def test_constructs_correctly(self) -> None:
        eval_result: EvalResult = EvalResult(
            episode=99, mean_reward=480.0, std_reward=12.5, n_episodes=5
        )
        assert eval_result.episode == 99
        assert eval_result.mean_reward == 480.0
        assert eval_result.std_reward == 12.5
        assert eval_result.n_episodes == 5


class TestTrainingResult:
    """Tests for TrainingResult model."""

    def test_constructs_correctly(self) -> None:
        config: TrainingConfig = TrainingConfig()
        episodes: list[EpisodeMetrics] = [
            EpisodeMetrics(
                episode=i, total_reward=float(i * 10), steps=i * 10, loss=0.1
            )
            for i in range(5)
        ]
        result: TrainingResult = TrainingResult(
            episodes=episodes,
            config=config,
            env_spec=CARTPOLE_SPEC,
            final_avg_reward=20.0,
            solved=False,
        )
        assert len(result.episodes) == 5
        assert result.final_avg_reward == 20.0
        assert result.solved is False
        assert result.env_spec.env_id == "CartPole-v1"

    def test_solved_flag(self) -> None:
        config: TrainingConfig = TrainingConfig()
        result: TrainingResult = TrainingResult(
            episodes=[],
            config=config,
            env_spec=CARTPOLE_SPEC,
            final_avg_reward=480.0,
            solved=True,
        )
        assert result.solved is True
