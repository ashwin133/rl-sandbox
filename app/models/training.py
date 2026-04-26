"""Pydantic models for RL training configuration and results."""

from pydantic import BaseModel


class TrainingConfig(BaseModel):
    """Configuration for a training run."""

    num_episodes: int = 1000
    gamma: float = 0.99
    learning_rate: float = 1e-3
    hidden_size: int = 128


class EpisodeMetrics(BaseModel):
    """Metrics from a single training episode."""

    episode: int
    total_reward: float
    steps: int
    loss: float | None
    entropy: float | None = None


class EvalResult(BaseModel):
    """Aggregate result of a deterministic evaluation pass."""

    episode: int
    mean_reward: float
    std_reward: float
    n_episodes: int


class TrainingResult(BaseModel):
    """Result of a complete training run."""

    episodes: list[EpisodeMetrics]
    config: TrainingConfig
    final_avg_reward: float
    solved: bool
    evaluations: list[EvalResult] = []
