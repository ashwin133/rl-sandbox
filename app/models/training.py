"""Pydantic models for RL training configuration and results."""

from typing import Literal

from pydantic import BaseModel


class TrainingConfig(BaseModel):
    """Configuration for a training run."""

    algorithm: Literal["reinforce", "dqn"]
    num_episodes: int = 1000
    gamma: float = 0.99
    learning_rate: float = 1e-3
    hidden_size: int = 128

    # DQN-specific
    batch_size: int = 64
    buffer_capacity: int = 10000
    target_update_freq: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995


class EpisodeMetrics(BaseModel):
    """Metrics from a single training episode."""

    episode: int
    total_reward: float
    steps: int
    loss: float | None


class TrainingResult(BaseModel):
    """Result of a complete training run."""

    episodes: list[EpisodeMetrics]
    config: TrainingConfig
    final_avg_reward: float
    solved: bool
