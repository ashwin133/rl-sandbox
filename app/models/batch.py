"""Pydantic models for batch episode management."""

from pydantic import BaseModel

from app.models.episode import EpisodeConfig, EpisodeResult


class BatchConfig(BaseModel):
    """Configuration for a batch of episodes."""

    no_episodes: int
    episode_config: EpisodeConfig


class BatchResult(BaseModel):
    """Result of a batch of episode runs."""

    episodes: list[EpisodeResult]
    batch_config: BatchConfig
