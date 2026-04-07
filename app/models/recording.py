"""Pydantic models for batch recording metadata (serialised to disk)."""

from datetime import datetime

from pydantic import BaseModel


class EpisodeRecording(BaseModel):
    """Metadata for a single captured episode."""

    episode_index: int
    policy_name: str
    timestamp: datetime
    timesteps_reached: int
    sim_speed_multiplier: float


class BatchRecording(BaseModel):
    """Metadata for a batch of episodes."""

    no_episodes: int  # total episodes in the batch (including non-captured)
    replay_freq: int
    episodes: list[EpisodeRecording]  # only the captured episodes
