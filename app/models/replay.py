"""In-memory models for episode replay (not serialised)."""

from dataclasses import dataclass

import numpy as np

from app.models.recording import EpisodeRecording


@dataclass
class EpisodeReplay:
    """Everything needed to replay one episode."""

    frames: list[np.ndarray]
    metadata: EpisodeRecording


@dataclass
class BatchReplay:
    """Everything needed to replay a set of saved episodes."""

    episodes: list[EpisodeReplay]
    source_path: str
    total_source_episodes: int
    replay_freq: int
