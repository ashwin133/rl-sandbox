"""Pydantic models for batch recording metadata (serialised to disk)."""

from datetime import datetime

from pydantic import BaseModel

from rl_sandbox.envs.registry import EnvSpec


class EpisodeRecording(BaseModel):
    """Metadata for a single captured episode."""

    episode_index: int
    policy_name: str
    timestamp: datetime
    timesteps_reached: int
    sim_speed_multiplier: float


class BatchRecording(BaseModel):
    """Metadata for a batch of episodes.

    Embeds the full EnvSpec so replay is self-describing — playback can
    compute its physics tick from env_spec.physics_tick_s without needing
    the registry to still contain that env id.
    """

    no_episodes: int  # total episodes in the batch (including non-captured)
    replay_freq: int
    env_spec: EnvSpec
    episodes: list[EpisodeRecording]  # only the captured episodes
