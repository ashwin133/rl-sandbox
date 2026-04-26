"""Pydantic models for episode management."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class StepResult(BaseModel):
    """Result of a single environment step."""

    obs: Any
    reward: float
    terminated: bool
    truncated: bool
    done: bool  # terminated or truncated


class EpisodeConfig(BaseModel):
    """Configuration for a single episode."""

    max_steps: int | None = None  # None = run until terminated/truncated
    sim_speed_multiplier: float = 1  # Speed multiplier for the simulation (e.g., 2.0 = 2× speed, 0.5 = half speed)


class EpisodeResult(BaseModel):
    """Result of a single episode run."""

    sequence: list[StepResult]
    timestamp: datetime
    policy_name: str  # policy class name, e.g. "RandomActionPolicy"
    episode_config: EpisodeConfig
    timesteps_reached: int
