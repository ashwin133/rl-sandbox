"""Base trainer ABC for RL algorithms."""

from abc import ABC, abstractmethod
from pathlib import Path

import gymnasium as gym

from app.models.training import TrainingConfig, TrainingResult


class BaseTrainer(ABC):
    """Abstract base class for RL trainers.

    Subclasses implement the specific training loop for each algorithm.
    The trainer owns the environment and config, and produces a TrainingResult.
    """

    def __init__(self, env: gym.Env, config: TrainingConfig) -> None:
        self._env: gym.Env = env
        self._config: TrainingConfig = config

    @abstractmethod
    def train(self) -> TrainingResult:
        """Run the full training loop and return results."""
        ...

    @abstractmethod
    def save_model(self, path: Path) -> None:
        """Save the trained model to disk."""
        ...

    @abstractmethod
    def load_model(self, path: Path) -> None:
        """Load a trained model from disk."""
        ...
