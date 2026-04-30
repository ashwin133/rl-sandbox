"""BaseEnvApp — the abstract interface every concrete env wrapper implements."""

import asyncio
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np

from app.envs.registry import EnvSpec
from app.models.episode import StepResult
from app.policy.base import BaseActionPolicy


class BaseEnvApp(ABC):
    """Abstract env wrapper.

    Owns the gym.Env, an asyncio.Lock for concurrent access, and tracks the
    current observation across steps. Concrete subclasses (e.g., GymEnvApp)
    implement the gym-specific construction.
    """

    spec: EnvSpec
    env: gym.Env
    lock: asyncio.Lock

    @property
    @abstractmethod
    def current_obs(self) -> np.ndarray: ...

    @abstractmethod
    def apply_action(self, action_policy: BaseActionPolicy) -> StepResult: ...

    @abstractmethod
    def reset_env(self) -> None: ...
