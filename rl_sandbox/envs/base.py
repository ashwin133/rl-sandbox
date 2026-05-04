"""BaseEnvApp — the abstract interface every concrete env wrapper implements."""

import asyncio
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np

from rl_sandbox.envs.registry import EnvSpec
from rl_sandbox.models.episode import StepResult
from rl_sandbox.policy.base import BaseActionPolicy


class BaseEnvApp(ABC):
    """Abstract env wrapper.

    Owns the gym.Env, an asyncio.Lock for concurrent access, and tracks the
    current observation across steps. Concrete subclasses (e.g., GymEnvApp)
    must set `spec`, `env`, `lock`, and `current_obs` as instance attributes
    in `__init__`, and implement `apply_action` and `reset_env`.
    """

    spec: EnvSpec
    env: gym.Env
    lock: asyncio.Lock
    current_obs: np.ndarray

    @abstractmethod
    def apply_action(self, action_policy: BaseActionPolicy) -> StepResult: ...

    @abstractmethod
    def reset_env(self) -> None: ...
