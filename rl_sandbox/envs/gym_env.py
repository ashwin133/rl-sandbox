"""GymEnvApp — concrete BaseEnvApp backed by a Gymnasium environment.

Replaces the old CartPoleApp. Construction takes an EnvSpec; gym.make is
called using spec.env_id, so the same class works for any registered env.

Physics timing:
    Each gym env advances by a fixed timestep (spec.physics_tick_s) per
    env.step() call. The wall-clock delay between steps controls how fast
    the simulation *appears* to run, not how the physics behaves.
"""

import asyncio
import logging

import gymnasium as gym
import numpy as np

from rl_sandbox.envs.base import BaseEnvApp
from rl_sandbox.envs.registry import EnvSpec
from rl_sandbox.models.episode import StepResult
from rl_sandbox.policy.base import BaseActionPolicy

logger = logging.getLogger(__name__)

ACTION_SPACE_SEED = 42


class GymEnvApp(BaseEnvApp):
    """Generic gym-env wrapper parameterised by an EnvSpec."""

    def __init__(self, spec: EnvSpec) -> None:
        self.spec: EnvSpec = spec
        self.env: gym.Env = gym.make(spec.env_id, render_mode="rgb_array")
        self.env.action_space.seed(ACTION_SPACE_SEED)
        self.lock: asyncio.Lock = asyncio.Lock()
        obs, _info = self.env.reset()
        self.current_obs: np.ndarray = obs
        logger.info(f"Created GymEnvApp({spec.env_id}). Initial observation: {obs}")
        logger.info(
            f"Action space: {self.env.action_space}, "
            f"Observation space: {self.env.observation_space}"
        )

    def apply_action(self, action_policy: BaseActionPolicy) -> StepResult:
        """Take an action using the current observation, advance env, return result."""
        action: int = action_policy.choose_action(
            self.env.action_space, self.current_obs
        )
        obs, reward, terminated, truncated, _info = self.env.step(action)
        self.current_obs = obs
        done: bool = terminated or truncated
        if done:
            logger.info(
                f"Received reward {reward} and observation {obs}. "
                f"Episode ended by {'termination' if terminated else 'truncation'}."
            )
        return StepResult(
            obs=obs,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            done=done,
        )

    def reset_env(self) -> None:
        """Reset the environment."""
        obs, _info = self.env.reset()
        self.current_obs = obs
        logger.info(f"Environment reset. Initial observation: {obs}")
