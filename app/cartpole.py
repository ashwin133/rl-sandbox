import asyncio
import logging

import gymnasium as gym
import numpy as np

from app.models.episode import StepResult
from app.policy.base import BaseActionPolicy

logger = logging.getLogger(__name__)

MODEL_TICK_BETWEEN_ACTIONS_S = (
    0.02  # CartPole-v1 advances by a fixed timestep (tau = 0.02s) per env.step() call.
)


class CartPoleApp:
    """A simple app to demonstrate creating a CartPole environment and logging.

    Physics timing:
        CartPole-v1 advances by a fixed timestep (tau = 0.02s) per env.step() call.
        The wall-clock delay between steps controls how fast the simulation *appears*
        to run, not how the physics behaves. Each step always simulates exactly 0.02s
        of physics regardless of how long the caller waits.

        - 1000 steps/sec (1ms sleep)  → 20× real-time
        - 50 steps/sec  (20ms sleep) → 1× real-time  (matches tau = 0.02s)
        - 10 steps/sec  (100ms sleep) → 0.2× real-time (slow motion)

        The action frequency is always 1 action per step — Gymnasium's API requires
        exactly one action per env.step() call.
    """

    def __init__(self) -> None:
        # Create env in rgb_array mode — gives us pixel frames instead of a window
        self.env: gym.Env = gym.make("CartPole-v1", render_mode="rgb_array")
        self.env.action_space.seed(42)  # Make the episode deterministic (for testing)
        self.lock: asyncio.Lock = asyncio.Lock()
        obs, _info = self.env.reset()
        self._current_obs: np.ndarray = obs
        logger.info(f"Initial observation: {obs}")
        logger.info(
            f"Action space: {self.env.action_space}, Observation space: {self.env.observation_space}"
        )

    def apply_action(self, action_policy: BaseActionPolicy) -> StepResult:
        """Take an action using the current observation, advance the env, return result."""
        action: int = action_policy.choose_action(
            self.env.action_space, self._current_obs
        )
        obs, reward, terminated, truncated, _info = self.env.step(action)
        self._current_obs = obs
        done: bool = (
            terminated or truncated
        )  # terminated = pole fell (or in another case could mean reward hit), truncated = hit max episode length
        if done:
            logger.info(
                f"Received reward {reward} and observation {obs}. Episode ended by {'termination' if terminated else 'truncation'}."
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
        self._current_obs = obs
        logger.info(f"Environment reset. Initial observation: {obs}")
