import asyncio
import logging

import gymnasium as gym

from app.policy import BaseActionPolicy

logger = logging.getLogger(__name__)


class CartPoleApp:
    """A simple app to demonstrate creating a CartPole environment and logging."""

    def __init__(self) -> None:
        # Create env in rgb_array mode — gives us pixel frames instead of a window
        self.env: gym.Env = gym.make("CartPole-v1", render_mode="rgb_array")
        self.env.action_space.seed(42)  # Make the episode deterministic (for testing)
        self.lock = asyncio.Lock()
        obs, info = self.env.reset()
        logger.info("Initial observation: %s", obs)
        logger.info(
            "Action space: %s, Observation space: %s",
            self.env.action_space,
            self.env.observation_space,
        )

    def apply_action(self, action_policy: BaseActionPolicy) -> bool:
        """Take an action and return whether the episode has ended."""
        action = action_policy.choose_action(
            self.env.action_space, self.env.observation_space
        )
        obs, reward, terminated, truncated, _info = self.env.step(action)
        if terminated or truncated:
            logger.info(
                f"Received reward {reward} and observation {obs}. Episode ended by {'termination' if terminated else 'truncation'}."
            )
        return terminated or truncated

    def reset_env(self) -> None:
        """Reset the environment."""
        obs, info = self.env.reset()
        logger.info("Environment reset. Initial observation: %s", obs)

    async def loop_policy_over_episodes(self, action_policy: BaseActionPolicy) -> None:
        """Loop over episodes, applying the policy until the episode ends, then reset."""
        while True:
            logger.info("Starting new episode.")
            await self.apply_policy_until_episode_end(action_policy)
            self.reset_env()

    async def apply_policy_until_episode_end(
        self, action_policy: BaseActionPolicy
    ) -> None:
        """Apply the given policy to take an action and log the result."""
        episode_end: bool = False
        while not episode_end:
            async with self.lock:
                episode_end: bool = self.apply_action(action_policy)
                if episode_end:
                    logger.info("Episode ended.")

            await asyncio.sleep(0.001)  # Sleep for tick length
