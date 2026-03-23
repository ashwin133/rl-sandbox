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
        obs, info = self.env.reset()
        logger.info("Initial observation: %s", obs)
        logger.info(
            "Action space: %s, Observation space: %s",
            self.env.action_space,
            self.env.observation_space,
        )

    def apply_action(self, action_policy: BaseActionPolicy) -> None:
        """Take a random action and log the result."""
        action = action_policy.choose_action(
            self.env.action_space, self.env.observation_space
        )
        obs, reward, terminated, truncated, _info = self.env.step(action)
        logger.info(
            "Took action %s, got obs %s, reward %.2f, terminated %s, truncated %s",
            action,
            obs,
            reward,
            terminated,
            truncated,
        )
