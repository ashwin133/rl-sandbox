from gymnasium import spaces

from app.policy.base import BaseActionPolicy


class RandomActionPolicy(BaseActionPolicy):
    def choose_action(
        self, action_space: spaces.Space, observation: spaces.Space
    ) -> int:
        """Choose a random action from the action space."""
        return action_space.sample()
