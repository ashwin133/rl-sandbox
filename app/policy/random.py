import numpy as np
from gymnasium import spaces

from app.policy.base import BaseActionPolicy


class RandomActionPolicy(BaseActionPolicy):
    def choose_action(
        self, action_space: spaces.Space, observation: np.ndarray
    ) -> int:
        """Choose a random action from the action space (observation ignored)."""
        return action_space.sample()
