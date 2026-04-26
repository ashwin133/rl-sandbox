from abc import ABC, abstractmethod

import numpy as np
from gymnasium import spaces


class BaseActionPolicy(ABC):
    @abstractmethod
    def choose_action(
        self, action_space: spaces.Space, observation: np.ndarray
    ) -> int:
        """Given the action space and the current observation, choose an action.

        observation is the actual observation array returned by the env on the
        last reset/step — NOT the observation space. Trained policies need the
        real state to make meaningful decisions.
        """
        ...
