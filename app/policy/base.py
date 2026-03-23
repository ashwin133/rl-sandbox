from abc import ABC, abstractmethod

from gymnasium import spaces


class BaseActionPolicy(ABC):
    @abstractmethod
    def choose_action(
        self, action_space: spaces.Space, observation: spaces.Space
    ) -> int:
        """Given the action space and current observation, choose an action."""
        ...
