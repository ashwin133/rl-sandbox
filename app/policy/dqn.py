"""DQN policy — wraps a trained QNetwork as a BaseActionPolicy."""

from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

from app.policy.base import BaseActionPolicy
from app.training.networks import QNetwork


class DqnPolicy(BaseActionPolicy):
    """Policy that uses a trained QNetwork to select actions.

    Always picks the greedy action (argmax of Q-values) — no exploration.
    """

    def __init__(self, network: QNetwork) -> None:
        self._network: QNetwork = network

    def choose_action(
        self, action_space: spaces.Space, observation: spaces.Space
    ) -> int:
        """Pick the action with the highest Q-value for a sampled observation."""
        obs: np.ndarray = observation.sample()
        return self.choose_action_from_obs(obs)

    def choose_action_from_obs(self, obs: np.ndarray) -> int:
        """Pick the action with the highest Q-value for a given observation."""
        obs_tensor: torch.Tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values: torch.Tensor = self._network(obs_tensor)
        return int(q_values.argmax(dim=-1).item())

    def save(self, path: Path) -> None:
        """Save the Q-network weights to disk."""
        torch.save(self._network.state_dict(), path)

    @classmethod
    def load(
        cls, path: Path, obs_size: int, action_size: int, hidden_size: int = 128
    ) -> "DqnPolicy":
        """Load a trained policy from disk."""
        network: QNetwork = QNetwork(
            obs_size=obs_size, action_size=action_size, hidden_size=hidden_size
        )
        network.load_state_dict(torch.load(path, weights_only=True))
        network.eval()
        return cls(network=network)
