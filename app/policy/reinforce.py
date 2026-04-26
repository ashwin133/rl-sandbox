"""REINFORCE policy — wraps a trained PolicyNetwork as a BaseActionPolicy."""

from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

from app.policy.base import BaseActionPolicy
from app.training.networks import PolicyNetwork


class ReinforcePolicy(BaseActionPolicy):
    """Policy that uses a trained PolicyNetwork to select actions.

    Forward-passes the actual current observation through the network and
    samples from the resulting categorical distribution.
    """

    def __init__(self, network: PolicyNetwork) -> None:
        self._network: PolicyNetwork = network

    def choose_action(
        self, action_space: spaces.Space, observation: np.ndarray
    ) -> int:
        """Sample an action from the policy network's output distribution."""
        obs_tensor: torch.Tensor = torch.tensor(
            observation, dtype=torch.float32
        ).unsqueeze(0)
        with torch.no_grad():
            probs: torch.Tensor = self._network(obs_tensor)
        dist: torch.distributions.Categorical = torch.distributions.Categorical(probs)
        return int(dist.sample().item())

    def save(self, path: Path) -> None:
        """Save the policy network weights to disk."""
        torch.save(self._network.state_dict(), path)

    @classmethod
    def load(
        cls, path: Path, obs_size: int, action_size: int, hidden_size: int = 128
    ) -> "ReinforcePolicy":
        """Load a trained policy from disk."""
        network: PolicyNetwork = PolicyNetwork(
            obs_size=obs_size, action_size=action_size, hidden_size=hidden_size
        )
        network.load_state_dict(torch.load(path, weights_only=True))
        network.eval()
        return cls(network=network)
