"""REINFORCE policy — wraps a trained PolicyNetwork as a BaseActionPolicy."""

from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

from rl_sandbox.policy.base import BaseActionPolicy
from rl_sandbox.training.checkpoint import ALGORITHM_REINFORCE, Checkpoint
from rl_sandbox.training.networks import PolicyNetwork


class ReinforcePolicy(BaseActionPolicy):
    """Policy that uses a trained PolicyNetwork to select actions.

    Forward-passes the actual current observation through the network and
    samples from the resulting categorical distribution.
    """

    def __init__(self, network: PolicyNetwork) -> None:
        self._network: PolicyNetwork = network

    def choose_action(self, action_space: spaces.Space, observation: np.ndarray) -> int:
        """Sample an action from the policy network's output distribution."""
        obs_tensor: torch.Tensor = torch.tensor(
            observation, dtype=torch.float32
        ).unsqueeze(0)
        with torch.no_grad():
            probs: torch.Tensor = self._network(obs_tensor)
        dist: torch.distributions.Categorical = torch.distributions.Categorical(probs)
        return int(dist.sample().item())

    @classmethod
    def load(cls, path: Path, expected_env_id: str) -> "ReinforcePolicy":
        """Load a trained policy from disk; raise if algorithm or env_id mismatch."""
        raw: dict = torch.load(path, weights_only=True)
        checkpoint: Checkpoint = Checkpoint.model_validate(raw)
        if checkpoint.algorithm != ALGORITHM_REINFORCE:
            raise ValueError(
                f"Checkpoint algorithm {checkpoint.algorithm!r} "
                f"is not a REINFORCE checkpoint (expected {ALGORITHM_REINFORCE!r})"
            )
        if checkpoint.env_id != expected_env_id:
            raise ValueError(
                f"Checkpoint env_id {checkpoint.env_id!r} "
                f"does not match expected env {expected_env_id!r}"
            )
        network: PolicyNetwork = PolicyNetwork(
            obs_size=checkpoint.obs_size,
            action_size=checkpoint.action_size,
            hidden_size=checkpoint.hidden_size,
        )
        network.load_state_dict(checkpoint.state_dict)
        network.eval()
        return cls(network=network)
