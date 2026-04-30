"""REINFORCE policy — wraps a trained PolicyNetwork as a BaseActionPolicy."""

from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

from app.policy.base import BaseActionPolicy
from app.training.networks import PolicyNetwork

CHECKPOINT_STATE_DICT_KEY = "state_dict"
CHECKPOINT_ENV_ID_KEY = "env_id"
CHECKPOINT_OBS_SIZE_KEY = "obs_size"
CHECKPOINT_ACTION_SIZE_KEY = "action_size"
CHECKPOINT_HIDDEN_SIZE_KEY = "hidden_size"


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

    @classmethod
    def load(cls, path: Path, expected_env_id: str) -> "ReinforcePolicy":
        """Load a trained policy from disk; raise if the checkpoint env_id mismatches."""
        checkpoint = torch.load(path, weights_only=False)
        if checkpoint[CHECKPOINT_ENV_ID_KEY] != expected_env_id:
            raise ValueError(
                f"Checkpoint env_id {checkpoint[CHECKPOINT_ENV_ID_KEY]!r} "
                f"does not match expected env {expected_env_id!r}"
            )
        network: PolicyNetwork = PolicyNetwork(
            obs_size=checkpoint[CHECKPOINT_OBS_SIZE_KEY],
            action_size=checkpoint[CHECKPOINT_ACTION_SIZE_KEY],
            hidden_size=checkpoint[CHECKPOINT_HIDDEN_SIZE_KEY],
        )
        network.load_state_dict(checkpoint[CHECKPOINT_STATE_DICT_KEY])
        network.eval()
        return cls(network=network)
