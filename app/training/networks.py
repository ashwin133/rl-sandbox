"""Neural network modules for RL training."""

import torch
from torch import nn


class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE.

    Takes an observation and outputs a probability distribution over actions
    via softmax. Two hidden layers with ReLU activations.
    """

    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.network: nn.Sequential = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
