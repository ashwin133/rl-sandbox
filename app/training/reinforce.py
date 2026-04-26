"""REINFORCE (Monte Carlo Policy Gradient) trainer."""

import logging
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import optim

from app.models.training import EpisodeMetrics, TrainingConfig, TrainingResult
from app.training.base import BaseTrainer
from app.training.networks import PolicyNetwork

logger = logging.getLogger(__name__)

SOLVED_REWARD_THRESHOLD = 475.0
ROLLING_WINDOW = 100


class ReinforceTrainer(BaseTrainer):
    """Trains a policy network using the REINFORCE algorithm.

    At each episode:
    1. Run the policy, collecting (log_prob, reward) at each step
    2. Compute discounted returns G_t for each step
    3. Normalise returns for variance reduction
    4. Loss = -sum(log_prob * normalised_return)
    5. Backprop and update
    """

    def __init__(self, env: gym.Env, config: TrainingConfig) -> None:
        super().__init__(env, config)
        obs_size: int = int(np.prod(env.observation_space.shape))  # type: ignore[arg-type]
        action_size: int = int(env.action_space.n)  # type: ignore[attr-defined]
        self._network: PolicyNetwork = PolicyNetwork(
            obs_size=obs_size,
            action_size=action_size,
            hidden_size=config.hidden_size,
        )
        self._optimizer: optim.Adam = optim.Adam(
            self._network.parameters(), lr=config.learning_rate
        )

    def train(self) -> TrainingResult:
        """Run the REINFORCE training loop for num_episodes episodes."""
        all_metrics: list[EpisodeMetrics] = []
        reward_history: list[float] = []

        for episode_idx in range(self._config.num_episodes):
            metrics: EpisodeMetrics = self._run_episode(episode_idx)
            all_metrics.append(metrics)
            reward_history.append(metrics.total_reward)

            rolling_avg: float = np.mean(reward_history[-ROLLING_WINDOW:]).item()
            if episode_idx % 50 == 0:
                logger.info(
                    f"Episode {episode_idx}: reward={metrics.total_reward:.0f}, "
                    f"avg={rolling_avg:.1f}, loss={metrics.loss:.4f}"
                )

        final_avg: float = np.mean(reward_history[-ROLLING_WINDOW:]).item()
        return TrainingResult(
            episodes=all_metrics,
            config=self._config,
            final_avg_reward=final_avg,
            solved=final_avg >= SOLVED_REWARD_THRESHOLD,
        )

    def _run_episode(self, episode_idx: int) -> EpisodeMetrics:
        """Run one episode, collect log_probs and rewards, then update."""
        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []

        obs, _info = self._env.reset()
        done: bool = False

        while not done:
            obs_tensor: torch.Tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(
                0
            )
            probs: torch.Tensor = self._network(obs_tensor)
            dist: torch.distributions.Categorical = torch.distributions.Categorical(
                probs
            )
            action: torch.Tensor = dist.sample()
            log_prob: torch.Tensor = dist.log_prob(action)

            obs, reward, terminated, truncated, _info = self._env.step(action.item())
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(float(reward))

        # Compute discounted returns and normalise
        returns: list[float] = self.compute_returns(rewards, self._config.gamma)
        returns_tensor: torch.Tensor = torch.tensor(returns, dtype=torch.float32)
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (
                returns_tensor.std() + 1e-8
            )

        # Policy gradient loss: -sum(log_prob * return)
        loss: torch.Tensor = torch.zeros(1)
        for log_prob, g in zip(log_probs, returns_tensor):
            loss = loss - log_prob * g

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return EpisodeMetrics(
            episode=episode_idx,
            total_reward=sum(rewards),
            steps=len(rewards),
            loss=loss.item(),
        )

    @staticmethod
    def compute_returns(rewards: list[float], gamma: float) -> list[float]:
        """Compute discounted returns G_t for each timestep.

        G_t = r_{t+1} + gamma * G_{t+1}, computed backwards for efficiency.
        """
        returns: list[float] = []
        g: float = 0.0
        for r in reversed(rewards):
            g = r + gamma * g
            returns.insert(0, g)
        return returns

    def save_model(self, path: Path) -> None:
        """Save the policy network weights to disk."""
        torch.save(self._network.state_dict(), path)

    def load_model(self, path: Path) -> None:
        """Load policy network weights from disk."""
        self._network.load_state_dict(torch.load(path, weights_only=True))
