"""DQN (Deep Q-Network) trainer."""

import logging
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import optim

from app.models.training import EpisodeMetrics, TrainingConfig, TrainingResult
from app.training.base import BaseTrainer
from app.training.networks import QNetwork
from app.training.replay_buffer import ReplayBuffer, Transition

logger = logging.getLogger(__name__)

SOLVED_REWARD_THRESHOLD = 475.0
ROLLING_WINDOW = 100


class DqnTrainer(BaseTrainer):
    """Trains a Q-network using the DQN algorithm.

    Key components:
    - Online Q-network: updated every step via mini-batch gradient descent
    - Target Q-network: frozen copy, synced periodically for stable targets
    - Replay buffer: stores transitions, sampled randomly to break correlation
    - Epsilon-greedy: balances exploration and exploitation
    """

    def __init__(self, env: gym.Env, config: TrainingConfig) -> None:
        super().__init__(env, config)
        obs_size: int = int(np.prod(env.observation_space.shape))  # type: ignore[arg-type]
        action_size: int = int(env.action_space.n)  # type: ignore[attr-defined]
        self._action_size: int = action_size

        self._online_network: QNetwork = QNetwork(
            obs_size=obs_size,
            action_size=action_size,
            hidden_size=config.hidden_size,
        )
        self._target_network: QNetwork = QNetwork(
            obs_size=obs_size,
            action_size=action_size,
            hidden_size=config.hidden_size,
        )
        self.sync_target_network()

        self._optimizer: optim.Adam = optim.Adam(
            self._online_network.parameters(), lr=config.learning_rate
        )
        self._buffer: ReplayBuffer = ReplayBuffer(capacity=config.buffer_capacity)
        self._epsilon: float = config.epsilon_start
        self._total_steps: int = 0

    def train(self) -> TrainingResult:
        """Run the DQN training loop for num_episodes episodes."""
        all_metrics: list[EpisodeMetrics] = []
        reward_history: list[float] = []

        for episode_idx in range(self._config.num_episodes):
            metrics: EpisodeMetrics = self._run_episode(episode_idx)
            all_metrics.append(metrics)
            reward_history.append(metrics.total_reward)

            # Decay epsilon after each episode
            self._epsilon = max(
                self._config.epsilon_end,
                self._epsilon * self._config.epsilon_decay,
            )

            rolling_avg: float = np.mean(reward_history[-ROLLING_WINDOW:]).item()
            if episode_idx % 50 == 0:
                logger.info(
                    f"Episode {episode_idx}: reward={metrics.total_reward:.0f}, "
                    f"avg={rolling_avg:.1f}, epsilon={self._epsilon:.3f}"
                )

        final_avg: float = np.mean(reward_history[-ROLLING_WINDOW:]).item()
        return TrainingResult(
            episodes=all_metrics,
            config=self._config,
            final_avg_reward=final_avg,
            solved=final_avg >= SOLVED_REWARD_THRESHOLD,
        )

    def _run_episode(self, episode_idx: int) -> EpisodeMetrics:
        """Run one episode, collecting transitions and training at each step."""
        obs, _info = self._env.reset()
        done: bool = False
        total_reward: float = 0.0
        steps: int = 0
        episode_loss: float = 0.0
        loss_count: int = 0

        while not done:
            state: np.ndarray = np.array(obs, dtype=np.float32)
            action: int = self.select_action(state, self._epsilon)

            next_obs, reward, terminated, truncated, _info = self._env.step(action)
            done = terminated or truncated
            next_state: np.ndarray = np.array(next_obs, dtype=np.float32)

            self._buffer.push(
                Transition(
                    state=state,
                    action=action,
                    reward=float(reward),
                    next_state=next_state,
                    done=done,
                )
            )

            total_reward += float(reward)
            steps += 1
            self._total_steps += 1

            # Train if buffer has enough samples
            if len(self._buffer) >= self._config.batch_size:
                batch: list[Transition] = self._buffer.sample(self._config.batch_size)
                loss: float = self._update(batch)
                episode_loss += loss
                loss_count += 1

            # Sync target network periodically
            if self._total_steps % self._config.target_update_freq == 0:
                self.sync_target_network()

            obs = next_obs

        avg_loss: float | None = episode_loss / loss_count if loss_count > 0 else None
        return EpisodeMetrics(
            episode=episode_idx,
            total_reward=total_reward,
            steps=steps,
            loss=avg_loss,
        )

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """Select an action using epsilon-greedy strategy."""
        if random.random() < epsilon:
            return random.randint(0, self._action_size - 1)

        state_tensor: torch.Tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(
            0
        )
        with torch.no_grad():
            q_values: torch.Tensor = self._online_network(state_tensor)
        return int(q_values.argmax(dim=-1).item())

    def _update(self, batch: list[Transition]) -> float:
        """Perform one gradient step on a mini-batch of transitions.

        Computes Bellman targets using the target network:
            y = r + gamma * (1 - done) * max_a' Q_target(s', a')
        Then minimises (Q_online(s, a) - y)^2.
        """
        states: torch.Tensor = torch.tensor(
            np.array([t.state for t in batch]), dtype=torch.float32
        )
        actions: torch.Tensor = torch.tensor(
            [t.action for t in batch], dtype=torch.long
        )
        rewards: torch.Tensor = torch.tensor(
            [t.reward for t in batch], dtype=torch.float32
        )
        next_states: torch.Tensor = torch.tensor(
            np.array([t.next_state for t in batch]), dtype=torch.float32
        )
        dones: torch.Tensor = torch.tensor([t.done for t in batch], dtype=torch.float32)

        # Q(s, a) for the actions actually taken
        q_values: torch.Tensor = self._online_network(states)
        q_taken: torch.Tensor = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Bellman target: r + gamma * (1 - done) * max Q_target(s')
        with torch.no_grad():
            next_q: torch.Tensor = self._target_network(next_states)
            max_next_q: torch.Tensor = next_q.max(dim=-1).values
            targets: torch.Tensor = (
                rewards + self._config.gamma * (1 - dones) * max_next_q
            )

        loss: torch.Tensor = torch.nn.functional.mse_loss(q_taken, targets)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def sync_target_network(self) -> None:
        """Copy online network weights into the target network."""
        self._target_network.load_state_dict(self._online_network.state_dict())

    def save_model(self, path: Path) -> None:
        """Save the online Q-network weights to disk."""
        torch.save(self._online_network.state_dict(), path)

    def load_model(self, path: Path) -> None:
        """Load Q-network weights from disk into both online and target networks."""
        state_dict = torch.load(path, weights_only=True)
        self._online_network.load_state_dict(state_dict)
        self._target_network.load_state_dict(state_dict)
