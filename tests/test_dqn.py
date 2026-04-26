"""Tests for DQN trainer and policy."""

import gymnasium as gym
import numpy as np
import torch

from app.models.training import TrainingConfig
from app.policy.dqn import DqnPolicy
from app.training.dqn import DqnTrainer
from app.training.networks import QNetwork

CARTPOLE_OBS_SIZE = 4
CARTPOLE_ACTION_SIZE = 2


class TestDqnTrainerSelectAction:
    """Tests for epsilon-greedy action selection."""

    def test_epsilon_one_always_random(self) -> None:
        """With epsilon=1.0, all actions should come from random sampling."""
        env: gym.Env = gym.make("CartPole-v1")
        config: TrainingConfig = TrainingConfig(algorithm="dqn", num_episodes=1)
        trainer: DqnTrainer = DqnTrainer(env=env, config=config)
        state: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0])

        actions: set[int] = set()
        for _ in range(100):
            action: int = trainer.select_action(state, epsilon=1.0)
            assert action in {0, 1}
            actions.add(action)
        # With 100 samples at epsilon=1.0, we should see both actions
        assert len(actions) == 2
        env.close()

    def test_epsilon_zero_is_greedy(self) -> None:
        """With epsilon=0.0, action should be argmax of Q-values."""
        env: gym.Env = gym.make("CartPole-v1")
        config: TrainingConfig = TrainingConfig(algorithm="dqn", num_episodes=1)
        trainer: DqnTrainer = DqnTrainer(env=env, config=config)
        state: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0])

        # All calls with epsilon=0 should return the same action
        actions: list[int] = [
            trainer.select_action(state, epsilon=0.0) for _ in range(10)
        ]
        assert len(set(actions)) == 1
        env.close()


class TestDqnTrainerTargetSync:
    """Tests for target network synchronisation."""

    def test_target_network_syncs(self) -> None:
        env: gym.Env = gym.make("CartPole-v1")
        config: TrainingConfig = TrainingConfig(algorithm="dqn", num_episodes=1)
        trainer: DqnTrainer = DqnTrainer(env=env, config=config)

        # Modify online network
        with torch.no_grad():
            for param in trainer._online_network.parameters():
                param.fill_(99.0)

        # Before sync, target should differ
        target_params: list[torch.Tensor] = list(trainer._target_network.parameters())
        assert not torch.allclose(target_params[0], torch.tensor(99.0))

        # After sync, target should match online
        trainer.sync_target_network()
        for online_p, target_p in zip(
            trainer._online_network.parameters(),
            trainer._target_network.parameters(),
        ):
            assert torch.allclose(online_p, target_p)
        env.close()


class TestDqnTrainerTrain:
    """Tests for the DQN training loop."""

    def test_smoke_test_runs_without_error(self) -> None:
        env: gym.Env = gym.make("CartPole-v1")
        config: TrainingConfig = TrainingConfig(
            algorithm="dqn",
            num_episodes=5,
            batch_size=16,
            buffer_capacity=500,
            target_update_freq=50,
        )
        trainer: DqnTrainer = DqnTrainer(env=env, config=config)
        result = trainer.train()
        assert len(result.episodes) == 5
        for ep in result.episodes:
            assert ep.total_reward > 0
            assert ep.steps > 0
        env.close()

    def test_epsilon_decays(self) -> None:
        env: gym.Env = gym.make("CartPole-v1")
        config: TrainingConfig = TrainingConfig(
            algorithm="dqn",
            num_episodes=10,
            epsilon_start=1.0,
            epsilon_decay=0.9,
            epsilon_end=0.01,
        )
        trainer: DqnTrainer = DqnTrainer(env=env, config=config)
        trainer.train()
        # After 10 episodes with decay=0.9: 1.0 * 0.9^10 ≈ 0.349
        assert trainer._epsilon < config.epsilon_start
        env.close()


class TestDqnPolicy:
    """Tests for DqnPolicy conforming to BaseActionPolicy."""

    def test_produces_valid_action(self) -> None:
        env: gym.Env = gym.make("CartPole-v1")
        network: QNetwork = QNetwork(
            obs_size=CARTPOLE_OBS_SIZE, action_size=CARTPOLE_ACTION_SIZE
        )
        policy: DqnPolicy = DqnPolicy(network=network)
        action: int = policy.choose_action(env.action_space, env.observation_space)
        assert env.action_space.contains(action)
        env.close()

    def test_greedy_is_deterministic(self) -> None:
        """DqnPolicy should always pick argmax — same obs should give same action."""
        env: gym.Env = gym.make("CartPole-v1")
        network: QNetwork = QNetwork(
            obs_size=CARTPOLE_OBS_SIZE, action_size=CARTPOLE_ACTION_SIZE
        )
        policy: DqnPolicy = DqnPolicy(network=network)
        obs: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0])
        actions: list[int] = [policy.choose_action_from_obs(obs) for _ in range(10)]
        assert len(set(actions)) == 1
        env.close()

    def test_save_and_load_roundtrip(self, tmp_path: "np.str_") -> None:
        network: QNetwork = QNetwork(
            obs_size=CARTPOLE_OBS_SIZE, action_size=CARTPOLE_ACTION_SIZE
        )
        policy: DqnPolicy = DqnPolicy(network=network)
        save_path = tmp_path / "model.pt"  # type: ignore[operator]
        policy.save(save_path)

        loaded: DqnPolicy = DqnPolicy.load(
            save_path,
            obs_size=CARTPOLE_OBS_SIZE,
            action_size=CARTPOLE_ACTION_SIZE,
        )
        obs: torch.Tensor = torch.randn(1, CARTPOLE_OBS_SIZE)
        with torch.no_grad():
            original_out: torch.Tensor = network(obs)
            loaded_out: torch.Tensor = loaded._network(obs)
        assert torch.allclose(original_out, loaded_out)
