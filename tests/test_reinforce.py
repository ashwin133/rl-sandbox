"""Tests for REINFORCE trainer and policy."""

import math

import gymnasium as gym
import numpy as np
import torch

from app.models.training import TrainingConfig
from app.policy.reinforce import ReinforcePolicy
from app.training.networks import PolicyNetwork
from app.training.reinforce import ReinforceTrainer

CARTPOLE_OBS_SIZE = 4
CARTPOLE_ACTION_SIZE = 2
MAX_ENTROPY = math.log(CARTPOLE_ACTION_SIZE)


class TestReinforceTrainerComputeReturns:
    """Tests for discounted return computation."""

    def test_single_reward(self) -> None:
        trainer: ReinforceTrainer = self._make_trainer()
        returns: list[float] = trainer.compute_returns([1.0], gamma=0.99)
        assert len(returns) == 1
        assert returns[0] == 1.0

    def test_undiscounted(self) -> None:
        """With gamma=1.0, return at each step is just the sum of future rewards."""
        trainer: ReinforceTrainer = self._make_trainer()
        returns: list[float] = trainer.compute_returns([1.0, 1.0, 1.0], gamma=1.0)
        assert returns == [3.0, 2.0, 1.0]

    def test_discounted(self) -> None:
        trainer: ReinforceTrainer = self._make_trainer()
        returns: list[float] = trainer.compute_returns([1.0, 1.0, 1.0], gamma=0.5)
        # G_2 = 1.0
        # G_1 = 1.0 + 0.5 * 1.0 = 1.5
        # G_0 = 1.0 + 0.5 * 1.5 = 1.75
        assert returns[2] == 1.0
        assert returns[1] == 1.5
        assert returns[0] == 1.75

    def test_empty_rewards(self) -> None:
        trainer: ReinforceTrainer = self._make_trainer()
        returns: list[float] = trainer.compute_returns([], gamma=0.99)
        assert returns == []

    def _make_trainer(self) -> ReinforceTrainer:
        env: gym.Env = gym.make("CartPole-v1")
        config: TrainingConfig = TrainingConfig(algorithm="reinforce", num_episodes=1)
        trainer: ReinforceTrainer = ReinforceTrainer(env=env, config=config)
        return trainer


class TestReinforceTrainerTrain:
    """Tests for the REINFORCE training loop."""

    def test_smoke_test_runs_without_error(self) -> None:
        env: gym.Env = gym.make("CartPole-v1")
        config: TrainingConfig = TrainingConfig(
            algorithm="reinforce", num_episodes=5, learning_rate=1e-3
        )
        trainer: ReinforceTrainer = ReinforceTrainer(env=env, config=config)
        result = trainer.train()
        assert len(result.episodes) == 5
        for ep in result.episodes:
            assert ep.total_reward > 0
            assert ep.steps > 0

    def test_loss_is_tracked(self) -> None:
        env: gym.Env = gym.make("CartPole-v1")
        config: TrainingConfig = TrainingConfig(algorithm="reinforce", num_episodes=3)
        trainer: ReinforceTrainer = ReinforceTrainer(env=env, config=config)
        result = trainer.train()
        for ep in result.episodes:
            assert ep.loss is not None

    def test_entropy_is_tracked_and_in_range(self) -> None:
        env: gym.Env = gym.make("CartPole-v1")
        config: TrainingConfig = TrainingConfig(algorithm="reinforce", num_episodes=3)
        trainer: ReinforceTrainer = ReinforceTrainer(env=env, config=config)
        result = trainer.train()
        for ep in result.episodes:
            assert ep.entropy is not None
            assert 0.0 <= ep.entropy <= MAX_ENTROPY + 1e-6

    def test_final_eval_recorded(self) -> None:
        env: gym.Env = gym.make("CartPole-v1")
        config: TrainingConfig = TrainingConfig(algorithm="reinforce", num_episodes=3)
        trainer: ReinforceTrainer = ReinforceTrainer(env=env, config=config)
        result = trainer.train()
        # With num_episodes=3 < EVAL_INTERVAL_EPISODES, only the final eval fires
        assert len(result.evaluations) == 1
        eval_result = result.evaluations[0]
        assert eval_result.n_episodes > 0
        assert eval_result.mean_reward >= 0.0
        assert eval_result.std_reward >= 0.0

    def test_periodic_eval_fires(self) -> None:
        """At num_episodes=50, exactly one periodic eval + one final eval should fire."""
        env: gym.Env = gym.make("CartPole-v1")
        config: TrainingConfig = TrainingConfig(
            algorithm="reinforce", num_episodes=50
        )
        trainer: ReinforceTrainer = ReinforceTrainer(env=env, config=config)
        result = trainer.train()
        assert len(result.evaluations) == 2


class TestReinforcePolicy:
    """Tests for ReinforcePolicy conforming to BaseActionPolicy."""

    def test_produces_valid_action(self) -> None:
        env: gym.Env = gym.make("CartPole-v1")
        network: PolicyNetwork = PolicyNetwork(
            obs_size=CARTPOLE_OBS_SIZE, action_size=CARTPOLE_ACTION_SIZE
        )
        policy: ReinforcePolicy = ReinforcePolicy(network=network)
        action: int = policy.choose_action(env.action_space, env.observation_space)
        assert env.action_space.contains(action)
        env.close()

    def test_multiple_actions_are_valid(self) -> None:
        env: gym.Env = gym.make("CartPole-v1")
        network: PolicyNetwork = PolicyNetwork(
            obs_size=CARTPOLE_OBS_SIZE, action_size=CARTPOLE_ACTION_SIZE
        )
        policy: ReinforcePolicy = ReinforcePolicy(network=network)
        env.reset()
        for _ in range(20):
            action: int = policy.choose_action(env.action_space, env.observation_space)
            assert env.action_space.contains(action)
        env.close()

    def test_save_and_load_roundtrip(self, tmp_path: "np.str_") -> None:
        network: PolicyNetwork = PolicyNetwork(
            obs_size=CARTPOLE_OBS_SIZE, action_size=CARTPOLE_ACTION_SIZE
        )
        policy: ReinforcePolicy = ReinforcePolicy(network=network)
        save_path = tmp_path / "model.pt"  # type: ignore[operator]
        policy.save(save_path)

        loaded: ReinforcePolicy = ReinforcePolicy.load(
            save_path,
            obs_size=CARTPOLE_OBS_SIZE,
            action_size=CARTPOLE_ACTION_SIZE,
        )
        # Same output for same input
        torch.manual_seed(0)
        obs: torch.Tensor = torch.randn(1, CARTPOLE_OBS_SIZE)
        with torch.no_grad():
            original_out: torch.Tensor = network(obs)
            loaded_out: torch.Tensor = loaded._network(obs)
        assert torch.allclose(original_out, loaded_out)
