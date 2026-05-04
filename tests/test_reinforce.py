"""Tests for REINFORCE trainer and policy."""

import math
from pathlib import Path

import gymnasium as gym
import pytest
import torch

from rl_sandbox.envs.registry import SUPPORTED_ENVS, EnvSpec
from rl_sandbox.models.training import TrainingConfig
from rl_sandbox.policy.reinforce import ReinforcePolicy
from rl_sandbox.training.checkpoint import ALGORITHM_REINFORCE, Checkpoint
from rl_sandbox.training.networks import PolicyNetwork
from rl_sandbox.training.reinforce import ReinforceTrainer

CARTPOLE_OBS_SIZE = 4
CARTPOLE_ACTION_SIZE = 2
MAX_ENTROPY = math.log(CARTPOLE_ACTION_SIZE)
CARTPOLE_SPEC: EnvSpec = SUPPORTED_ENVS["CartPole-v1"]


def _make_trainer(
    num_episodes: int = 1, learning_rate: float = 1e-3
) -> ReinforceTrainer:
    env: gym.Env = gym.make("CartPole-v1")
    config: TrainingConfig = TrainingConfig(
        num_episodes=num_episodes, learning_rate=learning_rate
    )
    return ReinforceTrainer(env=env, config=config, spec=CARTPOLE_SPEC)


class TestReinforceTrainerComputeReturns:
    """Tests for discounted return computation."""

    def test_single_reward(self) -> None:
        trainer: ReinforceTrainer = _make_trainer()
        returns: list[float] = trainer.compute_returns([1.0], gamma=0.99)
        assert len(returns) == 1
        assert returns[0] == 1.0

    def test_undiscounted(self) -> None:
        trainer: ReinforceTrainer = _make_trainer()
        returns: list[float] = trainer.compute_returns([1.0, 1.0, 1.0], gamma=1.0)
        assert returns == [3.0, 2.0, 1.0]

    def test_discounted(self) -> None:
        trainer: ReinforceTrainer = _make_trainer()
        returns: list[float] = trainer.compute_returns([1.0, 1.0, 1.0], gamma=0.5)
        assert returns[2] == 1.0
        assert returns[1] == 1.5
        assert returns[0] == 1.75

    def test_empty_rewards(self) -> None:
        trainer: ReinforceTrainer = _make_trainer()
        returns: list[float] = trainer.compute_returns([], gamma=0.99)
        assert returns == []


class TestReinforceTrainerTrain:
    """Tests for the REINFORCE training loop."""

    def test_smoke_test_runs_without_error(self) -> None:
        trainer = _make_trainer(num_episodes=5)
        result = trainer.train()
        assert len(result.episodes) == 5
        for ep in result.episodes:
            assert ep.total_reward > 0
            assert ep.steps > 0

    def test_loss_is_tracked(self) -> None:
        trainer = _make_trainer(num_episodes=3)
        result = trainer.train()
        for ep in result.episodes:
            assert ep.loss is not None

    def test_entropy_is_tracked_and_in_range(self) -> None:
        trainer = _make_trainer(num_episodes=3)
        result = trainer.train()
        for ep in result.episodes:
            assert ep.entropy is not None
            assert 0.0 <= ep.entropy <= MAX_ENTROPY + 1e-6

    def test_final_eval_recorded(self) -> None:
        trainer = _make_trainer(num_episodes=3)
        result = trainer.train()
        # With num_episodes=3 < EVAL_INTERVAL_EPISODES, only the final eval fires
        assert len(result.evaluations) == 1
        eval_result = result.evaluations[0]
        assert eval_result.n_episodes > 0
        assert eval_result.mean_reward >= 0.0
        assert eval_result.std_reward >= 0.0

    def test_periodic_eval_fires(self) -> None:
        """At num_episodes=50, exactly one periodic eval + one final eval should fire."""
        trainer = _make_trainer(num_episodes=50)
        result = trainer.train()
        assert len(result.evaluations) == 2

    def test_result_carries_env_spec(self) -> None:
        trainer = _make_trainer(num_episodes=2)
        result = trainer.train()
        assert result.env_spec.env_id == "CartPole-v1"


class TestReinforcePolicy:
    """Tests for ReinforcePolicy conforming to BaseActionPolicy."""

    def test_produces_valid_action(self) -> None:
        env: gym.Env = gym.make("CartPole-v1")
        network: PolicyNetwork = PolicyNetwork(
            obs_size=CARTPOLE_OBS_SIZE, action_size=CARTPOLE_ACTION_SIZE
        )
        policy: ReinforcePolicy = ReinforcePolicy(network=network)
        obs, _info = env.reset()
        action: int = policy.choose_action(env.action_space, obs)
        assert env.action_space.contains(action)
        env.close()

    def test_multiple_actions_are_valid(self) -> None:
        env: gym.Env = gym.make("CartPole-v1")
        network: PolicyNetwork = PolicyNetwork(
            obs_size=CARTPOLE_OBS_SIZE, action_size=CARTPOLE_ACTION_SIZE
        )
        policy: ReinforcePolicy = ReinforcePolicy(network=network)
        obs, _info = env.reset()
        for _ in range(20):
            action: int = policy.choose_action(env.action_space, obs)
            assert env.action_space.contains(action)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()


class TestReinforcePolicyCheckpoint:
    """Tests for the self-describing checkpoint format."""

    def _save_checkpoint(
        self,
        path: Path,
        env_id: str,
        algorithm: str = ALGORITHM_REINFORCE,
    ) -> PolicyNetwork:
        """Build a fresh CartPole-shaped network, save it, return the network."""
        network: PolicyNetwork = PolicyNetwork(
            obs_size=CARTPOLE_OBS_SIZE, action_size=CARTPOLE_ACTION_SIZE
        )
        checkpoint: Checkpoint = Checkpoint(
            state_dict=dict(network.state_dict()),
            algorithm=algorithm,
            env_id=env_id,
            obs_size=CARTPOLE_OBS_SIZE,
            action_size=CARTPOLE_ACTION_SIZE,
            hidden_size=128,
        )
        torch.save(checkpoint.model_dump(), path)
        return network

    def test_load_round_trip_preserves_weights(self, tmp_path: Path) -> None:
        save_path = tmp_path / "model.pt"
        network: PolicyNetwork = self._save_checkpoint(save_path, env_id="CartPole-v1")

        loaded: ReinforcePolicy = ReinforcePolicy.load(
            save_path, expected_env_id="CartPole-v1"
        )

        torch.manual_seed(0)
        obs: torch.Tensor = torch.randn(1, CARTPOLE_OBS_SIZE)
        with torch.no_grad():
            original_out: torch.Tensor = network(obs)
            loaded_out: torch.Tensor = loaded._network(obs)
        assert torch.allclose(original_out, loaded_out)

    def test_load_rejects_wrong_env_id(self, tmp_path: Path) -> None:
        save_path = tmp_path / "model.pt"
        self._save_checkpoint(save_path, env_id="CartPole-v1")

        with pytest.raises(ValueError, match="CartPole-v1.*Acrobot-v1"):
            ReinforcePolicy.load(save_path, expected_env_id="Acrobot-v1")

    def test_trainer_save_uses_self_describing_format(self, tmp_path: Path) -> None:
        trainer = _make_trainer(num_episodes=1)
        save_path = tmp_path / "model.pt"
        trainer.save_model(save_path)

        raw: dict = torch.load(save_path, weights_only=True)
        checkpoint: Checkpoint = Checkpoint.model_validate(raw)
        assert checkpoint.algorithm == ALGORITHM_REINFORCE
        assert checkpoint.env_id == "CartPole-v1"
        assert checkpoint.obs_size == CARTPOLE_OBS_SIZE
        assert checkpoint.action_size == CARTPOLE_ACTION_SIZE
        assert checkpoint.hidden_size == 128
        assert len(checkpoint.state_dict) > 0

    def test_load_rejects_corrupt_checkpoint(self, tmp_path: Path) -> None:
        """A dict missing required fields should fail Pydantic validation."""
        save_path = tmp_path / "model.pt"
        torch.save({"env_id": "CartPole-v1"}, save_path)
        with pytest.raises(ValueError):
            ReinforcePolicy.load(save_path, expected_env_id="CartPole-v1")

    def test_load_rejects_wrong_algorithm(self, tmp_path: Path) -> None:
        save_path = tmp_path / "model.pt"
        self._save_checkpoint(save_path, env_id="CartPole-v1", algorithm="dqn")
        with pytest.raises(ValueError, match="dqn"):
            ReinforcePolicy.load(save_path, expected_env_id="CartPole-v1")

    def test_trainer_load_model_rejects_wrong_env_id(self, tmp_path: Path) -> None:
        save_path = tmp_path / "model.pt"
        self._save_checkpoint(save_path, env_id="Acrobot-v1")
        trainer = _make_trainer(num_episodes=1)
        with pytest.raises(ValueError, match="Acrobot-v1"):
            trainer.load_model(save_path)

    def test_trainer_load_model_rejects_wrong_algorithm(self, tmp_path: Path) -> None:
        save_path = tmp_path / "model.pt"
        self._save_checkpoint(save_path, env_id="CartPole-v1", algorithm="dqn")
        trainer = _make_trainer(num_episodes=1)
        with pytest.raises(ValueError, match="dqn"):
            trainer.load_model(save_path)
