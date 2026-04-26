"""Tests for neural network modules (PolicyNetwork, QNetwork)."""

import torch

from app.training.networks import PolicyNetwork, QNetwork

CARTPOLE_OBS_SIZE = 4
CARTPOLE_ACTION_SIZE = 2


class TestPolicyNetwork:
    """Tests for PolicyNetwork used by REINFORCE."""

    def test_output_shape(self) -> None:
        net: PolicyNetwork = PolicyNetwork(
            obs_size=CARTPOLE_OBS_SIZE, action_size=CARTPOLE_ACTION_SIZE
        )
        obs: torch.Tensor = torch.randn(1, CARTPOLE_OBS_SIZE)
        probs: torch.Tensor = net(obs)
        assert probs.shape == (1, CARTPOLE_ACTION_SIZE)

    def test_output_is_probability_distribution(self) -> None:
        net: PolicyNetwork = PolicyNetwork(
            obs_size=CARTPOLE_OBS_SIZE, action_size=CARTPOLE_ACTION_SIZE
        )
        obs: torch.Tensor = torch.randn(1, CARTPOLE_OBS_SIZE)
        probs: torch.Tensor = net(obs)
        assert torch.allclose(probs.sum(dim=-1), torch.tensor([1.0]), atol=1e-6)
        assert (probs >= 0).all()

    def test_batch_forward_pass(self) -> None:
        net: PolicyNetwork = PolicyNetwork(
            obs_size=CARTPOLE_OBS_SIZE, action_size=CARTPOLE_ACTION_SIZE
        )
        batch: torch.Tensor = torch.randn(32, CARTPOLE_OBS_SIZE)
        probs: torch.Tensor = net(batch)
        assert probs.shape == (32, CARTPOLE_ACTION_SIZE)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(32), atol=1e-6)

    def test_gradients_flow(self) -> None:
        net: PolicyNetwork = PolicyNetwork(
            obs_size=CARTPOLE_OBS_SIZE, action_size=CARTPOLE_ACTION_SIZE
        )
        obs: torch.Tensor = torch.randn(1, CARTPOLE_OBS_SIZE)
        probs: torch.Tensor = net(obs)
        loss: torch.Tensor = -probs.log().sum()
        loss.backward()
        for param in net.parameters():
            assert param.grad is not None
            assert param.grad.abs().sum() > 0

    def test_deterministic_with_seed(self) -> None:
        torch.manual_seed(42)
        net1: PolicyNetwork = PolicyNetwork(
            obs_size=CARTPOLE_OBS_SIZE, action_size=CARTPOLE_ACTION_SIZE
        )
        torch.manual_seed(42)
        net2: PolicyNetwork = PolicyNetwork(
            obs_size=CARTPOLE_OBS_SIZE, action_size=CARTPOLE_ACTION_SIZE
        )
        obs: torch.Tensor = torch.randn(1, CARTPOLE_OBS_SIZE)
        assert torch.allclose(net1(obs), net2(obs))

    def test_custom_hidden_size(self) -> None:
        net: PolicyNetwork = PolicyNetwork(
            obs_size=CARTPOLE_OBS_SIZE, action_size=CARTPOLE_ACTION_SIZE, hidden_size=64
        )
        obs: torch.Tensor = torch.randn(1, CARTPOLE_OBS_SIZE)
        probs: torch.Tensor = net(obs)
        assert probs.shape == (1, CARTPOLE_ACTION_SIZE)


class TestQNetwork:
    """Tests for QNetwork used by DQN."""

    def test_output_shape(self) -> None:
        net: QNetwork = QNetwork(
            obs_size=CARTPOLE_OBS_SIZE, action_size=CARTPOLE_ACTION_SIZE
        )
        obs: torch.Tensor = torch.randn(1, CARTPOLE_OBS_SIZE)
        q_values: torch.Tensor = net(obs)
        assert q_values.shape == (1, CARTPOLE_ACTION_SIZE)

    def test_output_is_not_probability(self) -> None:
        """Q-values can be any real number, not bounded to [0,1]."""
        net: QNetwork = QNetwork(
            obs_size=CARTPOLE_OBS_SIZE, action_size=CARTPOLE_ACTION_SIZE
        )
        obs: torch.Tensor = torch.randn(100, CARTPOLE_OBS_SIZE)
        q_values: torch.Tensor = net(obs)
        # Q-values should NOT sum to 1 in general
        sums: torch.Tensor = q_values.sum(dim=-1)
        assert not torch.allclose(sums, torch.ones(100), atol=0.1)

    def test_batch_forward_pass(self) -> None:
        net: QNetwork = QNetwork(
            obs_size=CARTPOLE_OBS_SIZE, action_size=CARTPOLE_ACTION_SIZE
        )
        batch: torch.Tensor = torch.randn(32, CARTPOLE_OBS_SIZE)
        q_values: torch.Tensor = net(batch)
        assert q_values.shape == (32, CARTPOLE_ACTION_SIZE)

    def test_gradients_flow(self) -> None:
        net: QNetwork = QNetwork(
            obs_size=CARTPOLE_OBS_SIZE, action_size=CARTPOLE_ACTION_SIZE
        )
        obs: torch.Tensor = torch.randn(1, CARTPOLE_OBS_SIZE)
        q_values: torch.Tensor = net(obs)
        loss: torch.Tensor = q_values.sum()
        loss.backward()
        for param in net.parameters():
            assert param.grad is not None
            assert param.grad.abs().sum() > 0

    def test_deterministic_with_seed(self) -> None:
        torch.manual_seed(42)
        net1: QNetwork = QNetwork(
            obs_size=CARTPOLE_OBS_SIZE, action_size=CARTPOLE_ACTION_SIZE
        )
        torch.manual_seed(42)
        net2: QNetwork = QNetwork(
            obs_size=CARTPOLE_OBS_SIZE, action_size=CARTPOLE_ACTION_SIZE
        )
        obs: torch.Tensor = torch.randn(1, CARTPOLE_OBS_SIZE)
        assert torch.allclose(net1(obs), net2(obs))

    def test_custom_hidden_size(self) -> None:
        net: QNetwork = QNetwork(
            obs_size=CARTPOLE_OBS_SIZE, action_size=CARTPOLE_ACTION_SIZE, hidden_size=64
        )
        obs: torch.Tensor = torch.randn(1, CARTPOLE_OBS_SIZE)
        q_values: torch.Tensor = net(obs)
        assert q_values.shape == (1, CARTPOLE_ACTION_SIZE)
