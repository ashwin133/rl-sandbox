from collections.abc import Generator

import gymnasium as gym
import numpy as np
import pytest

from app.policy import BaseActionPolicy, RandomActionPolicy


class TestBaseActionPolicy:
    """Unit tests for BaseActionPolicy."""

    def test_base_policy_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseActionPolicy()  # type: ignore[abstract]


class TestRandomActionPolicy:
    """Unit tests for RandomActionPolicy."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> Generator[None, None, None]:
        self.env = gym.make("CartPole-v1")
        self.policy = RandomActionPolicy()
        yield  # type: ignore[misc]
        self.env.close()

    def test_action_in_space(self) -> None:
        action = self.policy.choose_action(
            self.env.action_space, self.env.observation_space
        )
        assert self.env.action_space.contains(action)

    def test_returns_int(self) -> None:
        action = self.policy.choose_action(
            self.env.action_space, self.env.observation_space
        )
        assert isinstance(action, (int, np.integer))

    def test_actions_vary(self) -> None:
        actions = {
            self.policy.choose_action(self.env.action_space, self.env.observation_space)
            for _ in range(50)
        }
        assert len(actions) > 1
