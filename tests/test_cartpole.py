from collections.abc import Generator

import gymnasium as gym
import pytest

from app.cartpole import CartPoleApp
from app.policy.random import RandomActionPolicy


class TestCartPoleApp:
    """Unit tests for CartPoleApp."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> Generator[None, None, None]:
        self.app = CartPoleApp()
        yield  # type: ignore[misc]
        self.app.env.close()

    def test_init_creates_env(self) -> None:
        assert isinstance(self.app.env, gym.Env)

    def test_init_env_is_rgb_array(self) -> None:
        assert self.app.env.render_mode == "rgb_array"

    def test_init_resets_env(self) -> None:
        obs, _info = self.app.env.reset()
        assert self.app.env.observation_space.contains(obs)

    def test_apply_action_steps_env(self) -> None:
        policy = RandomActionPolicy()
        self.app.apply_action(policy)

    def test_apply_action_does_not_raise(self) -> None:
        policy = RandomActionPolicy()
        self.app.apply_action(policy)
        self.app.apply_action(policy)

    def test_env_can_be_closed(self) -> None:
        self.app.env.close()
