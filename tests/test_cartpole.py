import logging
from collections.abc import Generator

import gymnasium as gym
import pytest

from app.cartpole import CartPoleApp
from app.policy import RandomActionPolicy


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

    def test_apply_action_logs_output(self, caplog: pytest.LogCaptureFixture) -> None:
        policy = RandomActionPolicy()
        with caplog.at_level(logging.INFO):
            self.app.apply_action(policy)
        assert "Took action" in caplog.text
        assert "reward" in caplog.text

    def test_env_can_be_closed(self) -> None:
        self.app.env.close()
