"""Tests for the env registry — every entry must be a real, loadable gym env."""

import gymnasium as gym
import pytest

from app.envs.registry import SUPPORTED_ENVS, EnvSpec


class TestEnvSpec:
    def test_minimal_construction(self) -> None:
        spec: EnvSpec = EnvSpec(
            env_id="CartPole-v1",
            physics_tick_s=0.02,
            solved_reward_threshold=475.0,
        )
        assert spec.env_id == "CartPole-v1"
        assert spec.rolling_window == 100
        assert spec.default_max_steps == 500


class TestSupportedEnvs:
    def test_registry_non_empty(self) -> None:
        assert len(SUPPORTED_ENVS) > 0

    def test_keys_match_env_ids(self) -> None:
        for key, spec in SUPPORTED_ENVS.items():
            assert key == spec.env_id, (
                f"Registry key {key!r} does not match spec.env_id {spec.env_id!r}"
            )

    @pytest.mark.parametrize("env_id", list(SUPPORTED_ENVS.keys()))
    def test_each_env_id_is_loadable(self, env_id: str) -> None:
        env = gym.make(env_id)
        assert env is not None
        env.close()

    def test_cartpole_present(self) -> None:
        assert "CartPole-v1" in SUPPORTED_ENVS
