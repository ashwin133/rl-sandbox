"""Tests for GymEnvApp — must work for every supported env."""

import asyncio

import numpy as np
import pytest

from app.envs.gym_env import GymEnvApp
from app.envs.registry import SUPPORTED_ENVS, EnvSpec
from app.policy.random import RandomActionPolicy


@pytest.mark.parametrize("env_id", list(SUPPORTED_ENVS.keys()))
class TestGymEnvAppPerEnv:
    def test_constructs_and_resets(self, env_id: str) -> None:
        spec: EnvSpec = SUPPORTED_ENVS[env_id]
        app: GymEnvApp = GymEnvApp(spec)
        assert app.spec is spec
        assert app.env is not None
        assert isinstance(app.lock, asyncio.Lock)
        assert app.current_obs is not None
        app.env.close()

    def test_apply_action_advances_obs(self, env_id: str) -> None:
        spec: EnvSpec = SUPPORTED_ENVS[env_id]
        app: GymEnvApp = GymEnvApp(spec)
        prev_obs: np.ndarray = app.current_obs.copy()
        result = app.apply_action(RandomActionPolicy())
        assert result.obs is not None
        assert app.current_obs is result.obs
        assert not np.array_equal(prev_obs, app.current_obs)
        app.env.close()

    def test_reset_refreshes_current_obs(self, env_id: str) -> None:
        spec: EnvSpec = SUPPORTED_ENVS[env_id]
        app: GymEnvApp = GymEnvApp(spec)
        app.apply_action(RandomActionPolicy())
        before_reset: np.ndarray = app.current_obs.copy()
        app.reset_env()
        # Reset usually produces a different obs (random initial state)
        assert app.current_obs is not None
        # Check shape is consistent
        assert app.current_obs.shape == before_reset.shape
        app.env.close()
