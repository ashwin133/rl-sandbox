"""Integration tests for training pipeline."""

import gymnasium as gym
import numpy as np

from app.envs.registry import SUPPORTED_ENVS, EnvSpec
from app.models.training import TrainingConfig
from app.policy.reinforce import ReinforcePolicy
from app.training.reinforce import ReinforceTrainer

CARTPOLE_SPEC: EnvSpec = SUPPORTED_ENVS["CartPole-v1"]


class TestReinforceIntegration:
    """Integration test: train REINFORCE, save, load, run in env."""

    def test_train_save_load_run(self, tmp_path: "np.str_") -> None:
        env: gym.Env = gym.make("CartPole-v1")
        config: TrainingConfig = TrainingConfig(
            num_episodes=10, learning_rate=1e-3
        )
        trainer: ReinforceTrainer = ReinforceTrainer(
            env=env, config=config, spec=CARTPOLE_SPEC
        )
        result = trainer.train()
        assert len(result.episodes) == 10

        model_path = tmp_path / "reinforce.pt"  # type: ignore[operator]
        trainer.save_model(model_path)

        policy: ReinforcePolicy = ReinforcePolicy.load(
            model_path, expected_env_id="CartPole-v1"
        )
        obs, _info = env.reset()
        for _ in range(10):
            action: int = policy.choose_action(env.action_space, obs)
            assert env.action_space.contains(action)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()
