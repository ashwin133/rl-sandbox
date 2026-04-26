"""Integration tests for training pipeline."""

import gymnasium as gym
import numpy as np

from app.models.training import TrainingConfig
from app.policy.reinforce import ReinforcePolicy
from app.training.reinforce import ReinforceTrainer

CARTPOLE_OBS_SIZE = 4
CARTPOLE_ACTION_SIZE = 2


class TestReinforceIntegration:
    """Integration test: train REINFORCE, save, load, run in env."""

    def test_train_save_load_run(self, tmp_path: "np.str_") -> None:
        env: gym.Env = gym.make("CartPole-v1")
        config: TrainingConfig = TrainingConfig(
            num_episodes=10, learning_rate=1e-3
        )
        trainer: ReinforceTrainer = ReinforceTrainer(env=env, config=config)
        result = trainer.train()
        assert len(result.episodes) == 10

        # Save model
        model_path = tmp_path / "reinforce.pt"  # type: ignore[operator]
        trainer.save_model(model_path)

        # Load as policy and run in env
        policy: ReinforcePolicy = ReinforcePolicy.load(
            model_path,
            obs_size=CARTPOLE_OBS_SIZE,
            action_size=CARTPOLE_ACTION_SIZE,
        )
        obs, _info = env.reset()
        for _ in range(10):
            action: int = policy.choose_action(env.action_space, env.observation_space)
            assert env.action_space.contains(action)
        env.close()
