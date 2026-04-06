"""Episode runner — runs episodes against a CartPoleApp."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.cartpole import MODEL_TICK_BETWEEN_ACTIONS_S, CartPoleApp
from app.models.batch import BatchConfig, BatchResult
from app.models.episode import EpisodeConfig, EpisodeResult, StepResult
from app.policy import BaseActionPolicy

logger = logging.getLogger(__name__)


class EpisodeRunner:
    """Runs episodes against a CartPoleApp.

    Does not store policy or config — receives them per-call for clarity.
    All run methods are async and acquire app.lock around env access.
    """

    def __init__(self, app: CartPoleApp) -> None:
        self._app = app
        self._current_episode: int = 0
        self._total_steps: int = 0
        self._episode_steps: int = 0

    @property
    def current_episode(self) -> int:
        return self._current_episode

    def increment_episode(self) -> None:
        self._current_episode += 1

    @property
    def total_steps(self) -> int:
        return self._total_steps

    def increment_total_steps(self) -> None:
        self._total_steps += 1

    @property
    def episode_steps(self) -> int:
        return self._episode_steps

    def increment_episode_steps(self) -> None:
        self._episode_steps += 1

    def reset_episode_steps(self) -> None:
        self._episode_steps = 0

    def on_step_transition(self) -> None:
        self.increment_episode_steps()
        self.increment_total_steps()

    def _check_max_steps_reached(self, config: EpisodeConfig) -> bool:
        if config.max_steps is not None and self._episode_steps >= config.max_steps:
            logger.info(
                "Episode %d hit max_steps (%d).",
                self._current_episode,
                config.max_steps,
            )
            return True
        return False

    async def run_episode(
        self,
        policy: BaseActionPolicy,
        config: EpisodeConfig,
    ) -> EpisodeResult:
        """Run a single episode, stepping env under the lock until done or max_steps."""
        sequence: list[StepResult] = []
        self.reset_episode_steps()

        async with self._app.lock:
            self._app.reset_env()

        episode_done: bool = False
        while not episode_done:
            async with self._app.lock:
                step_result = self._app.apply_action(policy)

            self.on_step_transition()
            sequence.append(step_result)
            if step_result.done:
                logger.info(
                    "Episode %d ended after %d steps.",
                    self._current_episode,
                    self._episode_steps,
                )
            episode_done = self._check_max_steps_reached(config) or step_result.done

            tick = MODEL_TICK_BETWEEN_ACTIONS_S / config.sim_speed_multiplier
            await asyncio.sleep(tick)

        self.increment_episode()
        return EpisodeResult(
            sequence=sequence,
            timestamp=datetime.now(tz=timezone.utc),
            policy_name=type(policy).__name__,
            episode_config=config,
            timesteps_reached=len(sequence),
        )

    async def run_batch(
        self, policy: BaseActionPolicy, config: BatchConfig
    ) -> BatchResult:
        """Run a batch of episodes sequentially."""
        episodes: list[EpisodeResult] = []
        for _ in range(config.no_episodes):
            result = await self.run_episode(policy, config.episode_config)
            episodes.append(result)
        return BatchResult(episodes=episodes, batch_config=config)

    def log_batch_to_file(self, batch_result: BatchResult, path: Path) -> bool:
        """Save batch results to disk.

        Writes step_data.npz (numpy arrays) and metadata.json to the given directory.
        """
        path.mkdir(parents=True, exist_ok=True)

        # Build numpy arrays keyed by episode index
        arrays: dict[str, np.ndarray] = {}
        for i, ep in enumerate(batch_result.episodes):
            obs_list = [step.obs for step in ep.sequence]
            arrays[f"episode_{i}_obs"] = np.array(obs_list)
            arrays[f"episode_{i}_reward"] = np.array(
                [step.reward for step in ep.sequence]
            )
            arrays[f"episode_{i}_terminated"] = np.array(
                [step.terminated for step in ep.sequence]
            )
            arrays[f"episode_{i}_truncated"] = np.array(
                [step.truncated for step in ep.sequence]
            )

        np.savez(path / "step_data.npz", **arrays)

        # Build JSON metadata
        metadata = {
            "batch_config": batch_result.batch_config.model_dump(),
            "episodes": [
                {
                    "timestamp": ep.timestamp.isoformat(),
                    "policy_name": ep.policy_name,
                    "episode_config": ep.episode_config.model_dump(),
                    "timesteps_reached": ep.timesteps_reached,
                }
                for ep in batch_result.episodes
            ],
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return True

    @staticmethod
    def load_batch_from_file(path: Path) -> BatchResult:
        """Load batch results from disk, reconstructing from .npz + .json."""
        data = np.load(path / "step_data.npz")
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        batch_config = BatchConfig(**metadata["batch_config"])

        episodes: list[EpisodeResult] = []
        for i, ep_meta in enumerate(metadata["episodes"]):
            obs_arr = data[f"episode_{i}_obs"]
            reward_arr = data[f"episode_{i}_reward"]
            terminated_arr = data[f"episode_{i}_terminated"]
            truncated_arr = data[f"episode_{i}_truncated"]

            sequence = [
                StepResult(
                    obs=obs_arr[j].tolist(),
                    reward=float(reward_arr[j]),
                    terminated=bool(terminated_arr[j]),
                    truncated=bool(truncated_arr[j]),
                    done=bool(terminated_arr[j] or truncated_arr[j]),
                )
                for j in range(len(reward_arr))
            ]

            episodes.append(
                EpisodeResult(
                    sequence=sequence,
                    timestamp=datetime.fromisoformat(ep_meta["timestamp"]),
                    policy_name=ep_meta["policy_name"],
                    episode_config=EpisodeConfig(**ep_meta["episode_config"]),
                    timesteps_reached=ep_meta["timesteps_reached"],
                )
            )

        return BatchResult(episodes=episodes, batch_config=batch_config)
