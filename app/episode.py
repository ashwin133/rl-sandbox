"""Episode runner — runs episodes against a CartPoleApp."""

import asyncio
import logging
from datetime import datetime, timezone

import numpy as np

from app.cartpole import MODEL_TICK_BETWEEN_ACTIONS_S, CartPoleApp
from app.models.batch import BatchConfig, BatchResult
from app.models.episode import EpisodeConfig, EpisodeResult, StepResult
from app.policy.base import BaseActionPolicy

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
                f"Episode {self._current_episode} hit max_steps ({config.max_steps})."
            )
            return True
        return False

    async def run_episode(
        self,
        policy: BaseActionPolicy,
        config: EpisodeConfig,
        capture_frames: bool = False,
    ) -> tuple[EpisodeResult, list[np.ndarray] | None]:
        """Run a single episode, stepping env under the lock until done or max_steps.

        When capture_frames is True, grabs env.render() under the same lock as
        env.step() at each step and returns the list of RGB frames. Otherwise
        returns None for the frames component.
        """
        sequence: list[StepResult] = []
        frames: list[np.ndarray] = [] if capture_frames else []
        self.reset_episode_steps()

        async with self._app.lock:
            self._app.reset_env()

        episode_done: bool = False
        while not episode_done:
            async with self._app.lock:
                step_result = self._app.apply_action(policy)
                if capture_frames:
                    frame: np.ndarray = self._app.env.render()  # type: ignore[assignment]
                    frames.append(frame)

            self.on_step_transition()
            sequence.append(step_result)
            if step_result.done:
                logger.info(
                    f"Episode {self._current_episode} ended after {self._episode_steps} steps."
                )
            episode_done = self._check_max_steps_reached(config) or step_result.done

            tick = MODEL_TICK_BETWEEN_ACTIONS_S / config.sim_speed_multiplier
            await asyncio.sleep(tick)

        self.increment_episode()
        result = EpisodeResult(
            sequence=sequence,
            timestamp=datetime.now(tz=timezone.utc),
            policy_name=type(policy).__name__,
            episode_config=config,
            timesteps_reached=len(sequence),
        )
        return result, frames if capture_frames else None

    async def run_batch(
        self, policy: BaseActionPolicy, config: BatchConfig
    ) -> tuple[BatchResult, dict[int, list[np.ndarray]]]:
        """Run a batch of episodes sequentially.

        Returns the BatchResult and a dict mapping episode index to captured
        frames for episodes selected by config.replay_freq.
        """
        episodes: list[EpisodeResult] = []
        captured_frames: dict[int, list[np.ndarray]] = {}
        for i in range(config.no_episodes):
            should_capture = i % config.replay_freq == 0
            result, frames = await self.run_episode(
                policy, config.episode_config, capture_frames=should_capture
            )
            episodes.append(result)
            if frames is not None:
                captured_frames[i] = frames
        return BatchResult(episodes=episodes, batch_config=config), captured_frames
