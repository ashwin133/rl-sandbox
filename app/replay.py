"""Replay runner — plays back recorded episode frames."""

import asyncio
import logging

from app.cartpole import MODEL_TICK_BETWEEN_ACTIONS_S
from app.models.recording import EpisodeRecording
from app.models.replay import BatchReplay, EpisodeReplay
from app.render import BaseRenderer

logger = logging.getLogger(__name__)


class ReplayRunner:
    """Plays back recorded episode frames via a renderer.

    No dependency on CartPoleApp, gymnasium, or asyncio.Lock —
    replay is pure frame playback.
    """

    def __init__(self, renderer: BaseRenderer) -> None:
        self._renderer = renderer

    async def replay_episode(self, episode: EpisodeReplay) -> None:
        """Play back all frames for a single episode at its recorded speed."""
        meta: EpisodeRecording = episode.metadata
        tick: float = MODEL_TICK_BETWEEN_ACTIONS_S / meta.sim_speed_multiplier

        logger.info(
            f"Replaying episode {meta.episode_index} ({meta.timesteps_reached} steps, {meta.sim_speed_multiplier:.1f}x speed)"
        )

        self._renderer.init_display()
        for frame in episode.frames:
            self._renderer.display_frame(frame)
            await asyncio.sleep(tick)

        logger.info(f"Episode {meta.episode_index} complete.")

    async def replay_batch(self, batch: BatchReplay) -> None:
        """Play back all episodes in a batch sequentially."""
        logger.info(
            f"Replaying {len(batch.episodes)} of {batch.total_source_episodes} episodes (replay_freq={batch.replay_freq}) from {batch.source_path}"
        )
        for episode in batch.episodes:
            await self.replay_episode(episode)
        logger.info("Batch replay complete.")
