"""Replay runner — plays back recorded episode frames."""

import asyncio
import logging

from rl_sandbox.models.recording import EpisodeRecording
from rl_sandbox.models.replay import BatchReplay, EpisodeReplay
from rl_sandbox.render import BaseRenderer

logger = logging.getLogger(__name__)


class ReplayRunner:
    """Plays back recorded episode frames via a renderer.

    No dependency on a live env, gymnasium, or asyncio.Lock — replay is
    pure frame playback. The physics tick comes from the EnvSpec embedded
    in the recording.
    """

    def __init__(self, renderer: BaseRenderer) -> None:
        self._renderer = renderer

    async def replay_episode(
        self, episode: EpisodeReplay, physics_tick_s: float
    ) -> None:
        """Play back all frames for a single episode at its recorded speed."""
        meta: EpisodeRecording = episode.metadata
        tick: float = physics_tick_s / meta.sim_speed_multiplier

        logger.info(
            f"Replaying episode {meta.episode_index} "
            f"({meta.timesteps_reached} steps, {meta.sim_speed_multiplier:.1f}x speed)"
        )

        self._renderer.init_display()
        for frame in episode.frames:
            self._renderer.display_frame(frame)
            await asyncio.sleep(tick)

        logger.info(f"Episode {meta.episode_index} complete.")

    async def replay_batch(self, batch: BatchReplay) -> None:
        """Play back all episodes in a batch sequentially."""
        physics_tick_s: float = batch.env_spec.physics_tick_s
        logger.info(
            f"Replaying {len(batch.episodes)} of {batch.total_source_episodes} episodes "
            f"(replay_freq={batch.replay_freq}, env={batch.env_spec.env_id}) "
            f"from {batch.source_path}"
        )
        for episode in batch.episodes:
            await self.replay_episode(episode, physics_tick_s)
        logger.info("Batch replay complete.")
