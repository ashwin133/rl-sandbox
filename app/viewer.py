import asyncio
import logging
from typing import Any

import gymnasium
from pydantic import BaseModel

from app.plot import BasePlotter, MatplotlibPlotter
from app.policy.base import BaseActionPolicy
from app.policy.random import RandomActionPolicy
from app.render import BaseRenderer, MatplotlibAnimator

logger = logging.getLogger(__name__)


class ViewerConfig(BaseModel):
    """Configuration for CartPoleViewer."""

    tick_length_ms: float = 100.0


class CartPoleViewer:
    """Allows one to visualize the CartPole environment."""

    def __init__(self, env: gymnasium.Env, lock: asyncio.Lock | None = None) -> None:
        self.env: gymnasium.Env = env
        self._quit_requested: bool = False
        self.lock: asyncio.Lock = lock or asyncio.Lock()
        self.plotter: BasePlotter = MatplotlibPlotter()
        self.renderer: BaseRenderer = MatplotlibAnimator()
        self.config: ViewerConfig = ViewerConfig()
        self._latest_frame: Any = self.env.render()

    def quit(self) -> None:
        """Signal the live rendering loop to quit."""
        self._quit_requested = True
        self.renderer.quit()

    def plot(self) -> None:
        """Renders the CartPole environment."""
        frame = self.env.render()
        self.plotter.plot(frame)  # type: ignore[arg-type]

    def simulate_and_render_sequence(
        self,
        sequence_length: int = 100,
        action_policy: BaseActionPolicy | None = None,
    ) -> None:
        if action_policy is None:
            action_policy = RandomActionPolicy()
        frames: list[Any] = []
        for _ in range(sequence_length):
            frame: Any = self.env.render()
            frames.append(frame)

        self.renderer.render_from_frames(frames)  # type: ignore[arg-type]

    async def render_live(self) -> None:
        async def refresh_frame() -> None:
            """Periodically snapshot the env frame under the lock."""
            while not self._quit_requested:
                async with self.lock:
                    self._latest_frame = self.env.render()
                await asyncio.sleep(0.02)  # ~50 fps
            logger.info("Quit requested. Stopping refresh frame task.")

        def update_frame(frame_num: int) -> list[Any]:
            """FuncAnimation callback — reads the latest cached frame."""
            return self.renderer.update_frame(self._latest_frame)  # type: ignore[arg-type]

        asyncio.ensure_future(refresh_frame())
        await self.renderer.start_live_render(update_frame)
