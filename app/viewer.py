import asyncio
import logging
from typing import Any

import gymnasium
from pydantic import BaseModel

from app.plot import BasePlotter, MatplotlibPlotter
from app.policy import BaseActionPolicy, RandomActionPolicy
from app.render import BaseRenderer, MatplotlibAnimator

logger = logging.getLogger(__name__)


class ViewerConfig(BaseModel):
    """Configuration for CartPoleViewer."""

    tick_length_ms: float = 100.0


class CartPoleViewer:
    """Allows one to visualize the CartPole environment."""

    def __init__(self, env: gymnasium.Env, lock: asyncio.Lock | None = None) -> None:
        self.env = env
        self.lock = lock or asyncio.Lock()
        self.plotter: BasePlotter = MatplotlibPlotter()
        self.renderer: BaseRenderer = MatplotlibAnimator()
        self.config = ViewerConfig()
        self._latest_frame = self.env.render()

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
        frames = []
        for _ in range(sequence_length):
            frame = self.env.render()
            frames.append(frame)

        self.renderer.render_from_frames(frames)  # type: ignore[arg-type]

    async def render_live(self) -> None:
        async def refresh_frame() -> None:
            """Periodically snapshot the env frame under the lock."""
            while True:
                async with self.lock:
                    self._latest_frame = self.env.render()
                await asyncio.sleep(0.02)  # ~50 fps

        def update_frame(frame_num: int) -> list[Any]:
            """FuncAnimation callback — reads the latest cached frame."""
            return self.renderer.update_frame(self._latest_frame)  # type: ignore[arg-type]

        asyncio.ensure_future(refresh_frame())
        await self.renderer.start_live_render(update_frame)
