import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class BaseRenderer(ABC):
    """Base class for rendering data."""

    @abstractmethod
    def render_from_frames(self, frames: list[np.ndarray]) -> None:
        """Render an animation from a list of frames."""
        ...

    @abstractmethod
    def update_frame(self, frame: np.ndarray) -> list[Any]:
        """Update the current display with a new frame."""
        ...

    @abstractmethod
    async def start_live_render(
        self, update_frame_callback: Callable[[int], list[Any]]
    ) -> None:
        """Start a live rendering session"""
        ...


class MatplotlibAnimator(BaseRenderer):
    """Renderer that uses Matplotlib to render frames."""

    def __init__(self) -> None:
        self._fig: Figure | None = None
        self._ax: Axes | None = None
        self._img: Any | None = None

    def _ensure_fig(self) -> tuple[Figure, Axes]:
        """Return the existing (fig, ax) or create a new one."""
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots(figsize=(5, 5))
            self._ax.axis("off")
        return self._fig, self._ax

    def update_frame(self, frame: np.ndarray) -> list[Any]:
        """Update the animation with a new frame."""
        _, ax = self._ensure_fig()
        if self._img is None:
            self._img = ax.imshow(frame)
        else:
            self._img.set_data(frame)
        return [self._img]

    def render_from_frames(self, frames: list[np.ndarray]) -> None:
        """Render an animation from a list of frames."""
        fig, _ax = self._ensure_fig()
        _ani = animation.FuncAnimation(
            fig,
            self.update_frame,
            frames=frames,
            interval=50,
            blit=True,
        )
        plt.show()

    async def start_live_render(
        self, update_frame_callback: Callable[[int], list[Any]]
    ) -> None:
        """Render frames continuously using FuncAnimation."""
        fig, _ax = self._ensure_fig()
        _ani = animation.FuncAnimation(
            fig,
            update_frame_callback,
            frames=None,
            interval=25,
            blit=True,
            cache_frame_data=False,
        )
        plt.show(block=False)

        while True:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            await asyncio.sleep(0.001)
