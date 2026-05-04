from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# TODO: Need to add logic to enforce the constraint below
RENDER_DELAY_MS = 3  # Needs to be < sim tick time to avoid laggy rendering


class BaseRenderer(ABC):
    """Base class for rendering data."""

    @abstractmethod
    def quit(self) -> None:
        """Signal the live rendering loop to quit."""
        ...

    @abstractmethod
    def render_from_frames(self, frames: list[np.ndarray]) -> None:
        """Render an animation from a list of frames."""
        ...

    @abstractmethod
    def update_frame(self, frame: np.ndarray) -> list[Any]:
        """Update the current display with a new frame."""
        ...

    @abstractmethod
    def init_display(self) -> None:
        """Initialise the display window (non-blocking). Idempotent."""
        ...

    @abstractmethod
    def display_frame(self, frame: np.ndarray) -> None:
        """Update display with a single frame and pump the event loop."""
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
        self._quit_requested: bool = False

    def quit(self) -> None:
        """Signal the live rendering loop to quit."""
        self._quit_requested = True

    def _ensure_fig(self) -> tuple[Figure, Axes]:
        """Return the existing (fig, ax) or create a new one."""
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots(figsize=(5, 5))
            self._ax.axis("off")
        return self._fig, self._ax

    def init_display(self) -> None:
        """Initialise the display window (non-blocking). Idempotent."""
        self._ensure_fig()
        plt.show(block=False)

    def display_frame(self, frame: np.ndarray) -> None:
        """Update display with a single frame and pump the event loop."""
        fig, _ = self._ensure_fig()
        self.update_frame(frame)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

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
            interval=RENDER_DELAY_MS,
            blit=True,
            cache_frame_data=False,
        )
        plt.show(block=False)

        while not self._quit_requested:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            await asyncio.sleep(RENDER_DELAY_MS / 1000)
        logger.info("Quit requested. Stopping live render.")
