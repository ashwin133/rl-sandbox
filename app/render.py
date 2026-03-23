from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.figure import Figure


class BaseRenderer(ABC):
    """Base class for rendering data."""

    @abstractmethod
    def render_from_frames(self, frames: list[np.ndarray]) -> None:
        """Render an animation from a list of frames."""
        ...

    @abstractmethod
    def start_live_render(self) -> Figure:
        """Start a live rendering session, returning the figure."""
        ...

    @abstractmethod
    def continuously_live_render(
        self, fig: Figure, update_frame_callback: Callable[[int], list[Any]]
    ) -> None:
        """Continuously render frames in real-time."""
        ...


class MatplotlibAnimator(BaseRenderer):
    """Renderer that uses Matplotlib to render frames."""

    def update_frame(self, frame: np.ndarray) -> list[Any]:
        """Update the animation with a new frame."""
        img = plt.imshow(frame)
        return [img]

    def render_from_frames(self, frames: list[np.ndarray]) -> None:
        """Render an animation from a list of frames."""
        fig = plt.figure(figsize=(5, 5))
        plt.axis("off")

        _ani = animation.FuncAnimation(
            fig,
            self.update_frame,
            frames=frames,
            interval=50,
            blit=True,
        )
        plt.show()

    def start_live_render(self) -> Figure:
        fig = plt.figure(figsize=(5, 5))
        plt.axis("off")
        return fig

    def continuously_live_render(
        self, fig: Figure, update_frame_callback: Callable[[int], list[Any]]
    ) -> None:
        """Render frames continuously using FuncAnimation."""
        _ani = animation.FuncAnimation(
            fig,
            update_frame_callback,
            frames=None,
            interval=50,
            blit=True,
        )
        plt.show()
