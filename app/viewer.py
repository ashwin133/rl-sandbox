import logging
from typing import Any

import matplotlib.pyplot as plt
from pydantic import BaseModel

from app.cartpole import CartPoleApp
from app.plot import BasePlotter, MatplotlibPlotter
from app.policy import BaseActionPolicy, RandomActionPolicy
from app.render import BaseRenderer, MatplotlibAnimator

logger = logging.getLogger(__name__)


class ViewerConfig(BaseModel):
    """Configuration for CartPoleViewer."""

    tick_length_ms: float = 100.0


class CartPoleViewer:
    """Allows one to visualize the CartPole environment."""

    def __init__(self) -> None:
        self.app = CartPoleApp()
        self.plotter: BasePlotter = MatplotlibPlotter()
        self.renderer: BaseRenderer = MatplotlibAnimator()
        self.config = ViewerConfig()

    def plot(self) -> None:
        """Renders the CartPole environment."""
        frame = self.app.env.render()
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
            self.app.apply_action(action_policy)
            frame = self.app.env.render()
            frames.append(frame)

        self.renderer.render_from_frames(frames)  # type: ignore[arg-type]

    def render_live(
        self,
        action_policy: BaseActionPolicy,
    ) -> None:
        fig = self.renderer.start_live_render()

        def update_frame(frame_num: int) -> list[Any]:
            self.app.apply_action(action_policy)
            frame = self.app.env.render()
            return [plt.imshow(frame)]  # type: ignore[arg-type]

        self.renderer.continuously_live_render(fig, update_frame)
