from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from matplotlib.figure import Figure

from app.render import BaseRenderer, MatplotlibAnimator


class TestBaseRenderer:
    """Unit tests for BaseRenderer."""

    def test_base_renderer_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseRenderer()  # type: ignore[abstract]


class TestMatplotlibAnimator:
    """Unit tests for MatplotlibAnimator."""

    def test_start_live_render_returns_figure(self) -> None:
        animator = MatplotlibAnimator()
        fig = animator.start_live_render()
        assert isinstance(fig, Figure)

    @patch("app.render.plt")
    @patch("app.render.animation.FuncAnimation")
    def test_render_from_frames_calls_func_animation(
        self, mock_anim: MagicMock, mock_plt: MagicMock
    ) -> None:
        animator = MatplotlibAnimator()
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
        mock_plt.figure.return_value = MagicMock(spec=Figure)
        animator.render_from_frames(frames)
        mock_anim.assert_called_once()

    @patch("app.render.plt")
    @patch("app.render.animation.FuncAnimation")
    def test_render_from_frames_calls_show(
        self, mock_anim: MagicMock, mock_plt: MagicMock
    ) -> None:
        animator = MatplotlibAnimator()
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        mock_plt.figure.return_value = MagicMock(spec=Figure)
        animator.render_from_frames(frames)
        mock_plt.show.assert_called_once()

    @patch("app.render.plt")
    @patch("app.render.animation.FuncAnimation")
    def test_continuously_live_render_calls_func_animation(
        self, mock_anim: MagicMock, mock_plt: MagicMock
    ) -> None:
        animator = MatplotlibAnimator()
        fig = MagicMock(spec=Figure)
        callback = MagicMock()
        animator.continuously_live_render(fig, callback)
        mock_anim.assert_called_once()

    def test_update_frame_returns_list(self) -> None:
        animator = MatplotlibAnimator()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = animator.update_frame(frame)
        assert isinstance(result, list)
        assert len(result) == 1
