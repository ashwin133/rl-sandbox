from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.render import BaseRenderer, MatplotlibAnimator


def _mock_subplots(mock_plt: MagicMock) -> MagicMock:
    """Configure mock plt.subplots to return (mock_fig, mock_ax)."""
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    return mock_fig


class TestBaseRenderer:
    """Unit tests for BaseRenderer."""

    def test_base_renderer_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseRenderer()  # type: ignore[abstract]


class TestMatplotlibAnimator:
    """Unit tests for MatplotlibAnimator."""

    @patch("app.render.plt")
    @patch("app.render.animation.FuncAnimation")
    def test_render_from_frames_calls_func_animation(
        self, mock_anim: MagicMock, mock_plt: MagicMock
    ) -> None:
        animator = MatplotlibAnimator()
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
        _mock_subplots(mock_plt)
        animator.render_from_frames(frames)
        mock_anim.assert_called_once()

    @patch("app.render.plt")
    @patch("app.render.animation.FuncAnimation")
    def test_render_from_frames_calls_show(
        self, mock_anim: MagicMock, mock_plt: MagicMock
    ) -> None:
        animator = MatplotlibAnimator()
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        _mock_subplots(mock_plt)
        animator.render_from_frames(frames)
        mock_plt.show.assert_called_once()

    @patch("app.render.plt")
    @patch("app.render.animation.FuncAnimation")
    def test_start_live_render_calls_func_animation(
        self, mock_anim: MagicMock, mock_plt: MagicMock
    ) -> None:
        import asyncio

        animator = MatplotlibAnimator()
        callback = MagicMock()
        _mock_subplots(mock_plt)

        async def run() -> None:
            task = asyncio.create_task(animator.start_live_render(callback))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run())
        mock_anim.assert_called_once()

    def test_update_frame_returns_list(self) -> None:
        animator = MatplotlibAnimator()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = animator.update_frame(frame)
        assert isinstance(result, list)
        assert len(result) == 1
