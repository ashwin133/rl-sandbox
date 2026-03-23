from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.plot import BasePlotter, MatplotlibPlotter


class TestBasePlotter:
    """Unit tests for BasePlotter."""

    def test_base_plotter_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BasePlotter()  # type: ignore[abstract]


class TestMatplotlibPlotter:
    """Unit tests for MatplotlibPlotter."""

    @patch("app.plot.plt")
    def test_calls_imshow(self, mock_plt: MagicMock) -> None:
        plotter = MatplotlibPlotter()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        plotter.plot(frame)
        mock_plt.imshow.assert_called_once_with(frame)

    @patch("app.plot.plt")
    def test_calls_show(self, mock_plt: MagicMock) -> None:
        plotter = MatplotlibPlotter()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        plotter.plot(frame)
        mock_plt.show.assert_called_once()
