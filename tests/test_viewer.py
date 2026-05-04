from collections.abc import Generator
from unittest.mock import MagicMock, patch

import gymnasium as gym
import pytest

from rl_sandbox.plot import BasePlotter
from rl_sandbox.render import BaseRenderer
from rl_sandbox.viewer import EnvViewer, ViewerConfig


class TestViewerConfig:
    """Unit tests for ViewerConfig."""

    def test_defaults(self) -> None:
        config = ViewerConfig()
        assert config.tick_length_ms == 100.0


class TestEnvViewer:
    """Unit tests for EnvViewer."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> Generator[None, None, None]:
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")
        self.env.reset()
        yield
        self.env.close()

    def test_init_creates_components(self) -> None:
        viewer = EnvViewer(env=self.env)
        assert viewer.env is self.env
        assert isinstance(viewer.plotter, BasePlotter)
        assert isinstance(viewer.renderer, BaseRenderer)
        assert isinstance(viewer.config, ViewerConfig)

    @patch.object(BasePlotter, "__abstractmethods__", set())
    def test_plot_calls_plotter(self) -> None:
        viewer = EnvViewer(env=self.env)
        viewer.plotter = MagicMock(spec=BasePlotter)
        viewer.plot()
        viewer.plotter.plot.assert_called_once()

    def test_simulate_and_render_collects_frames(self) -> None:
        viewer = EnvViewer(env=self.env)
        viewer.renderer = MagicMock(spec=BaseRenderer)
        sequence_length = 5
        viewer.simulate_and_render_sequence(sequence_length=sequence_length)
        viewer.renderer.render_from_frames.assert_called_once()
        frames = viewer.renderer.render_from_frames.call_args[0][0]
        assert len(frames) == sequence_length

    def test_simulate_defaults_to_random_policy(self) -> None:
        viewer = EnvViewer(env=self.env)
        viewer.renderer = MagicMock(spec=BaseRenderer)
        viewer.simulate_and_render_sequence(sequence_length=3)
        viewer.renderer.render_from_frames.assert_called_once()
        frames = viewer.renderer.render_from_frames.call_args[0][0]
        assert len(frames) == 3
