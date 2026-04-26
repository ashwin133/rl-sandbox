from pathlib import Path

import gymnasium as gym
import pytest


class TestPolicyImports:
    """Verify policy package imports resolve correctly."""

    def test_policy_import(self) -> None:
        from app.policy.base import BaseActionPolicy
        from app.policy.random import RandomActionPolicy

        assert BaseActionPolicy is not None
        assert RandomActionPolicy is not None

    def test_base_policy_is_abstract(self) -> None:
        from app.policy.base import BaseActionPolicy

        with pytest.raises(TypeError):
            BaseActionPolicy()  # type: ignore[abstract]

    def test_random_policy_returns_valid_action(self) -> None:
        from app.policy.random import RandomActionPolicy

        policy = RandomActionPolicy()
        env = gym.make("CartPole-v1")
        action = policy.choose_action(env.action_space, env.observation_space.sample())
        assert env.action_space.contains(action)
        env.close()


class TestPlotImports:
    """Verify plot module imports resolve correctly."""

    def test_plot_import(self) -> None:
        from app.plot import BasePlotter, MatplotlibPlotter

        assert BasePlotter is not None
        assert MatplotlibPlotter is not None

    def test_base_plotter_is_abstract(self) -> None:
        from app.plot import BasePlotter

        with pytest.raises(TypeError):
            BasePlotter()  # type: ignore[abstract]

    def test_matplotlib_plotter_instantiates(self) -> None:
        from app.plot import MatplotlibPlotter

        plotter = MatplotlibPlotter()
        assert plotter is not None


class TestRenderImports:
    """Verify render module imports resolve correctly."""

    def test_render_import(self) -> None:
        from app.render import BaseRenderer, MatplotlibAnimator

        assert BaseRenderer is not None
        assert MatplotlibAnimator is not None

    def test_base_renderer_is_abstract(self) -> None:
        from app.render import BaseRenderer

        with pytest.raises(TypeError):
            BaseRenderer()  # type: ignore[abstract]

    def test_matplotlib_animator_instantiates(self) -> None:
        from app.render import MatplotlibAnimator

        animator = MatplotlibAnimator()
        assert animator is not None


class TestOldFileRemoved:
    """Verify the old action_policy.py has been deleted."""

    def test_old_action_policy_removed(self) -> None:
        assert not Path("app/action_policy.py").exists()


class TestViewerComposition:
    """Verify CartPoleViewer composes the separated modules."""

    def test_viewer_import(self) -> None:
        from app.viewer import CartPoleViewer, ViewerConfig

        assert CartPoleViewer is not None
        assert ViewerConfig is not None

    def test_viewer_composes_correctly(self) -> None:
        import gymnasium as gym

        from app.plot import BasePlotter
        from app.render import BaseRenderer
        from app.viewer import CartPoleViewer

        env = gym.make("CartPole-v1", render_mode="rgb_array")
        env.reset()
        viewer = CartPoleViewer(env=env)
        assert isinstance(viewer.plotter, BasePlotter)
        assert isinstance(viewer.renderer, BaseRenderer)
        assert viewer.env is env
        env.close()

    def test_cartpole_uses_new_policy_path(self) -> None:
        from app.cartpole import CartPoleApp

        app = CartPoleApp()
        assert app is not None
        app.env.close()
