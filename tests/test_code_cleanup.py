import inspect
import logging
from collections.abc import Generator
from pathlib import Path

import pytest

from app.cartpole import CartPoleApp
from app.episode import EpisodeRunner
from app.file_ops import FileOps
from app.logging_config import LoggerFactory, LoggingConfig
from app.models.episode import StepResult
from app.plot import MatplotlibPlotter
from app.policy.random import RandomActionPolicy
from app.render import MatplotlibAnimator
from app.replay import ReplayRunner
from app.utilities import Utilities
from app.viewer import CartPoleViewer


class TestReturnTypeAnnotations:
    """Verify all public methods have return type annotations."""

    @pytest.mark.parametrize(
        "cls",
        [
            CartPoleApp,
            CartPoleViewer,
            MatplotlibPlotter,
            MatplotlibAnimator,
            RandomActionPolicy,
            LoggerFactory,
            EpisodeRunner,
            FileOps,
            ReplayRunner,
            Utilities,
        ],
    )
    def test_all_public_methods_have_return_types(self, cls: type) -> None:
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if name.startswith("_"):
                continue
            hints = method.__annotations__
            assert "return" in hints, (
                f"{cls.__name__}.{name} is missing a return type annotation"
            )

    def test_cartpole_init_returns_none(self) -> None:
        hints = CartPoleApp.__init__.__annotations__
        assert hints.get("return") is None

    def test_apply_action_returns_step_result(self) -> None:
        hints = CartPoleApp.apply_action.__annotations__
        assert hints.get("return") is StepResult


class TestFileHandlerFormat:
    """Verify the file handler format string has proper spacing."""

    @pytest.fixture(autouse=True)
    def _setup_logger(self, tmp_path: Path) -> Generator[None, None, None]:
        root = logging.getLogger()
        root.handlers.clear()
        config = LoggingConfig(log_file_prefix=str(tmp_path / "test_log"))
        LoggerFactory.setup(config)
        yield  # type: ignore[misc]
        root.handlers.clear()

    def test_file_handler_format_has_spaces(self) -> None:
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        fmt = file_handlers[0].formatter
        assert fmt is not None
        assert fmt._fmt == "%(asctime)s %(levelname)-8s %(name)s %(message)s"
