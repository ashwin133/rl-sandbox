import logging
import re
from collections.abc import Generator
from pathlib import Path

import pytest

from app.logging_config import ColourFormatter, LoggerFactory, LoggingConfig


class TestLoggingConfig:
    """Tests for LoggingConfig settings model."""

    def test_default_config(self) -> None:
        config = LoggingConfig()
        assert config.log_level == "INFO"
        assert config.log_file_prefix == "logs/cartpole"
        assert re.match(r"_\d{8}_\d{6}\.log$", config.log_file_suffix)

    def test_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        monkeypatch.setenv("LOG_FILE_PREFIX", "logs/test_run")
        config = LoggingConfig()
        assert config.log_level == "WARNING"
        assert config.log_file_prefix == "logs/test_run"

    def test_log_file_path_property(self) -> None:
        config = LoggingConfig()
        assert (
            config.log_file_path == f"{config.log_file_prefix}{config.log_file_suffix}"
        )
        assert config.log_file_path.endswith(".log")


class TestColourFormatter:
    """Tests for ColourFormatter."""

    def test_colour_formatter_output(self) -> None:
        formatter = ColourFormatter()
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello world",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        # Should contain ANSI escape codes
        assert "\033[" in output
        assert "hello world" in output


class TestLoggerFactory:
    """Tests for LoggerFactory.setup()."""

    @pytest.fixture(autouse=True)
    def _clean_root_logger(self, tmp_path: Path) -> Generator[None, None, None]:
        """Remove all handlers from root logger before and after each test."""
        root = logging.getLogger()
        root.handlers.clear()
        config = LoggingConfig(log_file_prefix=str(tmp_path / "test_log"))
        self.config = config  # noqa: PT023
        yield  # type: ignore[misc]
        root.handlers.clear()

    def test_setup_configures_root_logger(self) -> None:
        LoggerFactory.setup(self.config)
        root = logging.getLogger()
        handler_types = {type(h) for h in root.handlers}
        assert logging.StreamHandler in handler_types
        assert logging.FileHandler in handler_types

    def test_file_handler_is_debug(self) -> None:
        LoggerFactory.setup(self.config)
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert file_handlers[0].level == logging.DEBUG

    def test_console_handler_level_override(self) -> None:
        config = LoggingConfig(
            log_level="WARNING",
            log_file_prefix=self.config.log_file_prefix,
        )
        LoggerFactory.setup(config)
        root = logging.getLogger()
        stream_handlers = [h for h in root.handlers if type(h) is logging.StreamHandler]
        assert len(stream_handlers) == 1
        assert stream_handlers[0].level == logging.WARNING

    def test_log_file_created(self) -> None:
        LoggerFactory.setup(self.config)
        assert Path(self.config.log_file_path).exists()

    def test_child_logger_inherits(self) -> None:
        LoggerFactory.setup(self.config)
        child_logger = logging.getLogger("some.module")
        # Child should propagate to root and thus have effective level DEBUG
        assert child_logger.getEffectiveLevel() == logging.DEBUG
        # Writing through child should reach the file handler
        child_logger.debug("test message")
        log_path = Path(self.config.log_file_path)
        log_contents = log_path.read_text()
        assert "test message" in log_contents
