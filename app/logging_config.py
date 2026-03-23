import logging
import sys
from datetime import datetime
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingConfig(BaseSettings):
    """Pydantic settings model for logging configuration."""

    model_config = SettingsConfigDict(env_file="configs/logging_settings.env")

    log_level: str = "INFO"
    log_file_prefix: str = "logs/cartpole"
    log_file_suffix: str = ""

    @model_validator(mode="after")
    def _set_suffix(self) -> "LoggingConfig":
        if not self.log_file_suffix:
            self.log_file_suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        return self

    @property
    def log_file_path(self) -> str:
        return f"{self.log_file_prefix}{self.log_file_suffix}"


class ColourFormatter(logging.Formatter):
    """Custom formatter that adds ANSI colour codes to log output."""

    RESET: str = "\033[0m"
    CYAN: str = "\033[36m"
    MAGENTA: str = "\033[35m"

    LEVEL_COLOURS: dict[int, str] = {
        logging.DEBUG: "\033[37m",  # white
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-8s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        level_colour = self.LEVEL_COLOURS.get(record.levelno, self.RESET)

        record.asctime = (
            f"{self.CYAN}{self.formatTime(record, self.datefmt)}{self.RESET}"
        )
        padded_level = f"{record.levelname:<8}"
        record.levelname = f"{level_colour}{padded_level}{self.RESET}"
        record.name = f"{self.MAGENTA}{record.name}:{record.lineno}{self.RESET}"

        return f"{record.asctime} {record.levelname}{record.name} {record.getMessage()}"


class LoggerFactory:
    """Factory for configuring the root logger."""

    @staticmethod
    def setup(config: LoggingConfig | None = None) -> None:
        if config is None:
            config = LoggingConfig()

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(getattr(logging, config.log_level.upper()))
        console_handler.setFormatter(ColourFormatter())
        root_logger.addHandler(console_handler)

        # File handler
        log_file = Path(config.log_file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(str(log_file))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-8s %(name)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(file_handler)
