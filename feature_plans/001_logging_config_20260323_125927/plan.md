# Feature Plan: Logging Configuration

## Goals
- Create a structured, configurable logging system using Python's `logging` library
- Console handler at INFO level (overridable) with coloured, pretty-formatted output
- File handler at DEBUG level, with a configurable path prefix and auto-generated timestamp suffix
- Configuration driven by a Pydantic `BaseSettings` model that reads from `configs/logging_settings.env`

## Constraints
- Must use `pydantic-settings` (`BaseSettings`) for config
- Must use Python's built-in `logging` module (not loguru, structlog, etc.)
- Console colours via ANSI escape codes (no extra dependency beyond what's already available)
- The file log path prefix includes the directory (e.g. `logs/cartpole`), and the suffix is automatically the timestamp of the current run (e.g. `logs/cartpole_20260323_130000.log`)
- Log level for console is configurable; file handler is always DEBUG
- `LoggerFactory.setup()` configures the root logger and returns `None`. Callers obtain module-specific loggers via `logging.getLogger(__name__)` which inherit the root config.

## Detailed Plan

### Configuration (`app/logging_config.py`)

A Pydantic `BaseSettings` subclass `LoggingConfig`:

| Field | Type | Default | Description |
|---|---|---|---|
| `log_level` | `str` | `"INFO"` | Console log level (overridable) |
| `log_file_prefix` | `str` | `"logs/cartpole"` | Path prefix for the log file (directory + base name) |
| `log_file_suffix` | `str` | (computed) | Auto-set to `_<timestamp>.log` at init time |

The settings model will:
- Read from `configs/logging_settings.env` via `model_config = SettingsConfigDict(env_file="configs/logging_settings.env")`
- Auto-compute `log_file_suffix` as a timestamp on instantiation

### Logger Setup (same file)

A class `LoggerFactory` with a static method `setup(config: LoggingConfig | None = None) -> None` that:
1. Gets the root logger
2. Sets root logger level to DEBUG (so all messages flow to handlers)
3. Adds a `StreamHandler` (console) with the configured level and a colour formatter
4. Creates parent directories for the log file if needed
5. Adds a `FileHandler` at DEBUG level with a standard formatter

After calling `setup()`, any module can do `logger = logging.getLogger(__name__)` and the logger will inherit the root logger's handlers.

### Colour Formatter

A custom `logging.Formatter` subclass `ColourFormatter` that applies ANSI colours to each log record part:
- Timestamp: cyan
- Level name: colour varies by level (DEBUG=white, INFO=green, WARNING=yellow, ERROR=red, CRITICAL=bold red)
- Logger name: magenta
- Message: default/white

## Diagrams

- Class diagram: `feature_plans/001_logging_config_20260323_125927/class_diagram.png`
- Sequence diagram: `feature_plans/001_logging_config_20260323_125927/sequence_diagram.png`

## Files Touched / Created

| File | Action |
|---|---|
| `app/logging_config.py` | **Create** — LoggingConfig, ColourFormatter, LoggerFactory |
| `configs/logging_settings.env` | **Create** — default env file with LOG_LEVEL and LOG_FILE_PREFIX |
| `pyproject.toml` | **Edit** — add `pydantic-settings` dependency |
| `tests/test_logging_config.py` | **Create** — unit tests |

## Classes and Functions

```
class ColourFormatter(logging.Formatter):
    """Custom formatter that adds ANSI colour codes to log output."""
    def format(self, record: logging.LogRecord) -> str

class LoggingConfig(BaseSettings):
    """Pydantic settings model for logging configuration."""
    model_config = SettingsConfigDict(env_file="configs/logging_settings.env")
    log_level: str = "INFO"
    log_file_prefix: str = "logs/cartpole"
    log_file_suffix: str  # computed at init via model_validator

class LoggerFactory:
    """Factory for configuring the root logger."""
    @staticmethod
    def setup(config: LoggingConfig | None = None) -> None
```

## Key Tests

1. `test_default_config` — LoggingConfig instantiates with correct defaults (INFO level, logs/cartpole prefix, suffix matches timestamp pattern)
2. `test_config_from_env` — LoggingConfig reads overrides from env vars (LOG_LEVEL, LOG_FILE_PREFIX)
3. `test_colour_formatter_output` — ColourFormatter produces output containing ANSI escape codes
4. `test_setup_configures_root_logger` — After `LoggerFactory.setup()`, the root logger has both a StreamHandler and a FileHandler
5. `test_file_handler_is_debug` — The file handler level is DEBUG
6. `test_console_handler_level_override` — When config has a non-default level, the console handler matches it
7. `test_log_file_created` — After setup, the log file exists at the expected path
8. `test_child_logger_inherits` — `logging.getLogger("some.module")` after setup propagates to root handlers

## Stages

### Stage 1: Config & Formatter (single commit)
- Add `pydantic-settings` dependency
- Create `configs/logging_settings.env` with defaults
- Create `app/logging_config.py` with `LoggingConfig`, `ColourFormatter`, and `LoggerFactory`
- Create `tests/test_logging_config.py` with all tests
- Run ruff format, ruff check, pyrefly, pytest
- Verify all pass

## Progress

### Stage 1: Config & Formatter — COMPLETE (2026-03-23)
- [x] Added `pydantic-settings` dependency via `uv add`
- [x] Added `ruff`, `pytest`, `pyrefly` as dev dependencies
- [x] Created `configs/logging_settings.env`
- [x] Created `app/logging_config.py` (LoggingConfig, ColourFormatter, LoggerFactory)
- [x] Created `tests/test_logging_config.py` (9 tests)
- [x] ruff format: pass
- [x] ruff check: pass
- [x] pyrefly: 0 errors
- [x] pytest: 9/9 passed
