# Feature Backlog

Ordered by implementation priority. Each feature builds on the ones above it.

| # | Feature | Description | Status |
|---|---------|-------------|--------|
| 1 | Module Separation | Split `viewer.py` into `render.py`, `plot.py`, `viewer.py` (orchestrator). Move policies into `app/policy/` package. | Done |
| 2 | Code Cleanup | Add type hints everywhere, fix mutable defaults, use `@abstractmethod` properly, fix ABC/impl signature mismatches, remove unused imports, fix `render_live` bugs, consistent import ordering. | Done |
| 3 | Tests | Unit tests for cartpole, action_policy, viewer, render, plot. Integration test for full headless flow. | Pending |
| 4 | Episode Management | `EpisodeRunner` class — handles `terminated`/`truncated`, resets env, tracks episode boundaries. Configurable: stop after N episodes, N steps, or run indefinitely. | Pending |
| 5 | Fixed-Length Live Simulation | `render_live` accepts optional `max_frames` param. Frame counter logging. Collect frames during live sim for later inspection/export. | Pending |
| 6 | Graceful Termination | Clean shutdown on `Ctrl+C` — close window, close env, log summary. Signal handler around render loop. Configurable auto-reset vs stop on episode end. | Pending |
| 7 | Config Consolidation | `ViewerConfig` to `BaseSettings`. Top-level `AppConfig` composing logging + viewer + simulation configs. Single entry point from `configs/`. | Pending |
| 8 | Metrics Collection | `MetricsCollector` for per-step (reward, obs) and per-episode (total reward, length) data. Hooks into `EpisodeRunner`. | Pending |
| 9 | Animation Export | Save frame sequences as GIF/MP4 via moviepy. Configurable output path and FPS. | Pending |
| 10 | Metrics Plotting | Reward curves, episode length plots. Live-updating and post-run static modes. Export to PNG/PDF. | Pending |
| 11 | Environment Abstraction | Extract env creation behind config/factory. `env_name` and `render_mode` as config params. Support any Gymnasium env. | Pending |
| 12 | CLI Improvements | Expand argparse to subcommands (`run`, `render`, `export`, `plot`). Add `--max-frames`, `--policy`, `--env`, `--output` flags. Fix stale poe tasks in `pyproject.toml`. | Pending |
