# Feature Backlog

Ordered by implementation priority. Each feature builds on the ones above it.

| # | Feature | Description | Status |
|---|---------|-------------|--------|
| 1 | Module Separation | Split `viewer.py` into `render.py`, `plot.py`, `viewer.py` (orchestrator). Move policies into `app/policy/` package. | Done |
| 2 | Code Cleanup | Add type hints everywhere, fix mutable defaults, use `@abstractmethod` properly, fix ABC/impl signature mismatches, remove unused imports, fix `render_live` bugs, consistent import ordering. | Done |
| 3 | Tests | Unit tests for environment execution, policies, viewer, render, and plot. Integration test for the full headless flow. | Done |
| 4a | Async Architecture | Decouple simulation and rendering into two async tasks via `asyncio.TaskGroup`. Shared `gym.Env` access coordinated by `asyncio.Lock`. DI of env+lock into viewer. Explicit matplotlib figure ownership. | Done |
| 4b | Episode Management | `EpisodeRunner` class — handles `terminated`/`truncated`, resets env, tracks episode boundaries. Configurable episode and batch limits. | Done |
| 5 | Fixed-Length Live Simulation | `render_live` accepts optional `max_frames` param. Frame counter logging. Collect frames during live sim for later inspection/export. | Pending |
| 6 | Graceful Termination | Clean shutdown on `Ctrl+C` — close window, close env, log summary. Signal handler around render loop. Configurable auto-reset vs stop on episode end. | Pending |
| 7 | Config Consolidation | `ViewerConfig` to `BaseSettings`. Top-level `AppConfig` composing logging + viewer + simulation configs. Single entry point from `configs/`. | Pending |
| 8 | Metrics Collection | `MetricsCollector` for per-step (reward, obs) and per-episode (total reward, length) data. Hooks into `EpisodeRunner`. | Pending |
| 9 | Animation Export | Save frame sequences as GIF/MP4 via moviepy. Configurable output path and FPS. | Pending |
| 10 | Metrics Plotting | Reward curves, episode length plots. Live-updating and post-run static modes. Export to PNG/PDF. | Pending |
| 11 | Environment Abstraction | Extract environment creation behind `BaseEnvApp`, `GymEnvApp`, and the `EnvSpec` registry. | Done |
| 12 | CLI Improvements | Split argparse into training, simulation, and playback subcommands with registered environment selection. | Done |
| 13 | Algorithm Evaluation Toolbox | Add reusable trainer, policy, metrics, evaluation, and algorithm-aware checkpoint boundaries, with REINFORCE as the first implementation. | Done |
| 14 | Public API Documentation | Generate a browsable API reference with MkDocs and deploy it through GitHub Pages. | Done |
