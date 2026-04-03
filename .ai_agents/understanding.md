# Architecture — .ai_agents/understanding.md

## Overview

A CartPole reinforcement learning demo built on Gymnasium. The app runs a CartPole-v1 environment, applies action policies, and renders the simulation in real-time using matplotlib.

## Async Architecture

The app uses an **async concurrent architecture** with two independent tasks coordinated via `asyncio.Lock`:

1. **Simulation task** (`CartPoleApp.loop_policy_over_episodes`) — steps the environment using a policy, handles episode termination and reset
2. **Viewing task** (`CartPoleViewer.render_live`) — snapshots env frames under the lock and displays them via matplotlib

Both tasks share the same `gym.Env` instance. The lock ensures mutual exclusion — only one task accesses the env at a time. Short `asyncio.sleep()` calls after each lock release yield control to the event loop, preventing starvation.

Matplotlib is integrated with asyncio via `plt.show(block=False)` and a manual event-loop pump (`draw_idle` + `flush_events`).

## Module Structure

```
app/
├── __main__.py        # Entry point — bootstraps app + viewer, runs asyncio.TaskGroup
├── cartpole.py        # CartPoleApp — owns env + lock, simulation loop
├── viewer.py          # CartPoleViewer — visualisation orchestrator (DI of env + lock)
├── render.py          # BaseRenderer / MatplotlibAnimator — frame rendering
├── plot.py            # BasePlotter / MatplotlibPlotter — static frame plotting
├── policy/
│   ├── base.py        # BaseActionPolicy (ABC)
│   └── random.py      # RandomActionPolicy
└── logging_config.py  # LoggerFactory, ColourFormatter, LoggingConfig
```

## Key Classes

- **`CartPoleApp`** — Owns the `gym.Env` and `asyncio.Lock`. Provides `apply_action(policy) -> bool` for single steps and `loop_policy_over_episodes(policy)` for the async episode loop.
- **`CartPoleViewer`** — Receives env + lock via DI. Orchestrates rendering via `BaseRenderer` and plotting via `BasePlotter`. `render_live()` spawns a frame-refresh sub-task and runs the matplotlib event loop.
- **`MatplotlibAnimator`** — Owns `(Figure, Axes, AxesImage)` explicitly via `_ensure_fig()`. Uses `plt.subplots()` (object-oriented API) to avoid implicit figure creation.
- **`BaseActionPolicy` / `RandomActionPolicy`** — Policy abstraction. `choose_action(action_space, obs_space) -> int`.

## Testing

- `pytest` with TDD approach
- Tests mock matplotlib via `@patch("app.render.plt")` and `@patch("app.render.animation.FuncAnimation")`
- Async renderer tests use `asyncio.run()` with task cancellation
- Validation pipeline: `ruff format` → `ruff check` → `pyrefly` → `pytest`
