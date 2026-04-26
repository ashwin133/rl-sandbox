# Architecture — .ai_agents/understanding.md

## Overview

A CartPole reinforcement learning demo built on Gymnasium. The app runs a CartPole-v1 environment, applies action policies, and renders the simulation in real-time using matplotlib.

## Async Architecture

The app uses an **async concurrent architecture** with two independent tasks coordinated via `asyncio.Lock`:

1. **Simulation task** (`EpisodeRunner.run_episode` in a loop) — steps the environment using a policy, handles episode termination and reset
2. **Viewing task** (`CartPoleViewer.render_live`) — snapshots env frames under the lock and displays them via matplotlib

Both tasks share the same `gym.Env` instance. The lock ensures mutual exclusion — only one task accesses the env at a time. Short `asyncio.sleep()` calls after each lock release yield control to the event loop, preventing starvation.

Matplotlib is integrated with asyncio via `plt.show(block=False)` and a manual event-loop pump (`draw_idle` + `flush_events`).

## Module Structure

```
app/
├── __main__.py        # Entry point — CLI subparsers (run-sim / playback-from-logs)
├── cartpole.py        # CartPoleApp — owns env + lock, simulation loop
├── episode.py         # EpisodeRunner — async episode/batch execution with optional frame capture
├── replay.py          # ReplayRunner — pure frame playback, no env dependency
├── file_ops.py        # FileOps — save/load frames .npz + metadata .json
├── viewer.py          # CartPoleViewer — visualisation orchestrator (DI of env + lock)
├── render.py          # BaseRenderer / MatplotlibAnimator — frame rendering (incl. init_display / display_frame)
├── plot.py            # BasePlotter / MatplotlibPlotter — static frame plotting
├── utilities.py       # Utilities — batch directory creation
├── models/
│   ├── episode.py     # StepResult, EpisodeConfig, EpisodeResult
│   ├── batch.py       # BatchConfig (incl. replay_freq), BatchResult
│   ├── recording.py   # EpisodeRecording, BatchRecording (disk persistence metadata)
│   └── replay.py      # EpisodeReplay, BatchReplay (in-memory dataclasses)
├── policy/
│   ├── base.py        # BaseActionPolicy (ABC)
│   ├── random.py      # RandomActionPolicy
│   ├── reinforce.py   # ReinforcePolicy (wraps trained PolicyNetwork)
│   └── dqn.py         # DqnPolicy (wraps trained QNetwork)
├── training/
│   ├── base.py        # BaseTrainer (ABC)
│   ├── reinforce.py   # ReinforceTrainer (REINFORCE / policy gradient)
│   ├── dqn.py         # DqnTrainer (DQN with replay buffer + target network)
│   ├── networks.py    # PolicyNetwork, QNetwork (PyTorch nn.Modules)
│   └── replay_buffer.py # ReplayBuffer, Transition (experience replay for DQN)
├── models/
│   └── training.py    # TrainingConfig, EpisodeMetrics, TrainingResult
└── logging_config.py  # LoggerFactory, ColourFormatter, LoggingConfig
```

## Key Classes

- **`CartPoleApp`** — Owns the `gym.Env` and `asyncio.Lock`. Provides `apply_action(policy) -> StepResult` for single steps and `reset_env()` for resets.
- **`EpisodeRunner`** — Runs episodes against a `CartPoleApp`. Async `run_episode(policy, config, capture_frames)` steps env under lock, optionally captures RGB frames, returns `(EpisodeResult, frames | None)`. `run_batch` runs N episodes, selectively capturing frames by `replay_freq`.
- **`ReplayRunner`** — Pure frame playback. Takes a `BaseRenderer`, iterates stored frames with timing from recording metadata. No env, no lock, no physics.
- **`FileOps`** — Persistence. `save` writes frames `.npz` + metadata `.json`. `load_batch_replay` reads them back into a `BatchReplay`.
- **`CartPoleViewer`** — Receives env + lock via DI. Orchestrates rendering via `BaseRenderer` and plotting via `BasePlotter`. `render_live()` spawns a frame-refresh sub-task and runs the matplotlib event loop.
- **`MatplotlibAnimator`** — Owns `(Figure, Axes, AxesImage)` explicitly via `_ensure_fig()`. Uses `plt.subplots()` (object-oriented API) to avoid implicit figure creation.
- **`BaseActionPolicy` / `RandomActionPolicy`** — Policy abstraction. `choose_action(action_space, obs_space) -> int`.
- **`ReinforcePolicy` / `DqnPolicy`** — Trained policy subclasses wrapping `PolicyNetwork` / `QNetwork`. Support save/load via PyTorch `state_dict`.
- **`ReinforceTrainer`** — REINFORCE (Monte Carlo policy gradient). Collects full episodes, computes discounted returns, normalises, and updates via `-Σ log_prob * G_t`.
- **`DqnTrainer`** — Deep Q-Network. Experience replay buffer, target network (hard sync), ��-greedy exploration with decay. Updates per-step via Bellman MSE loss.
- **`PolicyNetwork` / `QNetwork`** — Two-hidden-layer ReLU networks. PolicyNetwork outputs softmax probabilities; QNetwork outputs raw Q-values.
- **`ReplayBuffer`** — Fixed-capacity deque of `Transition` namedtuples with random sampling for DQN mini-batches.

## Testing

- `pytest` with TDD approach
- Tests mock matplotlib via `@patch("app.render.plt")` and `@patch("app.render.animation.FuncAnimation")`
- Async renderer tests use `asyncio.run()` with task cancellation
- Validation pipeline: `ruff format` → `ruff check` → `pyrefly` → `pytest`
