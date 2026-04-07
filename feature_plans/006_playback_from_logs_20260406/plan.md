# Feature 006: Playback from Logs

## Goal

Allow a user to replay episodes from saved batch log files via the `playback-from-logs` CLI mode. During a simulation run, `replay_freq` controls which episodes have their RGB frames captured — e.g. `--replay-freq 5` on a 30-episode batch only stores frames for episodes 0, 5, 10, 15, 20, 25. This minimises storage. At replay time, we load those frames and play them all back at the sim speed they were recorded at — no env, no physics, no CLI speed override.

## Detailed Plan

### Core Insight: Store Frames, Not Actions

Rather than storing step-level data and re-running physics, we capture the rendered RGB frame at each step during the simulation — but **only for episodes selected by `replay_freq`**. Replay becomes pure frame playback. This is simpler, more robust, and decoupled from the env.

### `replay_freq` Is a Sim-Time Parameter

`replay_freq` determines which episodes get frames captured **during the simulation**, not at replay time:

- A 30-episode batch with `replay_freq=5` stores frames for 6 episodes instead of 30 — ~80% storage reduction.
- All 30 episodes still run (the policy still gets exercised and the live viewer shows everything), but only sampled episodes produce saved frames.
- At replay time, you replay everything that was saved. No filtering, no speed override — the metadata tells the replayer how fast each episode ran.

`replay_freq` belongs on `BatchConfig` and the `run-sim` CLI. The `playback-from-logs` CLI only takes `--path`.

### Replay Speed Comes from Metadata

Each `EpisodeRecording` stores `sim_speed_multiplier` — the speed at which the sim actually ran. `ReplayRunner` reads this from the metadata and uses it to time frame display. There is no `--sim-speed` on the `playback-from-logs` CLI. This ensures what you see in replay matches what happened during the sim.

### Renderer Modifications Required

The current `MatplotlibAnimator` API has a gap for replay:

| Existing method | Why it doesn't work for replay |
|---|---|
| `start_live_render(callback)` | Driven by `FuncAnimation` — designed for async sim with a callback. Replay controls timing itself |
| `render_from_frames(frames)` | Blocks on `plt.show()`, plays all frames with fixed interval. No control between episodes, no logging |

**Two new methods needed on `BaseRenderer` / `MatplotlibAnimator`:**

| Method | Signature | Purpose |
|---|---|---|
| `init_display` | `() -> None` | Creates figure/axes via `_ensure_fig()`, calls `plt.show(block=False)`. Idempotent. Called once before replay starts |
| `display_frame` | `(frame: np.ndarray) -> None` | Calls `update_frame(frame)`, then `fig.canvas.draw_idle()` + `fig.canvas.flush_events()`. Gives the caller frame-by-frame control without `FuncAnimation` |

This separates the timing concern (owned by `ReplayRunner`) from the rendering concern (owned by `MatplotlibAnimator`). See `sd_playback_episode.puml` for the interaction.

### What Changes in the Simulation Save Path

Currently `run_episode` records a `list[StepResult]` and `log_batch_to_file` serialises those to `.npz` + `.json`. Under the new approach:

1. `run_batch` decides which episodes to capture frames for based on `config.replay_freq`.
2. For captured episodes, `run_episode(capture_frames=True)` grabs `env.render()` under the same lock as `env.step()` at each step — one lock acquisition covers both.
3. For non-captured episodes, `run_episode(capture_frames=False)` runs as before with no `env.render()` call.
4. The save path writes **video frames** (per-episode numpy arrays of shape `(n_steps, H, W, 3)`) to `.npz`, plus **metadata** to `.json`.
5. `StepResult`, `EpisodeResult`, `BatchResult` remain as in-memory models for the simulation loop but are **not serialised**. The persistence layer only cares about frames + metadata.

### Replay Architecture

Replay does **not** use the async two-task architecture. There is no env, no lock, no simulation. It is single-task frame playback:

```
CLI (playback-from-logs --path ./results/batch_...)
  │
  ▼
FileOps.load_batch_replay(path) → BatchReplay
  │  - Reads .npz (frame arrays) + .json (metadata)
  │  - Builds BatchReplay with frames + metadata per episode
  │
  ▼
ReplayRunner.replay_batch(batch_replay)
  │
  ▼ (for each EpisodeReplay)
ReplayRunner.replay_episode(episode_replay)
  │  - Reads sim_speed_multiplier from metadata
  │  - Calls renderer.init_display() (idempotent)
  │  - For each frame:
  │      - Calls renderer.display_frame(frame)
  │      - Sleeps for MODEL_TICK / sim_speed_multiplier
  │  - Logs episode summary
```

## Sequence Diagrams

All diagrams are in PlantUML format in this directory:

| Diagram | File | Description |
|---|---|---|
| Batch execution overview | `sd_run_sim_batch.puml` | Shows the full `run-sim` flow: TaskGroup with sim + viewer tasks, selective frame capture by replay_freq, save via FileOps |
| Captured episode detail | `sd_run_sim_captured_episode.puml` | Single episode with `capture_frames=True` — shows `env.step()` + `env.render()` under the same lock acquisition |
| Normal episode detail | `sd_run_sim_normal_episode.puml` | Single episode with `capture_frames=False` — no `env.render()`, less work per step |
| Replay overview | `sd_playback_from_logs.puml` | Full `playback-from-logs` flow: load from FileOps, iterate episodes, speed from metadata |
| Replay episode detail | `sd_playback_episode.puml` | Single episode replay: `init_display()` + `display_frame()` loop with timing from metadata |

## New Models

**`EpisodeRecording`** — metadata saved per captured episode (in `.json`):
- `episode_index: int` — original index in the batch (e.g. 0, 5, 10 if replay_freq=5)
- `policy_name: str` — e.g. "RandomActionPolicy"
- `timestamp: datetime` — when the episode ran
- `timesteps_reached: int` — number of steps before termination/truncation
- `sim_speed_multiplier: float` — speed the sim ran at (used at replay time)

**`BatchRecording`** — metadata saved per batch (in `.json`):
- `no_episodes: int` — total episodes in the batch (including non-captured)
- `replay_freq: int` — the frequency used during capture
- `episodes: list[EpisodeRecording]` — only the captured episodes

**`EpisodeReplay`** — everything needed to replay one episode (in memory, dataclass):
- `frames: list[np.ndarray]` — RGB frames to display
- `metadata: EpisodeRecording` — for logging and sim speed

**`BatchReplay`** — everything needed to replay the saved episodes (dataclass):
- `episodes: list[EpisodeReplay]` — all saved episodes
- `source_path: str` — where this was loaded from
- `total_source_episodes: int` — how many total episodes were in the batch
- `replay_freq: int` — the frequency that was used at save time

### `BatchConfig` Changes

Add `replay_freq: int = 1` to `BatchConfig`. Default of 1 = capture every episode.

## Files to Touch / Create

| File | Action | Description |
|------|--------|-------------|
| `app/models/recording.py` | **Create** | `EpisodeRecording`, `BatchRecording` models |
| `app/models/replay.py` | **Create** | `EpisodeReplay`, `BatchReplay` dataclasses |
| `app/models/batch.py` | Modify | Add `replay_freq: int = 1` to `BatchConfig` |
| `app/models/__init__.py` | Modify | Re-export new models |
| `app/file_ops.py` | **Create** | `FileOps` class — `save`, `load_batch_replay` |
| `app/replay.py` | **Create** | `ReplayRunner` class — `replay_episode`, `replay_batch` |
| `app/render.py` | Modify | Add `init_display()` and `display_frame()` to `BaseRenderer` / `MatplotlibAnimator` |
| `app/episode.py` | Modify | Remove persistence methods. Add `capture_frames` param to `run_episode`. Update `run_batch` for selective capture |
| `app/__main__.py` | Modify | Add `--replay-freq` to `run-sim`. Wire `playback-from-logs` (only `--path`) |
| `results/` | **Delete contents** | Remove old log files (incompatible format) |
| `tests/test_file_ops.py` | **Create** | Tests for FileOps save/load |
| `tests/test_replay.py` | **Create** | Tests for ReplayRunner |
| `tests/test_render.py` | Modify | Add tests for `init_display` and `display_frame` |
| `tests/test_episode.py` | Modify | Remove persistence tests (moved), add frame capture tests |

## Classes and Functions

### `FileOps` (new — `app/file_ops.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `save` | `(captured_frames: dict[int, list[np.ndarray]], metadata: BatchRecording, path: Path) -> None` (static) | Saves per-episode frame arrays to `.npz` and metadata to `.json` |
| `load_batch_replay` | `(path: Path) -> BatchReplay` (static) | Loads frames + metadata, returns `BatchReplay` with all saved episodes |

### `ReplayRunner` (new — `app/replay.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(renderer: BaseRenderer)` | Takes the renderer to display frames on |
| `replay_episode` | `async (self, replay: EpisodeReplay) -> None` | Reads speed from metadata, calls `init_display`, loops frames via `display_frame`, sleeps per tick |
| `replay_batch` | `async (self, batch_replay: BatchReplay) -> None` | Iterates over episodes, logs progress, calls `replay_episode` |

### `BaseRenderer` / `MatplotlibAnimator` (modified — `app/render.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `init_display` | `() -> None` | New abstract + impl. Calls `_ensure_fig()` + `plt.show(block=False)`. Idempotent |
| `display_frame` | `(frame: np.ndarray) -> None` | New abstract + impl. Calls `update_frame(frame)` + `draw_idle()` + `flush_events()` |

### `EpisodeRunner` (modified — `app/episode.py`)

| Change | Description |
|--------|-------------|
| `run_episode` | New param `capture_frames: bool = False`. When True, grabs `env.render()` under the lock. Returns `tuple[EpisodeResult, list[np.ndarray] \| None]` |
| `run_batch` | Uses `config.replay_freq` to decide which episodes get `capture_frames=True`. Returns `tuple[BatchResult, dict[int, list[np.ndarray]]]` |
| Remove | `log_batch_to_file`, `load_batch_from_file` (moved to `FileOps`) |

### Models

**`app/models/batch.py`** — modified:

```python
class BatchConfig(BaseModel):
    no_episodes: int
    episode_config: EpisodeConfig
    replay_freq: int = 1  # capture frames every Nth episode (1 = all)
```

**`app/models/recording.py`** — serialised to disk:

```python
class EpisodeRecording(BaseModel):
    episode_index: int
    policy_name: str
    timestamp: datetime
    timesteps_reached: int
    sim_speed_multiplier: float

class BatchRecording(BaseModel):
    no_episodes: int
    replay_freq: int
    episodes: list[EpisodeRecording]
```

**`app/models/replay.py`** — in-memory only:

```python
@dataclass
class EpisodeReplay:
    frames: list[np.ndarray]
    metadata: EpisodeRecording

@dataclass
class BatchReplay:
    episodes: list[EpisodeReplay]
    source_path: str
    total_source_episodes: int
    replay_freq: int
```

## Constraints

1. **Storage size** — Each CartPole frame is ~480×640×3 uint8 (~900KB). With `replay_freq=5` on 30 episodes of 100 steps: 6 × 100 = 600 frames. `.npz` compression reduces this significantly.
2. **Frame capture timing** — Frames must be captured under the same lock acquisition as `env.step()` — see `sd_run_sim_captured_episode.puml`.
3. **No env needed for replay** — `ReplayRunner` depends only on `BaseRenderer`, not on `CartPoleApp`, `gymnasium`, or `asyncio.Lock`.
4. **`replay_freq` is write-time only** — once saved, the set of captured episodes is fixed. To change sampling, re-run the sim.
5. **Replay speed is not configurable** — it comes from `EpisodeRecording.sim_speed_multiplier`. What you see in replay is what happened during the sim.

## Goals

- [ ] `run-sim --replay-freq 5` captures RGB frames only for every 5th episode during simulation
- [ ] Non-captured episodes still run normally (policy exercised, live viewer shows everything) but produce no saved frames
- [ ] `playback-from-logs --path ./results/batch_...` replays all saved episodes at their original sim speed
- [ ] Replay uses new `init_display()` / `display_frame()` renderer methods — no `FuncAnimation`, caller controls timing
- [ ] `FileOps` cleanly separates persistence from episode running / replay logic
- [ ] Old log files deleted (incompatible format)

## Key Tests

### Unit Tests — Models
1. **`EpisodeRecording`** — constructs correctly, roundtrips to/from dict
2. **`BatchRecording`** — stores correct total episode count, replay_freq, and captured episode list
3. **`BatchConfig.replay_freq`** — defaults to 1, accepts custom values
4. **`EpisodeReplay`** — holds frames + metadata as dataclass
5. **`BatchReplay`** — holds list of EpisodeReplay + source metadata

### Unit Tests — FileOps
6. **`FileOps.save`** — creates `.npz` and `.json`, `.npz` contains correct array keys per captured episode
7. **`FileOps.save` / `load_batch_replay` roundtrip** — save frames + metadata, load back, verify frames and metadata match
8. **`FileOps.load_batch_replay`** — verify loaded `BatchReplay` has correct `total_source_episodes` and `replay_freq`

### Unit Tests — Renderer
9. **`init_display`** — calls `plt.subplots` and `plt.show(block=False)`, idempotent on second call
10. **`display_frame`** — calls `update_frame` + `draw_idle` + `flush_events`

### Unit Tests — EpisodeRunner frame capture
11. **`run_episode(capture_frames=True)` returns frames** — frames list is non-empty, each element is ndarray of shape (H, W, 3)
12. **`run_episode(capture_frames=False)` returns None** — no frame capture overhead
13. **`run_batch` with replay_freq=3 on 10 episodes** — captured frames dict has keys {0, 3, 6, 9}

### Integration Tests — ReplayRunner
14. **`replay_episode`** — plays back frames, verify `display_frame` called once per frame
15. **`replay_batch`** — replays correct number of episodes
16. **`replay_episode` uses metadata speed** — verify sleep duration = `MODEL_TICK / sim_speed_multiplier`

## Implementation Stages

### Stage 1: Renderer methods + frame capture + FileOps + recording models

**Changes:**
- Add `init_display()` and `display_frame()` to `BaseRenderer` / `MatplotlibAnimator` in `app/render.py`
- Add `replay_freq: int = 1` to `BatchConfig` in `app/models/batch.py`
- Create `app/models/recording.py` with `EpisodeRecording`, `BatchRecording`
- Add `capture_frames: bool = False` param to `run_episode`. When True, grabs `env.render()` under the lock. Returns `tuple[EpisodeResult, list[np.ndarray] | None]`
- Update `run_batch` to pass `capture_frames=True` for episodes where `index % replay_freq == 0`. Returns `tuple[BatchResult, dict[int, list[np.ndarray]]]`
- Create `app/file_ops.py` with `FileOps.save` and `FileOps.load_batch_replay`
- Remove `log_batch_to_file` and `load_batch_from_file` from `EpisodeRunner`
- Update `__main__.py` `run-sim` to accept `--replay-freq`, pass to `BatchConfig`, use `FileOps.save`
- Delete old `results/` contents
- Move/rewrite persistence tests from `test_episode.py` to `test_file_ops.py`

**Tests:** #1, #2, #3, #6, #7, #8, #9, #10, #11, #12, #13
**Estimated changes:** ~400 lines

### Stage 2: Replay models + ReplayRunner + CLI wiring

**Changes:**
- Create `app/models/replay.py` with `EpisodeReplay`, `BatchReplay` (dataclasses)
- Create `app/replay.py` with `ReplayRunner` class (`replay_episode`, `replay_batch`)
- Update `app/models/__init__.py` with new re-exports
- Wire `playback-from-logs` in `__main__.py` — only `--path`, load via `FileOps.load_batch_replay`, replay via `ReplayRunner`

**Tests:** #4, #5, #14, #15, #16
**Estimated changes:** ~200 lines

## Codebase Modifications Summary

| Area | What changes | Why |
|------|-------------|-----|
| `BaseRenderer` / `MatplotlibAnimator` | Add `init_display()` + `display_frame()` | Replay needs frame-by-frame control without `FuncAnimation`. Existing methods either block (`render_from_frames`) or require a callback (`start_live_render`) |
| `EpisodeRunner.run_episode` | Add `capture_frames` param, return tuple | Selective frame capture per episode |
| `EpisodeRunner.run_batch` | Use `replay_freq` to select capture episodes, return frames dict | Filtering happens at sim time, not replay time |
| `EpisodeRunner` | Remove `log_batch_to_file`, `load_batch_from_file` | Moved to `FileOps` (SRP) |
| `BatchConfig` | Add `replay_freq` field | Controls which episodes get frames saved |
| `__main__.py` `run-sim` | Add `--replay-freq` arg | User control over capture frequency |
| `__main__.py` `playback-from-logs` | Only `--path`, no `--sim-speed` | Speed comes from recording metadata |
| Save format | Frames `.npz` + metadata `.json` replaces old step-data `.npz` + metadata `.json` | Old format stored StepResult arrays, new format stores RGB frames |

## Progress

- [x] **Stage 1** — Complete. Renderer `init_display`/`display_frame` added, `replay_freq` on `BatchConfig`, recording models created, `run_episode` captures frames, `FileOps` extracted, old persistence removed, old results deleted. 95 tests passing, full validation green.
- [x] **Stage 2** — Complete. Replay models (dataclasses), `ReplayRunner`, CLI wired. 101 tests passing, full validation green.
