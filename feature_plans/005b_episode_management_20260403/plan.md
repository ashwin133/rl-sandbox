# Feature 005b: Episode Management

## Goals

- Create `EpisodeRunner` class that runs episodes, detects `terminated`/`truncated`, and auto-resets
- Track episode boundaries: episode number, steps per episode, total steps
- Return structured results via pydantic models (`StepResult`, `EpisodeResult`, `BatchResult`)
- Support batch runs (N episodes sequentially)
- Persist batch results to disk (`.npz` for step data, `.json` for metadata)
- Async-compatible — respects the lock protocol from 005a

## Constraints

- `EpisodeRunner` does not own the env — receives `CartPoleApp` via dependency injection
- Policy and config are passed per-call, not stored as instance attributes
- No metrics collection (feature #8) — just episode data and logging
- Must integrate with the async architecture from 005a (lock protocol, task structure)
- Existing tests must continue to pass

## New: `app/episode.py`

```python
class StepResult(BaseModel):
    """Result of a single environment step."""
    obs: Any
    reward: float
    terminated: bool
    truncated: bool
    done: bool  # terminated or truncated


class EpisodeConfig(BaseModel):
    """Configuration for a single episode."""
    max_steps: int | None = None  # None = run until terminated/truncated
    auto_reset: bool = True       # reset env when episode ends


class BatchConfig(BaseModel):
    """Configuration for a batch of episodes."""
    no_episodes: int
    episode_config: EpisodeConfig


class EpisodeResult(BaseModel):
    """Result of a single episode run."""
    sequence: list[StepResult]
    timestamp: datetime
    policy_name: str              # policy class name, e.g. "RandomActionPolicy"
    episode_config: EpisodeConfig
    timesteps_reached: int


class BatchResult(BaseModel):
    """Result of a batch of episode runs."""
    episodes: list[EpisodeResult]
    batch_config: BatchConfig


class EpisodeRunner:
    """Runs episodes against a CartPoleApp.

    Does not store policy or config — receives them per-call for clarity.
    All run methods are async and acquire self.app.lock around env access.
    """

    def __init__(self, app: CartPoleApp) -> None
        # Only holds a ref to the app (env + lock)

    async def run_episode(self, policy: BaseActionPolicy, config: EpisodeConfig) -> EpisodeResult
        # Steps env under lock until done or max_steps reached.
        # Resets env if auto_reset is True.
        # Returns EpisodeResult with full step sequence.

    async def run_batch(self, policy: BaseActionPolicy, config: BatchConfig) -> BatchResult
        # Calls run_episode sequentially for config.no_episodes episodes.
        # Returns BatchResult.

    def log_batch_to_file(self, batch_result: BatchResult, path: Path) -> bool
        # Saves step data as .npz (efficient numpy storage).
        # Saves metadata (timestamps, policy name, configs) as .json.
        # Both files written to the same directory at `path`.
        # Returns True on success.

    @staticmethod
    def load_batch_from_file(path: Path) -> BatchResult
        # Parses .npz + .json back into a BatchResult python object.

    @property
    def current_episode(self) -> int

    @property
    def total_steps(self) -> int

    @property
    def episode_steps(self) -> int
```

### Storage format

```
path/
├── step_data.npz      # numpy arrays: obs, reward, terminated, truncated per episode
└── metadata.json      # timestamps, policy_name, configs, timesteps_reached per episode
```

`step_data.npz` stores one array per episode keyed by index (e.g. `episode_0_obs`, `episode_0_reward`, etc.). This keeps the bulk numerical data in an efficient binary format.

`metadata.json` stores everything else — serialised `EpisodeConfig`, `BatchConfig`, timestamps, policy name, and the `timesteps_reached` for each episode. This is human-readable and easy to inspect.

`load_batch_from_file` reconstructs `BatchResult` by reading both files and reassembling `StepResult` sequences from the numpy arrays + metadata.

## Modified: `app/cartpole.py`

- `apply_action` returns `StepResult` instead of `bool`
- `loop_policy_over_episodes` and `apply_policy_until_episode_end` removed — replaced by `EpisodeRunner`

## Modified: `app/viewer.py`

- `simulate_and_render_sequence` uses `EpisodeRunner.run_episode` with `max_steps=sequence_length`

## Modified: `app/__main__.py`

- Simulation task uses `EpisodeRunner.run_batch` (or infinite loop of `run_episode`) instead of `CartPoleApp.loop_policy_over_episodes`

## Files Touched / Created

| File | Action |
|------|--------|
| `app/episode.py` | **Create** — `StepResult`, `EpisodeConfig`, `BatchConfig`, `EpisodeResult`, `BatchResult`, `EpisodeRunner` |
| `app/cartpole.py` | **Edit** — `apply_action` returns `StepResult`, remove `loop_policy_over_episodes` and `apply_policy_until_episode_end` |
| `app/__main__.py` | **Edit** — use `EpisodeRunner` for simulation task |
| `app/viewer.py` | **Edit** — use `EpisodeRunner` in `simulate_and_render_sequence` |
| `tests/test_episode.py` | **Create** — unit tests for all episode classes and `EpisodeRunner` |
| `tests/test_cartpole.py` | **Edit** — update for `StepResult` return type |

## Key Tests

### StepResult / EpisodeResult / BatchResult models
1. `test_step_result_done_flag` — `done` is `True` when `terminated` or `truncated` is `True`
2. `test_episode_result_stores_policy_name` — `policy_name` is a string, not a class instance
3. `test_batch_result_contains_episodes` — `BatchResult.episodes` matches `batch_config.no_episodes`

### EpisodeRunner.run_episode
4. `test_run_episode_returns_episode_result` — returns `EpisodeResult` with correct fields
5. `test_run_episode_auto_resets` — env is reset when `auto_reset=True` and episode ends
6. `test_run_episode_no_auto_reset` — env is not reset when `auto_reset=False`
7. `test_run_episode_respects_max_steps` — stops at `max_steps` even if not terminated
8. `test_run_episode_sequence_length_matches_timesteps` — `len(sequence) == timesteps_reached`
9. `test_run_episode_acquires_lock` — env access happens under the lock

### EpisodeRunner.run_batch
10. `test_run_batch_runs_correct_number_of_episodes` — runs `no_episodes` episodes
11. `test_run_batch_returns_batch_result` — returns `BatchResult` with correct structure

### EpisodeRunner properties
12. `test_current_episode_increments` — `current_episode` increments after each episode
13. `test_total_steps_accumulates` — `total_steps` counts across episodes
14. `test_episode_steps_resets` — `episode_steps` resets to 0 on new episode

### Persistence
15. `test_log_batch_to_file_creates_npz_and_json` — both files created at path
16. `test_load_batch_from_file_roundtrip` — save then load produces identical `BatchResult`
17. `test_npz_contains_correct_arrays` — numpy arrays have correct shapes and keys

### CartPoleApp
18. `test_apply_action_returns_step_result` — updated return type

## Stages

### Stage 1: Models and CartPoleApp update
- Create `app/episode.py` with `StepResult`, `EpisodeConfig`, `BatchConfig`, `EpisodeResult`, `BatchResult`
- Update `app/cartpole.py`: `apply_action` returns `StepResult`
- Update existing tests for new return type
- Write model tests (1–3)
- Run validation

### Stage 2: EpisodeRunner core
- Add `EpisodeRunner` to `app/episode.py` with `run_episode` (async, lock-aware)
- Add properties (`current_episode`, `total_steps`, `episode_steps`)
- Write tests (4–9, 12–14)
- Run validation

### Stage 3: Batch runs
- Add `run_batch` to `EpisodeRunner`
- Write tests (10–11)
- Run validation

### Stage 4: Persistence
- Add `log_batch_to_file` and `load_batch_from_file`
- Write tests (15–17)
- Run validation

### Stage 5: Integration
- Update `app/__main__.py` to use `EpisodeRunner`
- Update `app/viewer.py` to use `EpisodeRunner.run_episode`
- Remove `loop_policy_over_episodes` and `apply_policy_until_episode_end` from `CartPoleApp`
- Run full suite + validation

## Progress

- [x] Stage 1: Models (`StepResult`, `EpisodeConfig`, `BatchConfig`, `EpisodeResult`, `BatchResult`) + `apply_action` returns `StepResult`
- [x] Stage 2: `EpisodeRunner` core — async `run_episode` with lock, auto-reset, max_steps, properties
- [x] Stage 3: `run_batch` — sequential multi-episode runs
- [x] Stage 4: Persistence — `log_batch_to_file` (.npz + .json), `load_batch_from_file` roundtrip
- [x] Stage 5: Integration — `__main__.py` uses `EpisodeRunner`, removed `loop_policy_over_episodes` and `apply_policy_until_episode_end` from `CartPoleApp`
