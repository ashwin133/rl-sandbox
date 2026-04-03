# Feature 005b: Episode Management

## Goals

- Create `EpisodeRunner` class that manages the step loop, detects `terminated`/`truncated`, and auto-resets the environment
- Track episode boundaries: episode number, steps per episode, total steps
- Return structured step results via `StepResult` pydantic model
- Configurable stopping conditions: max total steps, max episodes, or run indefinitely
- Replace raw loops in `CartPoleApp` and `CartPoleViewer` with `EpisodeRunner`

## Constraints

- `EpisodeRunner` does not own the env — receives `CartPoleApp` via dependency injection
- No metrics collection (feature #8) — just episode boundary tracking and logging
- Must integrate with the async architecture from 005a (lock protocol, task structure)
- Existing tests must continue to pass

## New: `app/episode.py`

```python
class StepResult(BaseModel):
    obs: Any
    reward: float
    terminated: bool
    truncated: bool
    done: bool  # terminated or truncated

class EpisodeConfig(BaseModel):
    max_steps: int | None = None
    max_episodes: int | None = None
    auto_reset: bool = True

class EpisodeRunner:
    def __init__(self, app: CartPoleApp, policy: BaseActionPolicy, config: EpisodeConfig | None = None) -> None
    def step(self) -> StepResult
    def run(self) -> None
    def is_done(self) -> bool
    @property current_episode -> int
    @property total_steps -> int
    @property episode_steps -> int
```

## Modified: `app/cartpole.py`

- `apply_action` returns `StepResult` instead of `bool`
- `loop_policy_over_episodes` and `apply_policy_until_episode_end` replaced by `EpisodeRunner`

## Modified: `app/viewer.py`

- `simulate_and_render_sequence` uses `EpisodeRunner` with `max_steps=sequence_length`

## Files Touched / Created

| File | Action |
|------|--------|
| `app/episode.py` | **Create** — `StepResult`, `EpisodeConfig`, `EpisodeRunner` |
| `app/cartpole.py` | **Edit** — `apply_action` returns `StepResult`, remove episode loop methods |
| `app/viewer.py` | **Edit** — use `EpisodeRunner` in simulation methods |
| `tests/test_episode.py` | **Create** — unit tests for `EpisodeRunner` |
| `tests/test_cartpole.py` | **Edit** — update for `StepResult` return type |

## Key Tests

1. `test_step_returns_step_result` — `step()` returns a `StepResult` with correct fields
2. `test_auto_reset_on_terminated` — after episode ends, env is automatically reset
3. `test_episode_counter_increments` — `current_episode` increments on each reset
4. `test_total_steps_tracks` — `total_steps` counts across episodes
5. `test_episode_steps_resets` — `episode_steps` resets to 0 on new episode
6. `test_max_steps_stops` — runner stops after `max_steps` total steps
7. `test_max_episodes_stops` — runner stops after `max_episodes` episodes
8. `test_no_auto_reset` — with `auto_reset=False`, runner stops when episode ends
9. `test_is_done_false_initially` — `is_done()` returns `False` at start
10. `test_run_respects_max_steps` — `run()` executes correct number of steps
11. `test_cartpole_apply_action_returns_step_result` — updated return type

## Stages

### Stage 1: StepResult model and CartPoleApp update
- Create `app/episode.py` with `StepResult` and `EpisodeConfig`
- Update `app/cartpole.py` `apply_action` to return `StepResult`
- Update existing tests for new return type
- Run validation

### Stage 2: EpisodeRunner implementation
- Add `EpisodeRunner` to `app/episode.py`
- Write all `EpisodeRunner` tests
- Run validation

### Stage 3: Viewer integration
- Update `app/viewer.py` to use `EpisodeRunner`
- Run full suite + validation

## Progress

_(to be updated during implementation)_
