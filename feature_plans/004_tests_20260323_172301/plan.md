# Feature Plan: Comprehensive Tests

## Goals
- Add unit tests for every public method in `CartPoleApp`, `CartPoleViewer`, `MatplotlibPlotter`, `MatplotlibAnimator`
- Add integration test for the full headless flow (env -> policy -> step -> frame capture)
- All rendering/display calls must be mocked so tests run headlessly and fast
- Achieve high coverage across `app/` modules

## Constraints
- Tests must be deterministic and fast (no real GUI windows)
- Mock `plt.show()`, `plt.imshow()`, and `FuncAnimation` where needed
- Use `pytest` with fixtures for env setup/teardown
- Follow TDD — but since code already exists, tests verify existing behaviour

## Detailed Plan

### `tests/test_cartpole.py` — CartPoleApp unit tests

| Test | What it verifies |
|------|------------------|
| `test_init_creates_env` | `self.env` is a gym.Env after init |
| `test_init_env_is_rgb_array` | render mode is `rgb_array` |
| `test_init_resets_env` | env is in a valid state (obs shape matches observation_space) |
| `test_apply_action_steps_env` | after `apply_action`, env has advanced (call doesn't raise) |
| `test_apply_action_logs_output` | log output contains expected fields (action, obs, reward) |
| `test_env_can_be_closed` | `app.env.close()` doesn't raise |

### `tests/test_policy.py` — Policy unit tests

| Test | What it verifies |
|------|------------------|
| `test_base_policy_is_abstract` | Can't instantiate `BaseActionPolicy` |
| `test_random_policy_action_in_space` | Returned action is valid for the env's action space |
| `test_random_policy_returns_int` | Return type is int |
| `test_random_policy_different_seeds` | Actions vary across calls (not always the same) |

### `tests/test_plot.py` — Plotter unit tests

| Test | What it verifies |
|------|------------------|
| `test_base_plotter_is_abstract` | Can't instantiate `BasePlotter` |
| `test_matplotlib_plotter_calls_imshow` | `plot()` calls `plt.imshow` with the frame |
| `test_matplotlib_plotter_calls_show` | `plot()` calls `plt.show()` |

### `tests/test_render.py` — Renderer unit tests

| Test | What it verifies |
|------|------------------|
| `test_base_renderer_is_abstract` | Can't instantiate `BaseRenderer` |
| `test_start_live_render_returns_figure` | Returns a matplotlib `Figure` |
| `test_render_from_frames_calls_func_animation` | `FuncAnimation` is called with correct args |
| `test_render_from_frames_calls_show` | `plt.show()` is called |
| `test_continuously_live_render_calls_func_animation` | `FuncAnimation` is called for live render |
| `test_update_frame_returns_list` | `update_frame` returns a list of artists |

### `tests/test_viewer.py` — CartPoleViewer unit tests

| Test | What it verifies |
|------|------------------|
| `test_viewer_init_creates_components` | Has `.app`, `.plotter`, `.renderer`, `.config` |
| `test_viewer_config_defaults` | `ViewerConfig` has expected default `tick_length_ms` |
| `test_plot_calls_plotter` | `viewer.plot()` delegates to `self.plotter.plot()` |
| `test_simulate_and_render_collects_frames` | Correct number of frames passed to renderer |
| `test_simulate_defaults_to_random_policy` | When no policy given, uses `RandomActionPolicy` |

### `tests/test_integration.py` — Integration test

| Test | What it verifies |
|------|------------------|
| `test_full_headless_flow` | Create app -> apply N actions with random policy -> capture N frames -> frames are valid numpy arrays with correct shape |

## Files Touched / Created

| File | Action |
|------|--------|
| `tests/test_cartpole.py` | **Create** |
| `tests/test_policy.py` | **Create** |
| `tests/test_plot.py` | **Create** |
| `tests/test_render.py` | **Create** |
| `tests/test_viewer.py` | **Create** |
| `tests/test_integration.py` | **Create** |

Existing test files (`test_logging_config.py`, `test_module_separation.py`, `test_code_cleanup.py`) are unchanged.

## Classes and Functions

No new production classes or functions. Only test classes.

## Key Tests

All tests listed in the tables above — 25+ new tests total.

## Stages

### Stage 1: CartPoleApp + Policy tests
- Create `tests/test_cartpole.py` and `tests/test_policy.py`
- Run validation

### Stage 2: Plot + Render tests
- Create `tests/test_plot.py` and `tests/test_render.py`
- Run validation

### Stage 3: Viewer + Integration tests
- Create `tests/test_viewer.py` and `tests/test_integration.py`
- Run full test suite + validation

## Progress

### Stage 1: CartPoleApp + Policy tests — COMPLETE (2026-03-23)
- [x] Created `tests/test_cartpole.py` — 6 tests
- [x] Created `tests/test_policy.py` — 4 tests
- [x] All passing, validation clean

### Stage 2: Plot + Render tests — COMPLETE (2026-03-23)
- [x] Created `tests/test_plot.py` — 3 tests (mocked plt)
- [x] Created `tests/test_render.py` — 6 tests (mocked plt + FuncAnimation)
- [x] All passing, validation clean

### Stage 3: Viewer + Integration tests — COMPLETE (2026-03-23)
- [x] Created `tests/test_viewer.py` — 5 tests
- [x] Created `tests/test_integration.py` — 1 integration test (full headless flow)
- [x] Full suite: 56/56 passed
- [x] ruff format/check: clean
- [x] pyrefly: 0 errors
