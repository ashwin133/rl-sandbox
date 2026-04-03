# Feature 005a: Async Architecture

## Goals

- Decouple simulation and rendering into two independent async tasks
- Coordinate shared `gym.Env` access via `asyncio.Lock`
- Integrate matplotlib with asyncio using non-blocking rendering
- Use dependency injection so `CartPoleViewer` does not own `CartPoleApp`
- Use the object-oriented matplotlib API to prevent implicit figure creation

## Constraints

- Single-threaded — all concurrency via `asyncio` cooperative multitasking
- `CartPoleApp` owns the `gym.Env` and `asyncio.Lock`
- `CartPoleViewer` borrows `env` and `lock` — it never creates or resets the environment
- The lock is only held during `env.step()`, `env.reset()`, or `env.render()` — never during sleeps or GUI work
- `MatplotlibAnimator` owns exactly one `(Figure, Axes)` pair via `_ensure_fig()` using `plt.subplots()`

## Architecture

### Entry Point (`__main__.py`)

```python
app = CartPoleApp()
viewer = CartPoleViewer(env=app.env, lock=app.lock)

async def main() -> None:
    async with asyncio.TaskGroup() as tg:
        tg.create_task(viewer.render_live())
        tg.create_task(app.loop_policy_over_episodes(RandomActionPolicy()))

asyncio.run(main())
```

### Lock Protocol

| Holder | Operation | Duration |
|--------|-----------|----------|
| Simulation | `env.step(action)` | ~microseconds |
| Simulation | `env.reset()` | ~microseconds |
| Viewer | `env.render()` | ~microseconds |
| Nobody | sleeps, GUI event pump | 1–20ms |

Each task releases the lock then calls `await asyncio.sleep()` — this yields to the event loop, preventing starvation.

## Sequence Diagrams

See PlantUML sources in this directory. Render with `plantuml *.puml`.

1. **`sd_task_startup.puml`** — `__main__` creates `CartPoleApp` and `CartPoleViewer`, kicks off both async tasks via `TaskGroup`
2. **`sd_viewing_task.puml`** — `CartPoleViewer.render_live()` spawns `refresh_frame` sub-task, delegates to `MatplotlibAnimator.start_live_render()`, shows `FuncAnimation` callback flow
3. **`sd_simulation_task.puml`** — `CartPoleApp.loop_policy_over_episodes()` with two cases: env step (episode continues) and episode termination with reset

## Files Touched

| File | Change |
|------|--------|
| `app/__main__.py` | Rewritten — `asyncio.TaskGroup` entry point |
| `app/cartpole.py` | Added `asyncio.Lock`, `apply_action() -> bool`, `reset_env()`, `loop_policy_over_episodes()`, `apply_policy_until_episode_end()` |
| `app/viewer.py` | DI of `env` + `lock`, `render_live()` async with `refresh_frame` sub-task |
| `app/render.py` | `BaseRenderer.update_frame()` abstract method, `MatplotlibAnimator._ensure_fig()` with `plt.subplots()`, `start_live_render()` async with manual event pump |
| `tests/test_cartpole.py` | Updated for `apply_action() -> bool` |
| `tests/test_render.py` | Updated mocks for `plt.subplots()`, async `start_live_render` |
| `tests/test_viewer.py` | DI of `env` in fixtures |
| `tests/test_module_separation.py` | Updated `CartPoleViewer` instantiation |
| `tests/test_code_cleanup.py` | `test_apply_action_returns_bool` |

## Key Tests

1. `test_apply_action_steps_env` — `apply_action` steps the environment
2. `test_apply_action_does_not_raise` — calling `apply_action` twice (covers reset path)
3. `test_apply_action_returns_bool` — return type annotation is `bool`
4. `test_init_creates_components` — viewer receives `env` via DI
5. `test_start_live_render_calls_func_animation` — async renderer creates `FuncAnimation`
6. `test_update_frame_returns_list` — `update_frame` returns `[AxesImage]`
7. `test_render_from_frames_calls_func_animation` — batch renderer works with `_ensure_fig`
8. `test_viewer_composes_correctly` — viewer composes renderer + plotter correctly

## Progress

- [x] `CartPoleApp` owns env + lock, provides `apply_action`, `reset_env`, `loop_policy_over_episodes`
- [x] `CartPoleViewer` takes env + lock via DI, `render_live` is async
- [x] `MatplotlibAnimator._ensure_fig()` with `plt.subplots()` — explicit figure ownership
- [x] `BaseRenderer.update_frame()` added to abstract interface
- [x] Tests updated and passing (55/55)
- [x] Validation pipeline passing (ruff format, ruff check, pyrefly, pytest)
