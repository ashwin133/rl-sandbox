# Task Log — .ai_agents/task.md

## Feature 005a: Async Architecture — DONE

Branch: `feature/004-episode-management`

- [x] Decoupled simulation + viewing into async tasks via `asyncio.TaskGroup`
- [x] DI of env + lock into `CartPoleViewer`
- [x] Episode reset logic in `CartPoleApp` (`loop_policy_over_episodes`, `apply_policy_until_episode_end`)
- [x] Explicit figure/axes ownership in `MatplotlibAnimator` (`_ensure_fig` with `plt.subplots`)
- [x] Fixed duplicate matplotlib figure bug
- [x] Tests updated and passing (55/55)
- [x] Validation pipeline passing (ruff format, ruff check, pyrefly, pytest)
- [x] Plan split into 005a (async) and 005b (episode management) with PlantUML diagrams

## Feature 005b: Episode Management — DONE

- [x] `StepResult`, `EpisodeConfig`, `BatchConfig`, `EpisodeResult`, `BatchResult` models
- [x] `CartPoleApp.apply_action` returns `StepResult`
- [x] `EpisodeRunner` with async `run_episode` (lock-aware, auto-reset, max_steps)
- [x] `run_batch` for sequential multi-episode runs
- [x] Persistence: `log_batch_to_file` (.npz + .json) and `load_batch_from_file`
- [x] Integration: `__main__.py` uses `EpisodeRunner`, old loop methods removed from `CartPoleApp`
- [x] 80/80 tests passing, full validation pipeline green
