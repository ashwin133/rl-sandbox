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

## Feature 006: Playback from Logs — DONE

- [x] `EpisodeRecording`, `BatchRecording` models for disk persistence metadata
- [x] `EpisodeReplay`, `BatchReplay` dataclasses for in-memory replay objects
- [x] `replay_freq` on `BatchConfig` — controls which episodes get frames captured at sim time
- [x] `run_episode(capture_frames=True)` grabs `env.render()` under same lock as `env.step()`
- [x] `run_batch` selects episodes by `replay_freq` for frame capture
- [x] `FileOps` class — `save` (frames .npz + metadata .json) and `load_batch_replay`
- [x] Old persistence methods removed from `EpisodeRunner`
- [x] `ReplayRunner` — pure frame playback via `init_display()` / `display_frame()` on renderer
- [x] Replay speed read from recording metadata, not CLI
- [x] `init_display()` and `display_frame()` added to `BaseRenderer` / `MatplotlibAnimator`
- [x] CLI: `run-sim --replay-freq N`, `playback-from-logs --path <dir>`
- [x] Old incompatible log files deleted
- [x] 101/101 tests passing, full validation pipeline green

## Feature 007: RL Training from Scratch — DONE

- [x] `PolicyNetwork` and `QNetwork` nn.Modules (two hidden layers, ReLU)
- [x] `TrainingConfig`, `EpisodeMetrics`, `TrainingResult` pydantic models
- [x] `ReplayBuffer` with `Transition` namedtuple (circular deque, random sampling)
- [x] `BaseTrainer` ABC
- [x] `ReinforceTrainer` — full REINFORCE (Monte Carlo policy gradient) training loop
- [x] `DqnTrainer` — full DQN training loop (replay buffer, target network, ε-greedy)
- [x] `ReinforcePolicy` and `DqnPolicy` as `BaseActionPolicy` subclasses
- [x] Save/load via PyTorch state_dict
- [x] CLI: `python -m app train --algorithm reinforce|dqn --episodes N --lr X`
- [x] 148/148 tests passing, full validation pipeline green
