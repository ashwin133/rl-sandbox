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

## Feature 007: REINFORCE Training from Scratch — DONE

- [x] `PolicyNetwork` nn.Module (two hidden layers, ReLU, Softmax output)
- [x] `TrainingConfig`, `EpisodeMetrics`, `EvalResult`, `TrainingResult` pydantic models
- [x] `BaseTrainer` ABC
- [x] `ReinforceTrainer` — full REINFORCE (Monte Carlo policy gradient) training loop
- [x] Per-episode policy entropy tracking
- [x] Periodic deterministic (argmax) evaluation passes
- [x] `ReinforcePolicy` as `BaseActionPolicy` subclass
- [x] Save/load via PyTorch state_dict
- [x] CLI: `python -m app train --episodes N --lr X --init-model PATH`
- [x] CLI: `__main__.py` split into per-mode modules under `app/cli/`
- [x] DQN work stashed on side branch `wip/dqn-and-reinforce`
- [x] `learnings/` docs covering loss equation, gradient descent + backprop,
      problems with the REINFORCE loss, and the baseline bias in CartPole
- [x] 126/126 tests passing

## Feature 008: Generic Environment Support — DONE

- [x] `EnvSpec` pydantic model with per-env metadata (id, tau, solved threshold, max steps)
- [x] `SUPPORTED_ENVS` registry — initial entries: CartPole-v1, Acrobot-v1
- [x] `BaseEnvApp` ABC + `GymEnvApp` concrete (replaces `CartPoleApp`)
- [x] `EpisodeRunner` accepts `BaseEnvApp`; physics tick from `app.spec.physics_tick_s`
- [x] `BatchRecording` and `BatchReplay` embed full `EnvSpec` (replay self-describing)
- [x] `ReinforceTrainer` parameterised by `EnvSpec` — solved threshold, rolling window from spec
- [x] `TrainingResult` carries `env_spec` field for analysis
- [x] Self-describing checkpoint format: dict with state_dict, env_id, sizes, hidden_size
- [x] `ReinforcePolicy.load(path, expected_env_id)` rejects mismatched envs
- [x] `--env` required on `train` and `run-sim` CLI commands
- [x] Saved-model filename includes env id (e.g., `cartpole-v1_best.pt`)
- [x] `CartPoleViewer` renamed to `EnvViewer`
- [x] CartPole policy re-trained in new format
- [x] All tests passing including new env-registry, GymEnvApp, and checkpoint-validation tests
