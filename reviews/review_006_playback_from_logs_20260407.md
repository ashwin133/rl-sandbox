# Code Review: Feature 006 — Playback from Logs

**Branch:** `feature/004-episode-management`
**Date:** 2026-04-07
**Reviewer:** Claude (automated)

---

## 0. Summary of Changes Reviewed

This review covers all uncommitted changes on the branch, including Feature 006 (playback from logs) and in-session refactoring. The changeset introduces:

- `EpisodeRecording`, `BatchRecording` models for disk persistence metadata
- `EpisodeReplay`, `BatchReplay` dataclasses for in-memory replay
- `FileOps` class for save/load of frames (`.npz`) and metadata (`.json`)
- `ReplayRunner` for pure frame playback via `BaseRenderer`
- `replay_freq` on `BatchConfig` for selective frame capture
- `init_display()` and `display_frame()` on `BaseRenderer` / `MatplotlibAnimator`
- CLI: `run-sim --replay-freq N` and `playback-from-logs --path <dir>`
- Codebase-wide refactoring: inline imports moved to top-of-file, type hints added, magic strings extracted to constants, `__init__.py` re-exports removed

---

## 1. Coding Guidelines Compliance — Issues Found & Fixed

### 1a. f-strings in logging calls (Severity: Medium)

**Files:** `app/__main__.py:60-62`, `app/cartpole.py:57`

**Problem:** f-strings in `logger.warning()` and `logger.info()` calls eagerly evaluate string formatting even when the log level is disabled. This is both a minor performance issue and a violation of Python logging best practice (PEP 282).

**Fix:** Replaced with lazy `%s` / `%.2f` formatting, passing arguments to the logger.

### 1b. Formatting violation in file_ops.py (Severity: Low)

**File:** `app/file_ops.py`

**Problem:** Long line in `load_batch_replay` not conforming to ruff format.

**Fix:** Ran `ruff format` to reformat.

---

## 2. Bugs, Security Issues, Performance, and Style

### 2a. No bugs found

The core save/load logic is correct. The `replay_episodes` key in `.npz` correctly maps positional indices to episode indices, and the load path correctly reconstructs the lookup.

### 2b. `type: ignore` suppressions (Severity: Low, existing)

**Files:** `app/episode.py:92`, `app/viewer.py:42,56,69`

**Problem:** Four `# type: ignore` comments exist. These are all related to `gymnasium.Env.render()` returning `RenderFrame | list[RenderFrame] | None` while the code expects `np.ndarray`. This is a known gymnasium typing limitation and the suppressions are justified.

**Recommendation:** No action needed, but note for future — if gymnasium improves their type stubs, these can be removed.

### 2c. Hardcoded `no_episodes=10` in `__main__.py:70` (Severity: Low)

The number of episodes is hardcoded. Consider adding a `--episodes` CLI argument for flexibility. Not blocking.

### 2d. Hardcoded `"./results"` path in `__main__.py:74` (Severity: Low)

The results base path is hardcoded. Consider a `--output-dir` CLI argument. Not blocking.

### 2e. `ReplayRunner` replays episodes with empty frames (Severity: Low)

Since metadata is now recorded for all episodes but frames only for `replay_freq` episodes, `replay_batch` will iterate over episodes with empty frame lists. The loop in `replay_episode` is a no-op for empty frames, so no crash — but it logs "Replaying episode X (Y steps, Zx speed)" then immediately "Episode X complete" which could be confusing.

**Recommendation:** Consider filtering to only replay episodes that have frames, or skip logging for frameless episodes. Not blocking.

---

## 3. Related Code Impact

### 3a. `models/__init__.py` and `policy/__init__.py` cleared

All imports across the codebase were updated to import directly from source modules (e.g., `from app.policy.base import BaseActionPolicy` instead of `from app.policy import BaseActionPolicy`). This applies to all app and test files. `test_module_separation.py` import tests were also updated to test the direct paths.

### 3b. Inline imports eliminated

All inline/deferred imports across `app/file_ops.py`, `app/__main__.py`, `tests/test_code_cleanup.py`, `tests/test_render.py`, and `tests/test_replay.py` were moved to the top of their respective files.

### 3c. Type hints added across codebase

Type hints were added to local variables and instance attributes in: `cartpole.py`, `viewer.py`, `logging_config.py`, `utilities.py`, `replay.py`, `file_ops.py`.

### 3d. Magic strings extracted to constants

In `file_ops.py`, repeated string literals (`"frames.npz"`, `"metadata.json"`, `"replay_episodes"`, `"frames_"`) were extracted to module-level constants: `FRAMES_FILE_NAME`, `METADATA_FILE_NAME`, `REPLAY_EPISODES_KEY`, `FRAMES_KEY_PREFIX`.

### 3e. Test coverage

All new code has tests:
- `test_file_ops.py` — tests for `FileOps.save`, `FileOps.load_batch_replay`, roundtrip, recording models, frame capture
- `test_replay.py` — tests for `EpisodeReplay`, `BatchReplay`, `ReplayRunner` (frame display calls, batch replay, sleep scaling)
- Existing tests updated to match refactored imports

---

## 4. Documentation

- `app/__main__.py:52` help string references `metadata.json` and `frames.npz` — these should ideally reference the constants but since it's user-facing CLI help text, literal names are acceptable here.
- `.ai_agents/task.md` and `.ai_agents/understanding.md` are up to date with the Feature 006 work.

---

## 5. Validation Pipeline

| Check | Result |
|-------|--------|
| `ruff format` | Pass (32 files formatted) |
| `ruff check` | Pass (all checks passed) |
| `pyrefly check` | Pass (0 errors, 10 suppressed) |
| `pytest` | Pass (101/101) |

---

## 6. Overall Summary

The Feature 006 implementation is solid — clean separation between recording metadata and frame data, proper async architecture, and good test coverage. The in-session refactoring improved codebase consistency significantly:

- **f-string logging** fixed (medium severity)
- **Formatting** fixed (low severity)
- **No bugs** found in new code
- **Import hygiene** improved: no more inline imports, no `__init__.py` re-exports, all direct imports
- **Type safety** improved with comprehensive type hints
- **Magic strings** eliminated in `file_ops.py`

Minor recommendations (non-blocking): add CLI args for `--episodes` and `--output-dir`, and consider skipping frameless episodes during replay logging.
