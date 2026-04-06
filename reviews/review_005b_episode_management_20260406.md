# Code Review: Feature 005b — Episode Management

**Branch:** `feature/004-episode-management`
**Date:** 2026-04-06
**Reviewer:** Claude (automated)

---

## 0. Summary of Changes Reviewed

The changeset introduces episode and batch management: Pydantic models (`StepResult`, `EpisodeConfig`, `BatchConfig`, `EpisodeResult`, `BatchResult`), an async `EpisodeRunner`, batch persistence (`.npz` + `.json`), a `Utilities` helper, `sim_speed_multiplier` support, and CLI integration in `__main__.py`.

---

## 1. Coding Guidelines Compliance — Issues Found & Fixed

### 1a. Import ordering violations (Severity: Low)

**Files:** `app/__main__.py`, `app/episode.py`, `app/render.py`

**Problem:** Imports were not in PEP 8 / isort order. Stdlib imports were mixed with third-party and local imports. In `app/render.py`, `import logging` appeared after a module-level constant, violating the E402 (module-level import not at top of file) rule.

**Fix:** Reordered all imports to follow: stdlib -> third-party -> local, grouped and alphabetized. Moved `import logging` above the `RENDER_DELAY_MS` constant in `render.py`.

### 1b. Unused imports (Severity: Low)

**Files:** `app/__main__.py` (`EpisodeResult`), `app/episode.py` (`deque`)

**Problem:** `EpisodeResult` was imported in `__main__.py` but never used. `deque` from `collections` was imported in `episode.py` but never used.

**Fix:** Removed both unused imports.

### 1c. Missing module docstring (Severity: Low)

**File:** `app/utilities.py`

**Problem:** No module-level docstring.

**Fix:** Added `"""Utility helpers."""` docstring.

### 1d. PyreFly type-checker failures on `__init__.py` re-exports (Severity: Low)

**Files:** `tests/test_episode.py`, `tests/test_code_cleanup.py`

**Problem:** Tests imported from `app.models` (the `__init__.py` re-export), which PyreFly could not resolve as `missing-module-attribute`.

**Fix:** Changed test imports to use explicit submodule paths (`app.models.episode`, `app.models.batch`).

### 1e. New classes missing from return-type-annotation test (Severity: Low)

**File:** `tests/test_code_cleanup.py`

**Problem:** `EpisodeRunner` and `Utilities` were not included in the parametrized test that checks all public methods have return type annotations.

**Fix:** Added both classes to the test parametrize list.

---

## 2. Bugs, Security Issues, Performance, and Style

### 2a. BUG — `episode_done` scoped inside `async with` block (Severity: High)

**File:** `app/episode.py:80-83`

**Problem:** `episode_done: bool = False` was declared *inside* the `async with self._app.lock:` block, but the `while not episode_done:` loop ran outside it. While Python's scoping rules mean this technically works (variables in `with` blocks are accessible after the block), the misleading indentation made the code appear as if the while loop was intended to be inside the lock, which would deadlock (re-acquiring a non-reentrant `asyncio.Lock`). This is a readability/correctness hazard — a future refactor could easily introduce a real bug here.

**Fix:** Moved `episode_done = False` declaration outside the `async with` block, with a blank line separating the lock-release from the while loop for clarity.

### 2b. BUG — Double environment reset (Severity: Medium)

**File:** `app/episode.py`

**Problem:** `run_episode` reset the env at the start (line 81) AND at the end via `auto_reset` (line 99-101). This meant every episode was being reset twice when running in batch mode — once at the end of episode N and once at the start of episode N+1. The double reset is wasteful and semantically confusing.

**Fix:** Removed end-of-episode `auto_reset` logic. The start-of-episode reset is the correct pattern — it guarantees each episode starts from a clean state regardless of what happened before. Removed the now-unused `auto_reset` field from `EpisodeConfig`.

### 2c. Redundant `sim_speed_multiplier` parameter (Severity: Medium)

**File:** `app/episode.py:84`

**Problem:** `run_episode` accepted `sim_speed_multiplier` as both a method parameter AND it existed as a field on `EpisodeConfig`. This created a risk of inconsistency — the caller could pass a different speed than what was in the config. In `run_batch`, the value was read from config and passed through, making the parameter redundant.

**Fix:** Removed the `sim_speed_multiplier` parameter from `run_episode`. The method now reads `config.sim_speed_multiplier` directly. Also cleaned up the no-op `* 1` multiplication in the sleep calculation.

### 2d. Generated `results/` directory not in `.gitignore` (Severity: Medium)

**File:** `.gitignore`

**Problem:** The `results/` directory containing `.npz` and `.json` batch output files was staged for commit. These are generated artifacts that should not be version-controlled — they are large, machine-specific, and change on every run.

**Fix:** Added `results/` to `.gitignore`.

---

## 3. Related Code Impact

### 3a. Tests updated for removed `auto_reset` field

**File:** `tests/test_episode.py`

Removed `test_run_episode_auto_resets` and `test_run_episode_no_auto_reset`. Replaced with `test_run_episode_resets_env_at_start` which verifies the new reset-at-start behavior. Updated `TestEpisodeConfig` to test `sim_speed_multiplier` instead of `auto_reset`.

### 3b. No missing test coverage identified

All existing `EpisodeRunner` methods have test coverage. The `Utilities` class only has one static method — it would benefit from a unit test but was not part of this changeset's scope.

---

## 4. Documentation

No documentation updates required. The README already describes the architecture correctly. The `auto_reset` config option was not documented externally.

---

## 5. Overall Summary

| Category | Items | Severity |
|----------|-------|----------|
| Bug fixes | 3 (scoping bug, double reset, redundant param) | High/Medium |
| Lint/style fixes | 5 (imports, unused imports, docstring, type-checker, test coverage) | Low |
| .gitignore | 1 (results/ directory) | Medium |

**Key improvements:**
- Fixed a misleading scoping pattern in `run_episode` that could easily become a real deadlock bug during refactoring
- Eliminated double environment resets that were wasting compute in batch runs
- Removed parameter/config redundancy that could lead to inconsistent sim speed values
- All validation passing: `ruff format`, `ruff check`, `pyrefly`, `pytest` (81 tests)
