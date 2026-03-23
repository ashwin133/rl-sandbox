# Feature Plan: Module Separation

## Goals
- Split the monolithic `viewer.py` into single-responsibility modules: `render.py`, `plot.py`, and a slimmed-down `viewer.py` (orchestrator)
- Move policy classes into an `app/policy/` package for extensibility
- All existing functionality must continue to work identically after the split
- No new features — purely structural refactor

## Constraints
- No behaviour changes — this is a move-and-rewire refactor only
- All imports in `__main__.py` and `cartpole.py` must be updated to new locations
- Existing class APIs stay the same (signatures, return types)
- Tests must prove nothing broke

## Detailed Plan

### Current structure
```
app/
  __init__.py
  __main__.py
  action_policy.py    # BaseActionPolicy, RandomActionPolicy
  cartpole.py         # CartPoleApp
  logging_config.py   # LoggingConfig, ColourFormatter, LoggerFactory
  viewer.py           # BasePlotter, MatplotlibPlotter, BaseRenderer,
                      #   MatplotlibAnimator, ViewerConfig, CartPoleViewer
```

### Target structure
```
app/
  __init__.py
  __main__.py
  cartpole.py         # CartPoleApp (unchanged, update import path for policy)
  logging_config.py   # unchanged
  plot.py             # BasePlotter, MatplotlibPlotter
  render.py           # BaseRenderer, MatplotlibAnimator
  viewer.py           # ViewerConfig, CartPoleViewer (imports from plot + render)
  policy/
    __init__.py       # re-exports BaseActionPolicy, RandomActionPolicy
    base.py           # BaseActionPolicy
    random.py         # RandomActionPolicy
```

### What moves where

| Class | From | To |
|---|---|---|
| `BasePlotter` | `viewer.py` | `app/plot.py` |
| `MatplotlibPlotter` | `viewer.py` | `app/plot.py` |
| `BaseRenderer` | `viewer.py` | `app/render.py` |
| `MatplotlibAnimator` | `viewer.py` | `app/render.py` |
| `ViewerConfig` | `viewer.py` | `app/viewer.py` (stays) |
| `CartPoleViewer` | `viewer.py` | `app/viewer.py` (stays) |
| `BaseActionPolicy` | `action_policy.py` | `app/policy/base.py` |
| `RandomActionPolicy` | `action_policy.py` | `app/policy/random.py` |

### Import updates required

| File | Change |
|---|---|
| `app/cartpole.py` | `from app.action_policy import BaseActionPolicy` -> `from app.policy import BaseActionPolicy` |
| `app/viewer.py` | Remove plotter/renderer class definitions, add `from app.plot import ...` and `from app.render import ...`. Update policy imports to `from app.policy import ...` |
| `app/__main__.py` | `from app.action_policy import RandomActionPolicy` -> `from app.policy import RandomActionPolicy` |
| `tests/test_logging_config.py` | No change needed |

### Files Touched / Created

| File | Action |
|---|---|
| `app/plot.py` | **Create** — `BasePlotter`, `MatplotlibPlotter` |
| `app/render.py` | **Create** — `BaseRenderer`, `MatplotlibAnimator` |
| `app/policy/__init__.py` | **Create** — re-exports `BaseActionPolicy`, `RandomActionPolicy` |
| `app/policy/base.py` | **Create** — `BaseActionPolicy` |
| `app/policy/random.py` | **Create** — `RandomActionPolicy` |
| `app/viewer.py` | **Edit** — remove moved classes, add imports from `plot` + `render` + `policy` |
| `app/cartpole.py` | **Edit** — update policy import path |
| `app/__main__.py` | **Edit** — update policy import path |
| `app/action_policy.py` | **Delete** — replaced by `app/policy/` package |
| `tests/test_module_separation.py` | **Create** — verify imports resolve, classes instantiate |

## Classes and Functions

No new classes or functions. All existing classes are moved as-is. The only new code is `app/policy/__init__.py` which re-exports:

```python
from app.policy.base import BaseActionPolicy
from app.policy.random import RandomActionPolicy

__all__ = ["BaseActionPolicy", "RandomActionPolicy"]
```

## Diagrams

Class diagram and sequence diagram show the new module boundaries and how `CartPoleViewer` depends on the separated modules.

Diagrams generated as PNGs: `class_diagram.png`, `sequence_diagram.png`

## Key Tests

1. `test_policy_import` — `from app.policy import BaseActionPolicy, RandomActionPolicy` works
2. `test_plot_import` — `from app.plot import BasePlotter, MatplotlibPlotter` works
3. `test_render_import` — `from app.render import BaseRenderer, MatplotlibAnimator` works
4. `test_viewer_import` — `from app.viewer import CartPoleViewer, ViewerConfig` works
5. `test_old_action_policy_removed` — `app/action_policy.py` no longer exists
6. `test_cartpole_uses_new_policy_path` — `CartPoleApp` can be instantiated (imports resolve)
7. `test_viewer_composes_correctly` — `CartPoleViewer` has `.plotter` (BasePlotter), `.render` (BaseRenderer), `.app` (CartPoleApp)

## Stages

### Stage 1: Create new modules and policy package
- Create `app/plot.py` with `BasePlotter`, `MatplotlibPlotter`
- Create `app/render.py` with `BaseRenderer`, `MatplotlibAnimator`
- Create `app/policy/` package with `base.py`, `random.py`, `__init__.py`
- Write tests
- Run validation

### Stage 2: Rewire imports and delete old file
- Update `app/viewer.py` — remove moved classes, import from new modules
- Update `app/cartpole.py` — policy import path
- Update `app/__main__.py` — policy import path
- Delete `app/action_policy.py`
- Run all tests (new + existing `test_logging_config.py`)
- Run validation

## Progress

### Stage 1: Create new modules and policy package — COMPLETE (2026-03-23)
- [x] Created `app/plot.py` with `BasePlotter`, `MatplotlibPlotter`
- [x] Created `app/render.py` with `BaseRenderer`, `MatplotlibAnimator`
- [x] Created `app/policy/` package with `base.py`, `random.py`, `__init__.py`
- [x] Wrote 9 tests — all passed
- [x] ruff format/check, pyrefly: all clean

### Stage 2: Rewire imports and delete old file — COMPLETE (2026-03-23)
- [x] Rewrote `app/viewer.py` — removed moved classes, imports from `plot`, `render`, `policy`
- [x] Fixed mutable default args (`action_policy=RandomActionPolicy()` -> `None` + factory)
- [x] Renamed `self.render` -> `self.renderer` to avoid shadowing method name
- [x] Updated `app/cartpole.py` — removed unused imports, policy from `app.policy`
- [x] Updated `app/__main__.py` — cleaned imports, policy from `app.policy`
- [x] Deleted `app/action_policy.py`
- [x] Added 4 more tests (old file removed, viewer composition, cartpole import, viewer import)
- [x] 22/22 tests passed
- [x] ruff format/check: all clean
- [x] pyrefly: 0 errors (6 suppressed — pre-existing gym render type narrowing)
