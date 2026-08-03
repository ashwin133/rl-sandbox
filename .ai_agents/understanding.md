# Architecture — .ai_agents/understanding.md

## Overview

`rl-sandbox` is a toolbox for implementing, training, evaluating, and comparing reinforcement learning algorithms on Gymnasium environments. It separates algorithm-specific trainers and policies from reusable environment execution, metrics, checkpoints, replay, persistence, and visualisation infrastructure.

REINFORCE is the first complete algorithm implementation. `CartPole-v1` and `Acrobot-v1` are the initial registered discrete-action environments used to exercise the toolbox; neither defines the repository's intended scope.

## Extension Boundaries

- **Training algorithms** implement `BaseTrainer` and return the shared `TrainingResult` contract. Algorithm-specific code belongs under `rl_sandbox/training/`.
- **Runtime policies** implement `BaseActionPolicy`, allowing `EpisodeRunner` and the viewer to execute an algorithm without depending on its trainer.
- **Checkpoints** include an `algorithm` discriminator, environment id, network dimensions, and weights. Each algorithm loader validates compatibility before constructing a policy.
- **Environments** are declared through `EnvSpec` in `SUPPORTED_ENVS` and wrapped by `GymEnvApp`. Current support assumes a vector `Box` observation space and discrete actions.
- **Evaluation data** uses shared Pydantic models for episode metrics, deterministic evaluation results, training results, recordings, and replay data.

## Async Architecture

Live simulation uses two concurrent tasks coordinated by an `asyncio.Lock`:

1. **Simulation task** (`EpisodeRunner.run_episode`) steps the environment through a `BaseActionPolicy`, then handles termination and reset.
2. **Viewing task** (`EnvViewer.render_live`) snapshots environment frames under the same lock and displays them with matplotlib.

Both tasks share the `gym.Env` owned by `GymEnvApp`. The lock limits mutual exclusion to `env.step()`, `env.reset()`, and `env.render()` calls. Short sleeps after releasing the lock yield control to the event loop. Matplotlib integrates with asyncio through a non-blocking window and a manual event-loop pump.

## Module Structure

```text
rl_sandbox/
├── __main__.py          # CLI dispatcher
├── cli/
│   ├── run_sim.py       # Live simulation for a selected env and policy
│   ├── train.py         # Current REINFORCE training command
│   └── playback.py      # Recorded-frame playback command
├── envs/
│   ├── base.py          # BaseEnvApp environment-wrapper contract
│   ├── gym_env.py       # Generic GymEnvApp implementation
│   └── registry.py      # EnvSpec and SUPPORTED_ENVS
├── models/
│   ├── episode.py       # Step and episode contracts
│   ├── batch.py         # Batch execution contracts
│   ├── recording.py     # On-disk recording metadata
│   ├── replay.py        # In-memory replay data
│   └── training.py      # Training and evaluation metrics
├── policy/
│   ├── base.py          # BaseActionPolicy contract
│   ├── random.py        # Random baseline policy
│   └── reinforce.py     # Trained REINFORCE policy adapter
├── training/
│   ├── base.py          # BaseTrainer algorithm contract
│   ├── reinforce.py     # REINFORCE trainer and evaluation loop
│   ├── networks.py      # Current policy network
│   └── checkpoint.py    # Algorithm-aware checkpoint schema
├── episode.py           # Async episode and batch execution
├── file_ops.py          # Recording persistence
├── replay.py            # Environment-independent frame replay
├── viewer.py            # Live visualisation orchestration
├── render.py            # Frame animation and display
├── plot.py              # Static frame plotting
├── utilities.py         # Output-directory utilities
└── logging_config.py    # Logging configuration
```

## Key Components

- **`BaseTrainer` / `ReinforceTrainer`** — common training lifecycle and the first algorithm implementation. REINFORCE collects full trajectories, calculates discounted normalised returns, updates policy weights, records entropy, and runs deterministic evaluations.
- **`BaseActionPolicy` / policies** — environment-facing action selection. `RandomActionPolicy` is a baseline; `ReinforcePolicy` wraps a trained network and validates checkpoint compatibility.
- **`BaseEnvApp` / `GymEnvApp`** — environment ownership and concurrency boundary, parameterised by `EnvSpec` rather than a specific task.
- **`EpisodeRunner`** — algorithm-agnostic episode and batch execution with optional frame capture.
- **`EnvSpec` / `SUPPORTED_ENVS`** — environment metadata for timing, reward thresholds, rolling evaluation windows, and episode caps.
- **`Checkpoint`** — self-describing model payload with an open-ended algorithm name so new algorithms can add loaders without changing the schema.
- **`ReplayRunner` / `FileOps`** — recording persistence and replay independent of a live environment.
- **`EnvViewer` / `MatplotlibAnimator`** — live display using injected environment and renderer dependencies.

## Documentation

MkDocs and mkdocstrings generate the public API site from source signatures and docstrings. `mkdocs.yml` defines the navigation, `docs/` holds the landing and API pages, and `.github/workflows/docs.yml` publishes strict builds to the `gh-pages` branch after pushes to `main`.

## Validation

Run the checks in this order:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyrefly check
uv run pytest -v
uv run mkdocs build --strict
```

Tests use pytest, mock matplotlib at `rl_sandbox.render`, and run async components with `asyncio.run()` plus explicit task cancellation where needed.
