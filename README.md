# rl-sandbox

A toolbox for implementing, training, evaluating, and comparing reinforcement learning algorithms on [Gymnasium](https://gymnasium.farama.org/) environments.

The shared environment, episode, metrics, checkpoint, replay, and visualisation layers are designed to support algorithm experiments without rebuilding the surrounding infrastructure. REINFORCE is the first implemented trainer; `CartPole-v1` and `Acrobot-v1` are the initial registered environments, not the boundary of the project.

[Browse the public API documentation](https://ashwin133.github.io/demo-rl-1/).

## Setup

```bash
uv sync
```

## CLI overview

The CLI exposes three subcommands via `python -m rl_sandbox`:

| Subcommand | Purpose |
|------------|---------|
| `train` | Train a REINFORCE policy on a chosen env |
| `run-sim` | Run a live simulation with a random or trained policy |
| `playback-from-logs` | Replay frames captured during a previous `run-sim` |

`--env` is required on `train` and `run-sim`. Saved models are written to `./models/{env_id_lower}_best.pt` (e.g. `cartpole-v1_best.pt`, `acrobot-v1_best.pt`).

## Train

```bash
# CartPole — solves in roughly 600–800 episodes
python -m rl_sandbox train --env CartPole-v1

# Acrobot — harder; vanilla REINFORCE is high-variance, give it more episodes
python -m rl_sandbox train --env Acrobot-v1 --episodes 3000

# Tune hyperparameters
python -m rl_sandbox train --env CartPole-v1 --episodes 2000 --lr 5e-4 --hidden-size 256

# Custom output directory (avoid overwriting the default)
python -m rl_sandbox train --env CartPole-v1 --output-dir ./models/experiments/run-001

# Warm-start from an existing checkpoint (algorithm + env_id + hidden-size must match)
python -m rl_sandbox train --env CartPole-v1 --episodes 500 --init-model ./models/cartpole-v1_best.pt
```

Full options: `python -m rl_sandbox train --help`.

New algorithms should implement `BaseTrainer` for training and expose a `BaseActionPolicy` implementation for simulation. The checkpoint schema carries an algorithm discriminator so loaders can reject incompatible model files explicitly.

## Run sim

```bash
# Random policy on CartPole — no model needed
python -m rl_sandbox run-sim --env CartPole-v1 --policy random

# Trained REINFORCE policy
python -m rl_sandbox run-sim --env CartPole-v1 \
    --policy reinforce \
    --model-path ./models/cartpole-v1_best.pt

# Same on Acrobot
python -m rl_sandbox run-sim --env Acrobot-v1 \
    --policy reinforce \
    --model-path ./models/acrobot-v1_best.pt

# Playback speed and episode count
python -m rl_sandbox run-sim --env CartPole-v1 \
    --policy reinforce \
    --model-path ./models/cartpole-v1_best.pt \
    --sim-speed 2.0 \
    --episodes 20

# Slow motion (0.25x) — useful for inspecting failures
python -m rl_sandbox run-sim --env CartPole-v1 \
    --policy reinforce \
    --model-path ./models/cartpole-v1_best.pt \
    --sim-speed 0.25

# Capture every 5th episode's frames for later playback
python -m rl_sandbox run-sim --env CartPole-v1 \
    --policy reinforce \
    --model-path ./models/cartpole-v1_best.pt \
    --episodes 50 \
    --replay-freq 5
```

Full options: `python -m rl_sandbox run-sim --help`.

### Checkpoint safety

Loading a checkpoint verifies both `algorithm` and `env_id` against what the loader expects. Mismatches raise rather than silently producing garbage:

```bash
# Fails: CartPole model on Acrobot run
python -m rl_sandbox run-sim --env Acrobot-v1 --policy reinforce \
    --model-path ./models/cartpole-v1_best.pt
# ValueError: Checkpoint env_id 'CartPole-v1' does not match expected env 'Acrobot-v1'
```

## Playback from logs

`run-sim --replay-freq N` captures every Nth episode's frames into `./results/<batch_dir>/`. Replay them with:

```bash
python -m rl_sandbox playback-from-logs --path ./results/<batch_dir>
```

The recording embeds the env's `EnvSpec`, so playback tick is reconstructed automatically — no `--env` needed.

## Suggested first run

```bash
# 1. Train CartPole baseline
python -m rl_sandbox train --env CartPole-v1 --episodes 1000

# 2. Watch
python -m rl_sandbox run-sim --env CartPole-v1 --policy reinforce \
    --model-path ./models/cartpole-v1_best.pt --episodes 5

# 3. Try Acrobot (longer, may not solve)
python -m rl_sandbox train --env Acrobot-v1 --episodes 3000

# 4. Watch
python -m rl_sandbox run-sim --env Acrobot-v1 --policy reinforce \
    --model-path ./models/acrobot-v1_best.pt --episodes 5
```

## Extending the toolbox

Algorithm implementations live under `rl_sandbox/training/`; policies used by simulation live under `rl_sandbox/policy/`. The existing REINFORCE modules provide one complete implementation of those extension points, including metrics, deterministic evaluation, checkpoint persistence, and trained-policy loading.

### Adding a new env

The registry `rl_sandbox/envs/registry.py` is the single source of truth:

```python
SUPPORTED_ENVS["MountainCar-v0"] = EnvSpec(
    env_id="MountainCar-v0",
    physics_tick_s=...,
    solved_reward_threshold=...,
    default_max_steps=...,
)
```

CLI choices are derived from `SUPPORTED_ENVS.keys()`. Constraints today: discrete action space, `Box` observation space (vector). Image-input envs (Atari) need a CNN policy and are out of scope.

## Physics timing

Each env advances by a fixed simulated timestep (`tau`) per `env.step()`, declared in its `EnvSpec.physics_tick_s`. Wall-clock delay between steps controls how fast the sim *appears* to run — it does not change the physics.

| Env | tau | 1x real-time delay |
|-----|-----|-------------------|
| CartPole-v1 | 0.02s | 20ms |
| Acrobot-v1 | 0.2s | 200ms |

`--sim-speed` is a multiplier over real-time. The action frequency is always one action per `env.step()` — Gymnasium requires it.

### Episode termination

Episodes end when the env terminates (env-specific failure / success condition) or truncates (per-env step cap, defaulting to `EnvSpec.default_max_steps`). For CartPole specifically: terminates on pole >±12° or cart >±2.4 from centre; truncates at 500 steps.

## Architecture

Two concurrent async tasks share a `gym.Env` via `asyncio.Lock`:

- **Simulation task** (`EpisodeRunner.run_episode`) — steps the env via `GymEnvApp`, handles episode reset.
- **Viewing task** (`EnvViewer.render_live`) — snapshots frames and renders via matplotlib.

`GymEnvApp` is the generic env wrapper parameterised by `EnvSpec` (replaces the old `CartPoleApp`). `EnvViewer` is the env-agnostic visualiser.

See `feature_plans/005a_async_architecture_20260403/` for sequence diagrams of the async architecture and `feature_plans/008_generic_env_support_20260426/` for the environment-generalisation design. The generated [API reference](https://ashwin133.github.io/demo-rl-1/) documents the current extension points.

## Development

```bash
# Format
uv run ruff format .

# Lint
uv run ruff check .

# Type check
uv run pyrefly check

# Test
uv run pytest -v
```

Per the project guidelines (`.ai_agents/coding_guidelines.md`), every feature change must pass `ruff format`, `ruff check`, `pyrefly check`, and `pytest` before merging.
