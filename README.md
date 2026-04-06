# demo-rl-1

A CartPole reinforcement learning demo built on [Gymnasium](https://gymnasium.farama.org/).

## Setup

```bash
uv sync
```

## Run

```bash
# Live rendering with random policy
uv run python -m app --render

# Or via poe task
uv run poe cart-random-action-render
```

## Physics Timing

CartPole-v1 uses a fixed physics timestep of `tau = 0.02s` per `env.step()` call. The wall-clock delay between steps controls how fast the simulation appears to run — it does not change the physics.

| Step delay | Steps/sec | Simulation speed |
|------------|-----------|------------------|
| 1ms | ~1000 | 20x real-time |
| 20ms | ~50 | 1x real-time |
| 100ms | ~10 | 0.2x (slow motion) |

The action frequency is always 1 action per step. Gymnasium requires exactly one action per `env.step()` call — you cannot skip or batch actions.

### Episode Termination

An episode ends when:
- **Terminated**: pole angle exceeds +/-12 degrees, or cart moves beyond +/-2.4 units from centre
- **Truncated**: episode reaches 500 steps without failing

## Architecture

The app uses two concurrent async tasks sharing a `gym.Env` via `asyncio.Lock`:

- **Simulation task** (`EpisodeRunner.run_episode`) — steps the env via `CartPoleApp`, handles episode reset
- **Viewing task** (`CartPoleViewer.render_live`) — snapshots frames and renders via matplotlib

See `feature_plans/005a_async_architecture_20260403/` for detailed sequence diagrams.

## Development

```bash
# Format
uv run ruff format .

# Lint
uv run ruff check .

# Type check
uv run pyrefly check app/ tests/

# Test
uv run pytest -v
```
