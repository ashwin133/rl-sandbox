"""EnvSpec — per-environment metadata — and the SUPPORTED_ENVS registry.

The registry is the single source of truth for which gym environments this
codebase supports. CLI --env choices come from SUPPORTED_ENVS.keys(); the
spec entries supply env-specific tuning (physics tick, solved threshold,
max steps).
"""

from pydantic import BaseModel


class EnvSpec(BaseModel):
    """Per-environment metadata.

    Attributes:
        env_id: gymnasium env id (e.g., "CartPole-v1").
        physics_tick_s: env's per-step simulated time (tau). Used by the
            sim viewer / replay to scale wall-clock playback.
        solved_reward_threshold: rolling-average reward at which the env
            is considered "solved". Sign convention varies by env (CartPole
            475+, Acrobot −100+).
        rolling_window: number of trailing episodes used for the solved check.
        default_max_steps: natural episode cap for both training and run-sim.
    """

    env_id: str
    physics_tick_s: float
    solved_reward_threshold: float
    rolling_window: int = 100
    default_max_steps: int = 500


SUPPORTED_ENVS: dict[str, EnvSpec] = {
    "CartPole-v1": EnvSpec(
        env_id="CartPole-v1",
        physics_tick_s=0.02,
        solved_reward_threshold=475.0,
        default_max_steps=500,
    ),
    "Acrobot-v1": EnvSpec(
        env_id="Acrobot-v1",
        physics_tick_s=0.2,
        solved_reward_threshold=-100.0,
        default_max_steps=500,
    ),
}
