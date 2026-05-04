"""run-sim CLI command — run a live simulation of a chosen env with a chosen policy."""

import argparse
import asyncio
import logging
from pathlib import Path

from rl_sandbox.envs.gym_env import GymEnvApp
from rl_sandbox.envs.registry import SUPPORTED_ENVS, EnvSpec
from rl_sandbox.episode import EpisodeRunner
from rl_sandbox.file_ops import FileOps
from rl_sandbox.models.batch import BatchConfig
from rl_sandbox.models.episode import EpisodeConfig
from rl_sandbox.models.recording import BatchRecording, EpisodeRecording
from rl_sandbox.policy.base import BaseActionPolicy
from rl_sandbox.policy.random import RandomActionPolicy
from rl_sandbox.policy.reinforce import ReinforcePolicy
from rl_sandbox.render import RENDER_DELAY_MS
from rl_sandbox.utilities import Utilities
from rl_sandbox.viewer import EnvViewer

logger = logging.getLogger(__name__)

DEFAULT_EPISODES_PER_BATCH = 10
RESULTS_BASE_PATH = "./results"


def register_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--env",
        type=str,
        choices=list(SUPPORTED_ENVS),
        required=True,
        help="Gym environment id to simulate",
    )
    parser.add_argument(
        "--sim-speed",
        type=float,
        default=1.0,
        help="Speed multiplier (e.g., 2.0 = 2x speed, 0.5 = half speed)",
    )
    parser.add_argument(
        "--replay-freq",
        type=int,
        default=1,
        help="Capture frames every Nth episode for later replay (default: 1 = all)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["random", "reinforce"],
        default="random",
        help="Policy to use (default: random)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a trained model .pt file (required for reinforce policy)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES_PER_BATCH,
        help=f"Number of episodes to run (default: {DEFAULT_EPISODES_PER_BATCH})",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Per-episode step cap (default: from EnvSpec.default_max_steps)",
    )


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    spec: EnvSpec = SUPPORTED_ENVS[args.env]
    max_sim_speed_allowed: float = spec.physics_tick_s / (RENDER_DELAY_MS / 1000)

    sim_speed: float = args.sim_speed
    if sim_speed > max_sim_speed_allowed:
        logger.warning(
            f"Requested sim speed {sim_speed}x is faster than the max allowed "
            f"{max_sim_speed_allowed:.2f}x for smooth rendering. Capping to max allowed."
        )
        sim_speed = max_sim_speed_allowed

    app: GymEnvApp = GymEnvApp(spec)
    viewer: EnvViewer = EnvViewer(env=app.env, lock=app.lock)
    runner: EpisodeRunner = EpisodeRunner(app=app)

    policy: BaseActionPolicy = _build_policy(args, spec, parser)
    logger.info(f"Using policy: {type(policy).__name__} on env {spec.env_id}")

    max_steps: int = (
        args.max_steps if args.max_steps is not None else spec.default_max_steps
    )
    batch_config: BatchConfig = BatchConfig(
        no_episodes=args.episodes,
        episode_config=EpisodeConfig(
            max_steps=max_steps, sim_speed_multiplier=sim_speed
        ),
        replay_freq=args.replay_freq,
    )
    batch_dir: Path = Utilities.create_batch_directory(base_path=RESULTS_BASE_PATH)

    async def run_simulation() -> None:
        batch_result, captured_frames = await runner.run_batch(policy, batch_config)
        logger.info("Simulation task finished. Ending simulation and saving results.")
        recordings: list[EpisodeRecording] = [
            EpisodeRecording(
                episode_index=i,
                policy_name=ep.policy_name,
                timestamp=ep.timestamp,
                timesteps_reached=ep.timesteps_reached,
                sim_speed_multiplier=ep.episode_config.sim_speed_multiplier,
            )
            for i, ep in enumerate(batch_result.episodes)
        ]
        metadata: BatchRecording = BatchRecording(
            no_episodes=batch_config.no_episodes,
            replay_freq=batch_config.replay_freq,
            env_spec=spec,
            episodes=recordings,
        )
        FileOps.save(captured_frames, metadata, batch_dir)
        viewer.quit()

    async def main() -> None:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(viewer.render_live())
            tg.create_task(run_simulation())

    asyncio.run(main())


def _build_policy(
    args: argparse.Namespace, spec: EnvSpec, parser: argparse.ArgumentParser
) -> BaseActionPolicy:
    if args.policy == "random":
        return RandomActionPolicy()
    if args.model_path is None:
        parser.error("--model-path is required when using --policy reinforce")
    return ReinforcePolicy.load(Path(args.model_path), expected_env_id=spec.env_id)
