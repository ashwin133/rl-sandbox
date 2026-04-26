"""run-sim CLI command — run a live simulation with a chosen policy."""

import argparse
import asyncio
import logging
from pathlib import Path

from app.cartpole import MODEL_TICK_BETWEEN_ACTIONS_S, CartPoleApp
from app.episode import EpisodeRunner
from app.file_ops import FileOps
from app.models.batch import BatchConfig
from app.models.episode import EpisodeConfig
from app.models.recording import BatchRecording, EpisodeRecording
from app.policy.base import BaseActionPolicy
from app.policy.random import RandomActionPolicy
from app.policy.reinforce import ReinforcePolicy
from app.render import RENDER_DELAY_MS
from app.utilities import Utilities
from app.viewer import CartPoleViewer

logger = logging.getLogger(__name__)

MAX_SIM_SPEED_ALLOWED = MODEL_TICK_BETWEEN_ACTIONS_S / (RENDER_DELAY_MS / 1000)
CARTPOLE_OBS_SIZE = 4
CARTPOLE_ACTION_SIZE = 2
EPISODES_PER_BATCH = 10
EPISODE_MAX_STEPS = 100
RESULTS_BASE_PATH = "./results"


def register_args(parser: argparse.ArgumentParser) -> None:
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


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    sim_speed: float = args.sim_speed
    if sim_speed > MAX_SIM_SPEED_ALLOWED:
        logger.warning(
            f"Requested sim speed {sim_speed}x is faster than the max allowed "
            f"{MAX_SIM_SPEED_ALLOWED:.2f}x for smooth rendering. Capping to max allowed."
        )
        sim_speed = MAX_SIM_SPEED_ALLOWED

    app: CartPoleApp = CartPoleApp()
    viewer: CartPoleViewer = CartPoleViewer(env=app.env, lock=app.lock)
    runner: EpisodeRunner = EpisodeRunner(app=app)

    policy: BaseActionPolicy = _build_policy(args, parser)
    logger.info(f"Using policy: {type(policy).__name__}")

    batch_config: BatchConfig = BatchConfig(
        no_episodes=EPISODES_PER_BATCH,
        episode_config=EpisodeConfig(
            max_steps=EPISODE_MAX_STEPS, sim_speed_multiplier=sim_speed
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
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> BaseActionPolicy:
    if args.policy == "random":
        return RandomActionPolicy()
    if args.model_path is None:
        parser.error("--model-path is required when using --policy reinforce")
    return ReinforcePolicy.load(
        Path(args.model_path),
        obs_size=CARTPOLE_OBS_SIZE,
        action_size=CARTPOLE_ACTION_SIZE,
    )
