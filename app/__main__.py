import argparse
import asyncio
import logging
from pathlib import Path

from app import logging_config as logging_utils
from app.cartpole import MODEL_TICK_BETWEEN_ACTIONS_S, CartPoleApp
from app.episode import EpisodeRunner
from app.file_ops import FileOps
from app.models.batch import BatchConfig
from app.models.episode import EpisodeConfig
from app.models.recording import BatchRecording, EpisodeRecording
from app.policy.random import RandomActionPolicy
from app.render import RENDER_DELAY_MS, MatplotlibAnimator
from app.replay import ReplayRunner
from app.utilities import Utilities
from app.viewer import CartPoleViewer

logger = logging.getLogger(__name__)

MAX_SIM_SPEED_ALLOWED = MODEL_TICK_BETWEEN_ACTIONS_S / (RENDER_DELAY_MS / 1000)

if __name__ == "__main__":
    logging_utils.LoggerFactory.setup()

    parser = argparse.ArgumentParser(description="CartPole RL demo")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- run-sim mode ---
    sim_parser = subparsers.add_parser("run-sim", help="Run a live simulation")
    sim_parser.add_argument(
        "--sim-speed",
        type=float,
        default=1.0,
        help="Speed multiplier (e.g., 2.0 = 2x speed, 0.5 = half speed)",
    )
    sim_parser.add_argument(
        "--replay-freq",
        type=int,
        default=1,
        help="Capture frames every Nth episode for later replay (default: 1 = all)",
    )

    # --- playback-from-logs mode ---
    playback_parser = subparsers.add_parser(
        "playback-from-logs", help="Replay episodes from saved batch results"
    )
    playback_parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to a batch results directory containing metadata.json and frames.npz",
    )

    args = parser.parse_args()

    if args.mode == "run-sim":
        sim_speed: float = args.sim_speed
        if sim_speed > MAX_SIM_SPEED_ALLOWED:
            logger.warning(
                f"Requested sim speed {sim_speed}x is faster than the max allowed "
                f"{MAX_SIM_SPEED_ALLOWED:.2f}x for smooth rendering. Capping to max allowed."
            )
            sim_speed = MAX_SIM_SPEED_ALLOWED
        app = CartPoleApp()
        viewer = CartPoleViewer(env=app.env, lock=app.lock)
        runner = EpisodeRunner(app=app)
        policy = RandomActionPolicy()
        config = BatchConfig(
            no_episodes=10,
            episode_config=EpisodeConfig(max_steps=100, sim_speed_multiplier=sim_speed),
            replay_freq=args.replay_freq,
        )
        batch_dir = Utilities.create_batch_directory(base_path="./results")

        async def run_simulation() -> None:
            """Run specified episodes using EpisodeRunner."""
            batch_result, captured_frames = await runner.run_batch(policy, config)
            logger.info(
                "Simulation task finished. Ending simulation and saving results."
            )
            recordings = [
                EpisodeRecording(
                    episode_index=i,
                    policy_name=ep.policy_name,
                    timestamp=ep.timestamp,
                    timesteps_reached=ep.timesteps_reached,
                    sim_speed_multiplier=ep.episode_config.sim_speed_multiplier,
                )
                for i, ep in enumerate(batch_result.episodes)
            ]
            metadata = BatchRecording(
                no_episodes=config.no_episodes,
                replay_freq=config.replay_freq,
                episodes=recordings,
            )
            FileOps.save(captured_frames, metadata, batch_dir)
            viewer.quit()

        async def main() -> None:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(viewer.render_live())
                tg.create_task(run_simulation())

        asyncio.run(main())

    elif args.mode == "playback-from-logs":
        batch_replay = FileOps.load_batch_replay(Path(args.path))
        renderer = MatplotlibAnimator()
        replay_runner = ReplayRunner(renderer=renderer)
        asyncio.run(replay_runner.replay_batch(batch_replay))
