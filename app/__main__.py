import argparse
import asyncio
import logging

from app import logging_config as logging_utils
from app.cartpole import MODEL_TICK_BETWEEN_ACTIONS_S, CartPoleApp
from app.episode import EpisodeRunner
from app.models.batch import BatchConfig, BatchResult
from app.models.episode import EpisodeConfig
from app.policy import RandomActionPolicy
from app.render import RENDER_DELAY_MS
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

    # --- playback-from-logs mode ---
    playback_parser = subparsers.add_parser(
        "playback-from-logs", help="Replay episodes from saved batch results"
    )
    playback_parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to a batch results directory containing metadata.json and step_data.npz",
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
        )
        batch_dir = Utilities.create_batch_directory(base_path="./results")

        async def run_simulation() -> None:
            """Run specified episodes using EpisodeRunner."""
            batch_result: BatchResult = await runner.run_batch(policy, config)
            logger.info(
                "Simulation task finished. Ending simulation and saving results."
            )
            runner.log_batch_to_file(batch_result=batch_result, path=batch_dir)
            viewer.quit()

        async def main() -> None:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(viewer.render_live())
                tg.create_task(run_simulation())

        asyncio.run(main())

    elif args.mode == "playback-from-logs":
        raise NotImplementedError("Playback from logs is not yet implemented.")
