import argparse
import asyncio
import logging
from pathlib import Path

import gymnasium as gym

from app import logging_config as logging_utils
from app.cartpole import MODEL_TICK_BETWEEN_ACTIONS_S, CartPoleApp
from app.episode import EpisodeRunner
from app.file_ops import FileOps
from app.models.batch import BatchConfig
from app.models.episode import EpisodeConfig
from app.models.recording import BatchRecording, EpisodeRecording
from app.models.training import TrainingConfig
from app.policy.base import BaseActionPolicy
from app.policy.dqn import DqnPolicy
from app.policy.random import RandomActionPolicy
from app.policy.reinforce import ReinforcePolicy
from app.render import RENDER_DELAY_MS, MatplotlibAnimator
from app.replay import ReplayRunner
from app.training.dqn import DqnTrainer
from app.training.reinforce import ReinforceTrainer
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
    sim_parser.add_argument(
        "--policy",
        type=str,
        choices=["random", "reinforce", "dqn"],
        default="random",
        help="Policy to use (default: random)",
    )
    sim_parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a trained model .pt file (required for reinforce/dqn policies)",
    )

    # --- train mode ---
    train_parser = subparsers.add_parser("train", help="Train an RL agent")
    train_parser.add_argument(
        "--algorithm",
        type=str,
        choices=["reinforce", "dqn"],
        required=True,
        help="Training algorithm to use",
    )
    train_parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes (default: 1000)",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001)",
    )
    train_parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Hidden layer size (default: 128)",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="DQN mini-batch size (default: 64)",
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Directory to save trained model (default: ./models)",
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

        CARTPOLE_OBS_SIZE = 4
        CARTPOLE_ACTION_SIZE = 2
        policy: BaseActionPolicy
        if args.policy == "random":
            policy = RandomActionPolicy()
        elif args.policy == "reinforce":
            if args.model_path is None:
                parser.error("--model-path is required when using --policy reinforce")
            policy = ReinforcePolicy.load(
                Path(args.model_path),
                obs_size=CARTPOLE_OBS_SIZE,
                action_size=CARTPOLE_ACTION_SIZE,
            )
        else:
            if args.model_path is None:
                parser.error("--model-path is required when using --policy dqn")
            policy = DqnPolicy.load(
                Path(args.model_path),
                obs_size=CARTPOLE_OBS_SIZE,
                action_size=CARTPOLE_ACTION_SIZE,
            )
        logger.info(f"Using policy: {type(policy).__name__}")
        batch_config: BatchConfig = BatchConfig(
            no_episodes=10,
            episode_config=EpisodeConfig(max_steps=100, sim_speed_multiplier=sim_speed),
            replay_freq=args.replay_freq,
        )
        batch_dir = Utilities.create_batch_directory(base_path="./results")

        async def run_simulation() -> None:
            """Run specified episodes using EpisodeRunner."""
            batch_result, captured_frames = await runner.run_batch(policy, batch_config)
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

    elif args.mode == "train":
        env: gym.Env = gym.make("CartPole-v1")
        train_config: TrainingConfig = TrainingConfig(
            algorithm=args.algorithm,
            num_episodes=args.episodes,
            learning_rate=args.lr,
            hidden_size=args.hidden_size,
            batch_size=args.batch_size,
        )

        if args.algorithm == "reinforce":
            trainer = ReinforceTrainer(env=env, config=train_config)
        else:
            trainer = DqnTrainer(env=env, config=train_config)

        logger.info(f"Training {args.algorithm} for {args.episodes} episodes...")
        result = trainer.train()

        output_dir: Path = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path: Path = output_dir / f"{args.algorithm}_best.pt"
        trainer.save_model(model_path)

        logger.info(
            f"Training complete. Final avg reward: {result.final_avg_reward:.1f}, "
            f"Solved: {result.solved}. Model saved to {model_path}"
        )
        env.close()

    elif args.mode == "playback-from-logs":
        batch_replay = FileOps.load_batch_replay(Path(args.path))
        renderer = MatplotlibAnimator()
        replay_runner = ReplayRunner(renderer=renderer)
        asyncio.run(replay_runner.replay_batch(batch_replay))
