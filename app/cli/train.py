"""train CLI command — train a REINFORCE agent on CartPole."""

import argparse
import logging
from pathlib import Path

import gymnasium as gym

from app.models.training import TrainingConfig
from app.training.reinforce import ReinforceTrainer

logger = logging.getLogger(__name__)

ENV_ID = "CartPole-v1"
DEFAULT_OUTPUT_DIR = "./models"
MODEL_FILENAME = "reinforce_best.pt"
DEFAULT_NUM_EPISODES = 1000
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_HIDDEN_SIZE = 128


def register_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help=f"Number of training episodes (default: {DEFAULT_NUM_EPISODES})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=DEFAULT_HIDDEN_SIZE,
        help=f"Hidden layer size (default: {DEFAULT_HIDDEN_SIZE})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save trained model (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--init-model",
        type=str,
        default=None,
        help="Path to a .pt file to warm-start training from (default: train from scratch). "
        "The model architecture (hidden-size) must match.",
    )


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    env: gym.Env = gym.make(ENV_ID)
    train_config: TrainingConfig = TrainingConfig(
        num_episodes=args.episodes,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
    )

    trainer: ReinforceTrainer = ReinforceTrainer(env=env, config=train_config)

    if args.init_model is not None:
        init_path: Path = Path(args.init_model)
        if not init_path.is_file():
            parser.error(f"--init-model path does not exist: {init_path}")
        trainer.load_model(init_path)
        logger.info(f"Warm-starting from {init_path}")

    logger.info(f"Training reinforce for {args.episodes} episodes...")
    result = trainer.train()

    output_dir: Path = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path: Path = output_dir / MODEL_FILENAME
    trainer.save_model(model_path)

    final_eval_summary: str = ""
    if result.evaluations:
        final_eval = result.evaluations[-1]
        final_eval_summary = (
            f", final eval={final_eval.mean_reward:.1f} ± {final_eval.std_reward:.1f}"
        )
    logger.info(
        f"Training complete. Final avg reward: {result.final_avg_reward:.1f}, "
        f"Solved: {result.solved}{final_eval_summary}. Model saved to {model_path}"
    )
    env.close()
