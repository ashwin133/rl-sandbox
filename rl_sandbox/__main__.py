"""CLI entry point for the reinforcement learning evaluation toolbox."""

import argparse

from rl_sandbox import logging_config as logging_utils
from rl_sandbox.cli import playback, run_sim, train

if __name__ == "__main__":
    logging_utils.LoggerFactory.setup()

    parser = argparse.ArgumentParser(
        description="Train, evaluate, and inspect reinforcement learning algorithms"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    sim_parser = subparsers.add_parser("run-sim", help="Run a live simulation")
    run_sim.register_args(sim_parser)

    train_parser = subparsers.add_parser("train", help="Train an RL agent")
    train.register_args(train_parser)

    playback_parser = subparsers.add_parser(
        "playback-from-logs", help="Replay episodes from saved batch results"
    )
    playback.register_args(playback_parser)

    args = parser.parse_args()

    if args.mode == "run-sim":
        run_sim.run(args, sim_parser)
    elif args.mode == "train":
        train.run(args, train_parser)
    elif args.mode == "playback-from-logs":
        playback.run(args, playback_parser)
