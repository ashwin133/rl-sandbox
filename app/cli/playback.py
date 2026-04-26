"""playback-from-logs CLI command — replay saved batch results."""

import argparse
import asyncio
from pathlib import Path

from app.file_ops import FileOps
from app.render import MatplotlibAnimator
from app.replay import ReplayRunner


def register_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to a batch results directory containing metadata.json and frames.npz",
    )


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    batch_replay = FileOps.load_batch_replay(Path(args.path))
    renderer: MatplotlibAnimator = MatplotlibAnimator()
    replay_runner: ReplayRunner = ReplayRunner(renderer=renderer)
    asyncio.run(replay_runner.replay_batch(batch_replay))
