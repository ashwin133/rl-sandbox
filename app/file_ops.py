"""File operations for saving and loading batch recordings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.models.recording import BatchRecording
from app.models.replay import BatchReplay, EpisodeReplay

FRAMES_FILE_NAME = "frames.npz"
METADATA_FILE_NAME = "metadata.json"
REPLAY_EPISODES_KEY = "replay_episodes"
FRAMES_KEY_PREFIX = "frames_"


class FileOps:
    """Handles persistence of batch recording data (frames + metadata)."""

    @staticmethod
    def save(
        captured_frames: dict[int, list[np.ndarray]],
        metadata: BatchRecording,
        path: Path,
    ) -> None:
        """Save captured episode frames and metadata to disk.

        Writes frames.npz (numpy arrays) and metadata.json to the given directory.
        """
        path.mkdir(parents=True, exist_ok=True)

        sorted_keys: list[int] = sorted(captured_frames.keys())
        arrays: dict[str, np.ndarray] = {
            REPLAY_EPISODES_KEY: np.array(sorted_keys),
        }
        for pos, ep_idx in enumerate(sorted_keys):
            arrays[f"{FRAMES_KEY_PREFIX}{pos}"] = np.array(captured_frames[ep_idx])

        np.savez(path / FRAMES_FILE_NAME, **arrays)

        with open(path / METADATA_FILE_NAME, "w") as f:
            json.dump(metadata.model_dump(), f, indent=2, default=str)

    @staticmethod
    def load_batch_replay(path: Path) -> BatchReplay:
        """Load frames + metadata from disk and return a BatchReplay."""

        with open(path / METADATA_FILE_NAME) as f:
            recording: BatchRecording = BatchRecording(**json.load(f))

        data: np.lib.npyio.NpzFile = np.load(path / FRAMES_FILE_NAME)
        replay_episodes: np.ndarray = data[REPLAY_EPISODES_KEY]

        replay_lookup: dict[int, int] = {
            int(ep_idx): pos for pos, ep_idx in enumerate(replay_episodes)
        }

        episodes: list[EpisodeReplay] = []
        for ep_rec in recording.episodes:
            if ep_rec.episode_index in replay_lookup:
                pos: int = replay_lookup[ep_rec.episode_index]
                frame_array: np.ndarray = data[f"{FRAMES_KEY_PREFIX}{pos}"]
                frames: list[np.ndarray] = [
                    frame_array[i] for i in range(frame_array.shape[0])
                ]
            else:
                frames = []
            episodes.append(EpisodeReplay(frames=frames, metadata=ep_rec))

        return BatchReplay(
            episodes=episodes,
            source_path=str(path),
            total_source_episodes=recording.no_episodes,
            replay_freq=recording.replay_freq,
        )
