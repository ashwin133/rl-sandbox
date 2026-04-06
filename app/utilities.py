"""Utility helpers."""

from datetime import datetime
from pathlib import Path


class Utilities:
    @staticmethod
    def create_batch_directory(base_path: str) -> Path:
        """Create a new directory for storing batch results, named with a timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = Path(base_path) / f"batch_{timestamp}"
        batch_dir.mkdir(parents=True, exist_ok=False)
        return batch_dir
