"""Experience replay buffer for DQN training."""

import random
from collections import deque
from typing import NamedTuple

import numpy as np


class Transition(NamedTuple):
    """A single environment transition: (s, a, r, s', done)."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-capacity circular buffer of transitions.

    Stores (state, action, reward, next_state, done) tuples and supports
    random sampling for mini-batch training. Uses collections.deque for
    O(1) push with automatic eviction of the oldest entry at capacity.
    """

    def __init__(self, capacity: int) -> None:
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        """Add a transition to the buffer, evicting the oldest if at capacity."""
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        """Sample a random mini-batch of transitions.

        Raises ValueError if batch_size exceeds the current buffer size.
        """
        if batch_size > len(self._buffer):
            raise ValueError(
                f"Cannot sample {batch_size} from buffer of size {len(self._buffer)}"
            )
        return random.sample(list(self._buffer), batch_size)

    def __len__(self) -> int:
        return len(self._buffer)
