"""Tests for ReplayBuffer used by DQN."""

import numpy as np
import pytest

from app.training.replay_buffer import ReplayBuffer, Transition


class TestTransition:
    """Tests for Transition namedtuple."""

    def test_fields(self) -> None:
        t: Transition = Transition(
            state=np.array([1.0, 2.0]),
            action=0,
            reward=1.0,
            next_state=np.array([3.0, 4.0]),
            done=False,
        )
        assert t.action == 0
        assert t.reward == 1.0
        assert t.done is False
        np.testing.assert_array_equal(t.state, [1.0, 2.0])
        np.testing.assert_array_equal(t.next_state, [3.0, 4.0])


class TestReplayBuffer:
    """Tests for ReplayBuffer."""

    def _make_transition(self, action: int = 0) -> Transition:
        return Transition(
            state=np.array([1.0, 2.0, 3.0, 4.0]),
            action=action,
            reward=1.0,
            next_state=np.array([5.0, 6.0, 7.0, 8.0]),
            done=False,
        )

    def test_len_starts_at_zero(self) -> None:
        buf: ReplayBuffer = ReplayBuffer(capacity=100)
        assert len(buf) == 0

    def test_push_increases_length(self) -> None:
        buf: ReplayBuffer = ReplayBuffer(capacity=100)
        buf.push(self._make_transition())
        assert len(buf) == 1
        buf.push(self._make_transition())
        assert len(buf) == 2

    def test_capacity_overflow_evicts_oldest(self) -> None:
        buf: ReplayBuffer = ReplayBuffer(capacity=3)
        for i in range(5):
            buf.push(self._make_transition(action=i))
        assert len(buf) == 3
        # Oldest (action=0, action=1) should have been evicted
        sample: list[Transition] = buf.sample(3)
        actions: set[int] = {t.action for t in sample}
        assert actions == {2, 3, 4}

    def test_sample_returns_correct_size(self) -> None:
        buf: ReplayBuffer = ReplayBuffer(capacity=100)
        for i in range(10):
            buf.push(self._make_transition(action=i))
        sample: list[Transition] = buf.sample(5)
        assert len(sample) == 5

    def test_sample_raises_if_not_enough(self) -> None:
        buf: ReplayBuffer = ReplayBuffer(capacity=100)
        buf.push(self._make_transition())
        with pytest.raises(ValueError):
            buf.sample(5)

    def test_sample_returns_transitions(self) -> None:
        buf: ReplayBuffer = ReplayBuffer(capacity=100)
        for i in range(10):
            buf.push(self._make_transition(action=i))
        sample: list[Transition] = buf.sample(3)
        for t in sample:
            assert isinstance(t, Transition)
            assert isinstance(t.action, int)
            assert isinstance(t.reward, float)

    def test_sample_is_random(self) -> None:
        """Sampling twice should (almost certainly) give different orders."""
        buf: ReplayBuffer = ReplayBuffer(capacity=100)
        for i in range(50):
            buf.push(self._make_transition(action=i))
        sample1: list[int] = [t.action for t in buf.sample(20)]
        sample2: list[int] = [t.action for t in buf.sample(20)]
        # Extremely unlikely to be identical
        assert sample1 != sample2
