import numpy as np

from app.cartpole import CartPoleApp
from app.policy import RandomActionPolicy


class TestIntegration:
    """Integration test for the full headless flow."""

    def test_full_headless_flow(self) -> None:
        app = CartPoleApp()
        policy = RandomActionPolicy()
        num_steps = 10
        frames = []

        for _ in range(num_steps):
            app.apply_action(policy)
            frame = app.env.render()
            frames.append(frame)

        assert len(frames) == num_steps
        for frame in frames:
            assert isinstance(frame, np.ndarray)
            assert frame.ndim == 3
            assert frame.shape[2] == 3  # RGB

        app.env.close()
