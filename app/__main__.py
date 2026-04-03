import asyncio
import argparse
import logging

from app import logging_config as logging_utils
from app.cartpole import CartPoleApp
from app.policy import RandomActionPolicy
from app.viewer import CartPoleViewer

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging_utils.LoggerFactory.setup()
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render the environment")
    args = parser.parse_args()

    if args.render:
        app = CartPoleApp()
        viewer = CartPoleViewer(env=app.env, lock=app.lock)

        async def main() -> None:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(viewer.render_live())
                tg.create_task(app.loop_policy_over_episodes(RandomActionPolicy()))

        asyncio.run(main())
    else:
        raise NotImplementedError("Only rendering mode is currently implemented.")
