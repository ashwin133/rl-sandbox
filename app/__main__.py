import argparse
import logging

from app import logging_config as logging_utils
from app.policy import RandomActionPolicy

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging_utils.LoggerFactory.setup()
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render the environment")
    args = parser.parse_args()

    if args.render:
        from app.viewer import CartPoleViewer

        viewer = CartPoleViewer()
        viewer.render_live(action_policy=RandomActionPolicy())
    else:
        from app.cartpole import CartPoleApp

        logger.info("Starting CartPoleApp without rendering.")
        CartPoleApp()
