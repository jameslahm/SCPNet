import logging
import sys

from config import cfg

cfg.log_file = f"{cfg.checkpoint}/{cfg.log_file}"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[
                        logging.FileHandler(cfg.log_file),
                        logging.StreamHandler(sys.stdout)
                    ])

logger = logging.getLogger()

if cfg.resume:
    logger.info(f"Resume training... from {cfg.resume}")

