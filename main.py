"""Run the preprocessing pipeline end to end: ``python main.py``."""

import logging

from src import config
from src.download_data import ensure_raw_data
from src.load_data import load_raw_data, log_data_overview

logger = logging.getLogger("main")


def main() -> None:
    """Execute every pipeline stage in order."""
    config.setup_logging()

    logger.info("=== Step 0: Get raw data ===")
    raw_path = ensure_raw_data()

    logger.info("=== Step 1: Data understanding ===")
    raw_df = load_raw_data(raw_path)
    log_data_overview(raw_df)

    logger.info("Pipeline finished")


if __name__ == "__main__":
    main()
