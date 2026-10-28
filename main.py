"""Run the preprocessing pipeline end to end: ``python main.py``."""

import logging

from src import config
from src.cleaning import before_after_summary, clean_data, save_clean_data
from src.download_data import ensure_raw_data
from src.features import engineer_features
from src.load_data import load_raw_data, log_data_overview
from src.pipeline import FeatureGroups, fit_preprocessing, save_artifacts, save_model_ready
from src.visualize import generate_eda_figures

logger = logging.getLogger("main")


def main() -> None:
    """Execute every pipeline stage in order."""
    config.setup_logging()

    logger.info("=== Step 0: Get raw data ===")
    raw_path = ensure_raw_data()

    logger.info("=== Step 1: Data understanding ===")
    raw_df = load_raw_data(raw_path)
    log_data_overview(raw_df)

    logger.info("=== Step 2: Cleaning ===")
    clean_df, step_report, adr_cap = clean_data(raw_df)
    logger.info("Cleaning steps:\n%s", step_report.to_string())
    logger.info("Before vs after:\n%s", before_after_summary(raw_df, clean_df).to_string())
    save_clean_data(clean_df)

    logger.info("=== Step 3: Feature engineering ===")
    featured_df = engineer_features(clean_df)

    logger.info("=== Step 4: EDA figures ===")
    generate_eda_figures(raw_df, clean_df, featured_df, step_report)

    logger.info("=== Step 5: Encoding + scaling ===")
    result = fit_preprocessing(featured_df, FeatureGroups.from_config(), adr_cap)
    save_artifacts(result)
    save_model_ready(result)

    logger.info("Pipeline finished")


if __name__ == "__main__":
    main()
