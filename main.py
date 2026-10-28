"""Run the preprocessing pipeline end to end: ``python main.py``."""

import logging

from src import config
from src.cleaning import before_after_summary, clean_data, save_clean_data
from src.download_data import ensure_raw_data
from src.features import engineer_features, log_selection_report, validate_selection
from src.load_data import load_raw_data, log_data_overview
from src.pipeline import (
    FeatureGroups,
    fit_preprocessing,
    save_artifacts,
    save_model_ready,
    split_frame,
)
from src.visualize import generate_eda_figures, plot_correlation_heatmap, save_figure

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

    logger.info("=== Step 5: Feature selection (checks on training rows) ===")
    candidates = FeatureGroups.from_config()
    train_rows, _ = split_frame(featured_df)
    candidate_numeric = candidates.numeric + candidates.binary
    log_selection_report(
        train_rows, candidate_numeric, candidates.categorical + config.LEAKAGE_COLUMNS
    )
    save_figure(plot_correlation_heatmap(train_rows, candidate_numeric), "11_correlation_heatmap")
    selected = candidates.without(config.DROPPED_FEATURES)
    for column, reason in config.DROPPED_FEATURES.items():
        logger.info("Dropped %-26s %s", column, reason)
    validate_selection(train_rows, selected.numeric + selected.binary, selected.categorical)
    logger.info("Final features: %d (%s)", len(selected.all), ", ".join(selected.all))

    logger.info("=== Step 6: Encoding + scaling (fit on training rows only) ===")
    result = fit_preprocessing(featured_df, selected, adr_cap, dropped=config.DROPPED_FEATURES)
    save_artifacts(result)
    save_model_ready(result)

    logger.info("Pipeline finished")


if __name__ == "__main__":
    main()
