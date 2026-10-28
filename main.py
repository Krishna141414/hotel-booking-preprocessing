"""Run the whole preprocessing pipeline, raw CSV -> model-ready data: ``python main.py``.

Steps: download (if needed) -> understand -> clean -> engineer features ->
EDA figures -> select features -> encode + scale (fit on train only) -> save.
"""

import argparse
import logging
import time

import pandas as pd

from src import config
from src.cleaning import before_after_summary, clean_data, save_clean_data
from src.download_data import ensure_raw_data
from src.features import engineer_features
from src.load_data import load_raw_data, log_data_overview
from src.pipeline import (
    FeatureGroups,
    fit_preprocessing,
    save_artifacts,
    save_model_ready,
    select_features,
)
from src.visualize import generate_eda_figures, plot_correlation_heatmap, save_figure

logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    """Parse command-line options.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--skip-figures", action="store_true", help="do not regenerate reports/figures/"
    )
    return parser.parse_args()


def step(number: int, title: str) -> None:
    """Log a clearly visible step header.

    Args:
        number: Step number.
        title: Step title.
    """
    logger.info("=" * 72)
    logger.info("STEP %d | %s", number, title)
    logger.info("=" * 72)


def log_summary(rows: dict[str, tuple[str, str]], seconds: float) -> None:
    """Log the final raw -> model-ready summary table.

    Args:
        rows: Metric -> (before, after) display values.
        seconds: Total run time.
    """
    table = pd.DataFrame.from_dict(rows, orient="index", columns=["raw", "model-ready"])
    logger.info("SUMMARY (%.1f s)\n%s", seconds, table.to_string())


def main() -> None:
    """Execute every pipeline step in order and print a summary."""
    args = parse_args()
    config.setup_logging()
    started = time.perf_counter()

    step(0, "Get raw data")
    raw_df = load_raw_data(ensure_raw_data())

    step(1, "Data understanding")
    log_data_overview(raw_df)

    step(2, "Cleaning")
    clean_df, step_report, adr_cap = clean_data(raw_df)
    logger.info("Cleaning steps:\n%s", step_report.to_string())
    logger.info("Before vs after:\n%s", before_after_summary(raw_df, clean_df).to_string())
    save_clean_data(clean_df)

    step(3, "Feature engineering")
    featured_df = engineer_features(clean_df)

    step(4, "EDA figures")
    if args.skip_figures:
        logger.info("Skipped (--skip-figures)")
    else:
        generate_eda_figures(raw_df, clean_df, featured_df, step_report)

    step(5, "Feature selection (checks on training rows only)")
    candidates = FeatureGroups.from_config()
    selected, train_rows = select_features(featured_df, candidates)
    if not args.skip_figures:
        heatmap = plot_correlation_heatmap(train_rows, candidates.numeric + candidates.binary)
        save_figure(heatmap, "11_correlation_heatmap")

    step(6, "Encoding + scaling (fit on training rows only)")
    result = fit_preprocessing(featured_df, selected, adr_cap, dropped=config.DROPPED_FEATURES)
    save_artifacts(result)
    save_model_ready(result)

    n_train, n_test = result.metadata["rows"]["train"], result.metadata["rows"]["test"]
    log_summary(
        {
            "rows": (
                f"{len(raw_df):,}",
                f"{len(clean_df):,} ({n_train:,} train / {n_test:,} test)",
            ),
            "missing cells": (
                f"{int(raw_df.isna().sum().sum()):,}",
                f"{int(clean_df.isna().sum().sum()):,}",
            ),
            "duplicate rows": (
                f"{int(raw_df.duplicated().sum()):,}",
                f"{int(clean_df.duplicated().sum())} (distinct before recoding, decisions 2.1)",
            ),
            "ADR max": (f"{raw_df['adr'].max():,.0f}", f"{adr_cap:,.0f} (capped)"),
            "cancellation rate": (
                f"{raw_df[config.TARGET].mean():.1%}",
                f"{clean_df[config.TARGET].mean():.1%}",
            ),
            "columns / features": (
                f"{raw_df.shape[1]} columns",
                f"{len(selected.all)} inputs -> {len(result.metadata['output_features'])} features",
            ),
        },
        time.perf_counter() - started,
    )


if __name__ == "__main__":
    main()
