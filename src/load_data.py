"""Load the raw dataset and produce data-understanding summaries."""

import logging
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def load_raw_data(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV, treating the mirror's ``"NULL"`` strings as missing.

    Args:
        path: Location of the raw CSV.

    Returns:
        The raw bookings as a DataFrame (unmodified apart from NA parsing).

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If the target column is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}. Run `python -m src.download_data`.")

    df = pd.read_csv(path, na_values=config.MISSING_TOKENS, low_memory=False)
    if config.TARGET not in df.columns:
        raise ValueError(f"Target column '{config.TARGET}' not found in {path.name}.")

    logger.info("Loaded %s rows x %s columns from %s", f"{len(df):,}", df.shape[1], path.name)
    return df


def column_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise each column's dtype, missingness and cardinality.

    Args:
        df: Input DataFrame.

    Returns:
        One row per column with ``dtype``, ``non_null``, ``missing``,
        ``missing_pct``, ``n_unique`` and an ``example`` value.
    """
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "missing": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(),
            "example": df.apply(lambda col: col.dropna().iloc[0] if col.notna().any() else None),
        }
    )


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """List only the columns that contain missing values, worst first.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with ``missing`` count and ``missing_pct`` per affected column.
    """
    overview = column_overview(df)[["missing", "missing_pct"]]
    return overview[overview["missing"] > 0].sort_values("missing", ascending=False)


def data_quality_snapshot(df: pd.DataFrame) -> dict[str, float]:
    """Count the known data-quality issues without changing the data.

    Args:
        df: Raw bookings DataFrame.

    Returns:
        Mapping of issue name to count (or value for ``adr_max``).

    Raises:
        ValueError: If a required column is missing.
    """
    required = [*config.GUEST_COLUMNS, "adr", config.TARGET]
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    total_guests = df[config.GUEST_COLUMNS].sum(axis=1, min_count=1)
    return {
        "rows": len(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "rows_with_any_missing": int(df.isna().any(axis=1).sum()),
        "zero_guest_bookings": int((total_guests == 0).sum()),
        "negative_adr": int((df["adr"] < 0).sum()),
        "zero_adr": int((df["adr"] == 0).sum()),
        "adr_max": float(df["adr"].max()),
        "cancellation_rate_pct": round(float(df[config.TARGET].mean() * 100), 2),
    }


def log_data_overview(df: pd.DataFrame) -> None:
    """Log the headline data-understanding numbers in a readable block.

    Args:
        df: Raw bookings DataFrame.
    """
    memory_mb = df.memory_usage(deep=True).sum() / 1_000_000
    dtype_counts = df.dtypes.astype(str).value_counts().to_dict()
    logger.info(
        "Shape: %s rows x %s columns (%.1f MB in memory)", f"{len(df):,}", df.shape[1], memory_mb
    )
    logger.info("Column types: %s", dtype_counts)

    for row in missing_summary(df).itertuples():
        logger.info("Missing  %-12s %7s (%5.2f%%)", row.Index, f"{row.missing:,}", row.missing_pct)

    for issue, value in data_quality_snapshot(df).items():
        logger.info("Check    %-22s %s", issue, f"{value:,}")
