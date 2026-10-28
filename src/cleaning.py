"""Cleaning steps: invalid rows, duplicates, missing values and outliers.

Every step is a small pure function (DataFrame in, new DataFrame out) so it can
be tested on its own. :func:`clean_data` runs them in a fixed order and returns
a per-step report. The reasoning behind each rule is in ``docs/decisions.md``.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from src import config
from src.load_data import data_quality_snapshot

logger = logging.getLogger(__name__)

CleaningStep = Callable[[pd.DataFrame], pd.DataFrame]

REQUIRED_COLUMNS: list[str] = [
    *config.GUEST_COLUMNS,
    "adr",
    "agent",
    "company",
    "country",
    "meal",
    *config.MAX_PLAUSIBLE_COUNTS,
]


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise if any required column is missing.

    Args:
        df: DataFrame to check.
        required: Column names that must be present.

    Raises:
        ValueError: If one or more columns are absent.
    """
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# ---------------------------------------------------------------------------
# Invalid rows
# ---------------------------------------------------------------------------
def drop_zero_guest_bookings(df: pd.DataFrame) -> pd.DataFrame:
    """Drop bookings with no adults, children or babies (not a real stay).

    Args:
        df: Bookings DataFrame.

    Returns:
        DataFrame without zero-guest rows.
    """
    total_guests = df[config.GUEST_COLUMNS].fillna(0).sum(axis=1)
    invalid = total_guests == 0
    logger.info("Zero-guest bookings dropped: %s", f"{invalid.sum():,}")
    return df[~invalid]


def drop_negative_adr(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with a negative average daily rate (a price cannot be negative).

    Zero ADR is kept on purpose: it matches complimentary and day-use stays.

    Args:
        df: Bookings DataFrame.

    Returns:
        DataFrame without negative-ADR rows.
    """
    invalid = df["adr"] < 0
    logger.info("Negative-ADR rows dropped: %s", f"{invalid.sum():,}")
    return df[~invalid]


def drop_implausible_counts(
    df: pd.DataFrame, max_counts: dict[str, int] = config.MAX_PLAUSIBLE_COUNTS
) -> pd.DataFrame:
    """Drop rows where a count column exceeds its plausible maximum (entry errors).

    Rows are dropped rather than capped because the true value is unknown
    (e.g. ``adults = 55`` could have meant 5 or 2).

    Args:
        df: Bookings DataFrame.
        max_counts: Column name -> highest plausible value.

    Returns:
        DataFrame without implausible rows.
    """
    invalid = pd.Series(False, index=df.index)
    for column, max_value in max_counts.items():
        too_high = df[column] > max_value
        logger.info("  %-28s > %d: %s rows", column, max_value, f"{too_high.sum():,}")
        invalid |= too_high
    logger.info("Implausible-count rows dropped: %s", f"{invalid.sum():,}")
    return df[~invalid]


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------
def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows, keeping the first occurrence.

    Must run before any column is dropped or recoded, otherwise rows that only
    differed in that column would wrongly become duplicates.

    Args:
        df: Bookings DataFrame.

    Returns:
        DataFrame with unique rows.
    """
    duplicated = df.duplicated(keep="first")
    logger.info("Exact duplicate rows dropped: %s", f"{duplicated.sum():,}")
    return df[~duplicated]


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------
def replace_company_with_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the ~94%-empty ``company`` ID with a binary ``has_company`` flag.

    Args:
        df: Bookings DataFrame.

    Returns:
        DataFrame with ``has_company`` added and ``company`` removed.
    """
    has_company = df["company"].notna().astype(int)
    logger.info("has_company = 1 for %s bookings; company column dropped", f"{has_company.sum():,}")
    return df.assign(has_company=has_company).drop(columns="company")


def fill_agent(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``has_agent`` flag and fill missing agent IDs with the 'no agent' code.

    Args:
        df: Bookings DataFrame.

    Returns:
        DataFrame with integer ``agent`` and a ``has_agent`` flag.
    """
    has_agent = df["agent"].notna().astype(int)
    logger.info("agent missing -> %d: %s rows", config.NO_AGENT_ID, f"{(has_agent == 0).sum():,}")
    return df.assign(
        agent=df["agent"].fillna(config.NO_AGENT_ID).astype(int),
        has_agent=has_agent,
    )


def fill_country(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing country codes with an explicit 'Unknown' category.

    Args:
        df: Bookings DataFrame.

    Returns:
        DataFrame with no missing ``country``.
    """
    logger.info(
        "country missing -> '%s': %s rows",
        config.UNKNOWN_COUNTRY,
        f"{df['country'].isna().sum():,}",
    )
    return df.assign(country=df["country"].fillna(config.UNKNOWN_COUNTRY))


def fill_children(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing ``children`` with 0 and restore its integer type.

    Args:
        df: Bookings DataFrame.

    Returns:
        DataFrame with integer ``children``.
    """
    logger.info("children missing -> 0: %s rows", f"{df['children'].isna().sum():,}")
    return df.assign(children=df["children"].fillna(0).astype(int))


# ---------------------------------------------------------------------------
# Category consistency
# ---------------------------------------------------------------------------
def standardize_meal(df: pd.DataFrame) -> pd.DataFrame:
    """Merge meal labels that mean the same thing ("Undefined" -> "SC").

    Args:
        df: Bookings DataFrame.

    Returns:
        DataFrame with consistent ``meal`` labels.
    """
    affected = df["meal"].isin(config.MEAL_REPLACEMENTS).sum()
    logger.info("meal relabelled %s: %s rows", config.MEAL_REPLACEMENTS, f"{affected:,}")
    return df.assign(meal=df["meal"].replace(config.MEAL_REPLACEMENTS))


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------
def cap_adr(df: pd.DataFrame, quantile: float = config.ADR_CAP_QUANTILE) -> pd.DataFrame:
    """Cap (winsorise) ADR values above the given quantile.

    Args:
        df: Bookings DataFrame.
        quantile: Upper quantile used as the cap, between 0 and 1.

    Returns:
        DataFrame with capped ``adr``.

    Raises:
        ValueError: If ``quantile`` is outside (0, 1].
    """
    if not 0 < quantile <= 1:
        raise ValueError(f"quantile must be in (0, 1], got {quantile}")
    cap = float(df["adr"].quantile(quantile))
    above = df["adr"] > cap
    logger.info(
        "ADR capped at %.2f (%.1fth percentile): %s rows, old max %.2f",
        cap,
        quantile * 100,
        f"{above.sum():,}",
        df["adr"].max(),
    )
    return df.assign(adr=df["adr"].clip(upper=cap))


# ---------------------------------------------------------------------------
# Checks and orchestration
# ---------------------------------------------------------------------------
def flag_leakage_columns(df: pd.DataFrame) -> list[str]:
    """Log a warning for target-leaking columns that are still present.

    They are kept for EDA and removed in feature selection (Stage 6).

    Args:
        df: Bookings DataFrame.

    Returns:
        Names of the leakage columns found in ``df``.
    """
    present = [col for col in config.LEAKAGE_COLUMNS if col in df.columns]
    if present:
        logger.warning("Leakage columns kept for EDA, dropped before modelling: %s", present)
    return present


def check_no_missing(df: pd.DataFrame) -> None:
    """Raise if any missing values remain after cleaning.

    Args:
        df: Cleaned DataFrame.

    Raises:
        ValueError: If any column still contains missing values.
    """
    remaining = df.isna().sum()
    remaining = remaining[remaining > 0]
    if not remaining.empty:
        raise ValueError(f"Missing values remain after cleaning: {remaining.to_dict()}")


def _count_changed_values(before: pd.DataFrame, after: pd.DataFrame) -> int:
    """Count cells that changed value between two frames on shared rows/columns.

    Args:
        before: Frame before a step.
        after: Frame after the step.

    Returns:
        Number of changed cells (NaN -> value counts as a change).
    """
    columns = before.columns.intersection(after.columns)
    old = before.loc[after.index, columns]
    new = after[columns]
    unchanged = old.eq(new) | (old.isna() & new.isna())
    return int((~unchanged).to_numpy().sum())


CLEANING_STEPS: list[tuple[str, CleaningStep]] = [
    ("Drop zero-guest bookings", drop_zero_guest_bookings),
    ("Drop negative ADR", drop_negative_adr),
    ("Drop implausible counts", drop_implausible_counts),
    ("Drop exact duplicates", drop_duplicate_rows),
    ("company -> has_company flag", replace_company_with_flag),
    ("Fill agent + has_agent flag", fill_agent),
    ("Fill country 'Unknown'", fill_country),
    ("Fill children 0", fill_children),
    ("meal Undefined -> SC", standardize_meal),
    ("Cap extreme ADR", cap_adr),
]


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every cleaning step in order and record what each one changed.

    Args:
        df: Raw bookings DataFrame (as returned by ``load_raw_data``).

    Returns:
        A tuple ``(cleaned_df, step_report)``. The report has one row per step
        with rows before/after, rows removed, values changed and column changes.

    Raises:
        ValueError: If required columns are missing or nulls remain at the end.
    """
    validate_columns(df, REQUIRED_COLUMNS)
    report_rows = []
    current = df
    for name, step in CLEANING_STEPS:
        logger.info("-> %s", name)
        result = step(current)
        added = sorted(set(result.columns) - set(current.columns))
        removed = sorted(set(current.columns) - set(result.columns))
        report_rows.append(
            {
                "step": name,
                "rows_before": len(current),
                "rows_after": len(result),
                "rows_removed": len(current) - len(result),
                "values_changed": _count_changed_values(current, result),
                "columns": ", ".join([f"+{c}" for c in added] + [f"-{c}" for c in removed]),
            }
        )
        current = result

    check_no_missing(current)
    flag_leakage_columns(current)
    cleaned = current.reset_index(drop=True)
    logger.info(
        "Cleaning done: %s -> %s rows (%s removed, %.1f%%)",
        f"{len(df):,}",
        f"{len(cleaned):,}",
        f"{len(df) - len(cleaned):,}",
        (len(df) - len(cleaned)) / len(df) * 100,
    )
    return cleaned, pd.DataFrame(report_rows).set_index("step")


def before_after_summary(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Build a side-by-side table of key data-quality numbers before and after cleaning.

    Args:
        raw: Raw bookings DataFrame.
        clean: Cleaned bookings DataFrame.

    Returns:
        DataFrame indexed by metric with ``before`` and ``after`` columns.
        A column that no longer exists shows ``"dropped"``.
    """
    before, after = data_quality_snapshot(raw), data_quality_snapshot(clean)
    metrics = ["rows", "duplicate_rows", "zero_guest_bookings", "negative_adr", "adr_max"]
    table: dict[str, tuple[object, object]] = {m: (before[m], after[m]) for m in metrics}

    for column in raw.columns[raw.isna().any()]:
        after_missing = int(clean[column].isna().sum()) if column in clean else "dropped"
        table[f"missing: {column}"] = (int(raw[column].isna().sum()), after_missing)

    table["total missing cells"] = (int(raw.isna().sum().sum()), int(clean.isna().sum().sum()))
    table["cancellation_rate_pct"] = (
        before["cancellation_rate_pct"],
        after["cancellation_rate_pct"],
    )
    # object dtype keeps counts as ints instead of upcasting everything to float.
    return pd.DataFrame.from_dict(table, orient="index", columns=["before", "after"], dtype=object)


def save_clean_data(df: pd.DataFrame, path: Path = config.CLEAN_DATA_PATH) -> Path:
    """Write the cleaned data to CSV.

    Args:
        df: Cleaned bookings DataFrame.
        path: Output CSV path.

    Returns:
        The path written to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(
        "Saved cleaned data: %s (%s rows x %s columns)", path.name, f"{len(df):,}", df.shape[1]
    )
    return path
