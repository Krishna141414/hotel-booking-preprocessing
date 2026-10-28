"""Feature engineering and feature-selection helpers.

Every engineered feature is computed *within a single row* (no dataset-wide
statistics), so creating them before the train/test split cannot leak
information between the two sets.
"""

import calendar
import logging

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

MONTH_NAME_TO_NUMBER: dict[str, int] = {
    name: number for number, name in enumerate(calendar.month_name) if name
}

FEATURE_INPUT_COLUMNS: list[str] = [
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    *config.GUEST_COLUMNS,
    "adr",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_day_of_month",
    "country",
    "previous_cancellations",
    "previous_bookings_not_canceled",
]


def _require(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise if any of ``columns`` is missing from ``df``.

    Args:
        df: DataFrame to check.
        columns: Required column names.

    Raises:
        ValueError: If a column is missing.
    """
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns for feature engineering: {missing}")


def add_stay_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``total_nights``, ``total_guests`` and ``has_children``.

    Args:
        df: Cleaned bookings DataFrame.

    Returns:
        DataFrame with the three new columns.
    """
    _require(df, ["stays_in_weekend_nights", "stays_in_week_nights", *config.GUEST_COLUMNS])
    return df.assign(
        total_nights=df["stays_in_weekend_nights"] + df["stays_in_week_nights"],
        total_guests=df[config.GUEST_COLUMNS].sum(axis=1),
        has_children=((df["children"] + df["babies"]) > 0).astype(int),
    )


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``adr_per_person`` (per paying guest) and ``stay_value`` (whole stay).

    Babies usually stay free, so the per-person price divides by adults + children.
    Requires ``total_nights`` (see :func:`add_stay_features`).

    Args:
        df: Bookings DataFrame with ``total_nights``.

    Returns:
        DataFrame with the two price columns.

    Raises:
        ValueError: If a booking has no paying guests (would divide by zero).
    """
    _require(df, ["adr", "adults", "children", "total_nights"])
    paying_guests = df["adults"] + df["children"]
    if (paying_guests <= 0).any():
        raise ValueError("Found bookings with 0 paying guests; run cleaning first.")
    return df.assign(
        adr_per_person=(df["adr"] / paying_guests).round(2),
        stay_value=(df["adr"] * df["total_nights"]).round(2),
    )


def add_arrival_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add month number, weekday (0 = Monday) and season of the arrival date.

    Args:
        df: Bookings DataFrame with the split arrival-date columns.

    Returns:
        DataFrame with ``arrival_month_num``, ``arrival_weekday``, ``arrival_season``.

    Raises:
        ValueError: If a month name is not recognised or a date is invalid.
    """
    _require(df, ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"])
    month_num = df["arrival_date_month"].map(MONTH_NAME_TO_NUMBER)
    if month_num.isna().any():
        bad = sorted(df.loc[month_num.isna(), "arrival_date_month"].astype(str).unique())
        raise ValueError(f"Unrecognised arrival_date_month values: {bad}")

    arrival_date = pd.to_datetime(
        pd.DataFrame(
            {
                "year": df["arrival_date_year"],
                "month": month_num,
                "day": df["arrival_date_day_of_month"],
            }
        ),
        errors="raise",
    )
    return df.assign(
        arrival_month_num=month_num.astype(int),
        arrival_weekday=arrival_date.dt.day_of_week,
        arrival_season=month_num.map(config.SEASON_BY_MONTH),
    )


def add_guest_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``is_domestic`` and ``previous_cancel_ratio``.

    ``previous_cancel_ratio`` is the share of the guest's earlier bookings that
    were canceled (0 for guests with no history).

    Args:
        df: Bookings DataFrame.

    Returns:
        DataFrame with the two new columns.
    """
    _require(df, ["country", "previous_cancellations", "previous_bookings_not_canceled"])
    previous_total = df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    ratio = (df["previous_cancellations"] / previous_total.where(previous_total > 0)).fillna(0)
    return df.assign(
        is_domestic=(df["country"] == config.DOMESTIC_COUNTRY).astype(int),
        previous_cancel_ratio=ratio.round(3),
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run every feature-engineering step on cleaned data.

    Args:
        df: Cleaned bookings DataFrame.

    Returns:
        DataFrame with all columns in ``config.ENGINEERED_FEATURES`` added.
    """
    _require(df, FEATURE_INPUT_COLUMNS)
    result = add_guest_history_features(
        add_arrival_features(add_price_features(add_stay_features(df)))
    )
    logger.info(
        "Engineered %d features: %s", len(config.ENGINEERED_FEATURES), config.ENGINEERED_FEATURES
    )
    logger.info(
        "Shape after feature engineering: %s rows x %s columns", f"{len(result):,}", result.shape[1]
    )
    return result
