"""EDA aggregations shared by the static figures and the web app charts.

Each function takes a DataFrame and returns a small tidy table, so the same
numbers feed ``src/visualize.py`` (matplotlib) and ``app/`` (Chart.js).
"""

import calendar

import pandas as pd

from src import config

MONTH_ABBREVIATIONS: list[str] = list(calendar.month_abbr)[1:]


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
        raise ValueError(f"Missing columns for EDA: {missing}")


def cancellation_rate_by(df: pd.DataFrame, column: str, order: list | None = None) -> pd.DataFrame:
    """Count bookings and cancellation rate (%) per category of ``column``.

    Args:
        df: Bookings DataFrame with the target column.
        column: Column to group by.
        order: Optional category order; default sorts by number of bookings.

    Returns:
        DataFrame indexed by category with ``bookings`` and ``cancel_rate_pct``.
    """
    _require(df, [column, config.TARGET])
    table = df.groupby(column, observed=True)[config.TARGET].agg(
        bookings="size", cancel_rate_pct="mean"
    )
    table["cancel_rate_pct"] = (table["cancel_rate_pct"] * 100).round(1)
    if order is not None:
        return table.reindex([value for value in order if value in table.index])
    return table.sort_values("bookings", ascending=False)


def cancellation_by_lead_time(df: pd.DataFrame) -> pd.DataFrame:
    """Cancellation rate per lead-time group (days between booking and arrival).

    Args:
        df: Bookings DataFrame with ``lead_time``.

    Returns:
        DataFrame indexed by lead-time group, in chronological order.
    """
    _require(df, ["lead_time"])
    groups = pd.cut(
        df["lead_time"], bins=config.LEAD_TIME_BINS, labels=config.LEAD_TIME_LABELS
    ).rename("lead_time_days")
    table = cancellation_rate_by(df.assign(lead_time_days=groups), "lead_time_days")
    return table.reindex(config.LEAD_TIME_LABELS)


def monthly_by_hotel(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Monthly cancellation rate (%) or average ADR, one column per hotel.

    Args:
        df: Bookings DataFrame with ``arrival_month_num`` (see feature engineering).
        metric: ``"cancel_rate_pct"`` or ``"adr"``.

    Returns:
        DataFrame indexed by month abbreviation (Jan..Dec), columns = hotels.

    Raises:
        ValueError: If ``metric`` is not supported.
    """
    value_column = {"cancel_rate_pct": config.TARGET, "adr": "adr"}.get(metric)
    if value_column is None:
        raise ValueError(f"metric must be 'cancel_rate_pct' or 'adr', got {metric!r}")
    _require(df, ["hotel", "arrival_month_num", value_column])

    table = df.pivot_table(
        index="arrival_month_num", columns="hotel", values=value_column, aggfunc="mean"
    )
    if metric == "cancel_rate_pct":
        table = table * 100
    table = table.reindex(range(1, 13)).round(1)
    table.index = MONTH_ABBREVIATIONS
    table.columns.name = None
    return table


def top_countries(df: pd.DataFrame, n: int = config.TOP_N_COUNTRIES) -> pd.DataFrame:
    """Bookings and cancellation rate for the ``n`` countries with most bookings.

    Args:
        df: Bookings DataFrame with ``country``.
        n: Number of countries to return.

    Returns:
        DataFrame indexed by country code, most bookings first.
    """
    return cancellation_rate_by(df, "country").head(n)


def duplicates_profile(raw: pd.DataFrame) -> pd.DataFrame:
    """Compare exact-duplicate rows with the rest of the raw data.

    Args:
        raw: Raw bookings DataFrame (before cleaning).

    Returns:
        DataFrame with one row per group ("Removed duplicates", "Kept rows") and
        columns for row count, cancellation %, Groups-segment % and Non-Refund %.
    """
    _require(raw, [config.TARGET, "market_segment", "deposit_type"])
    is_duplicate = raw.duplicated(keep="first")
    parts = {"Removed duplicates": raw[is_duplicate], "Kept rows": raw[~is_duplicate]}
    return pd.DataFrame.from_dict(
        {
            name: {
                "rows": len(part),
                "cancel_rate_pct": round(part[config.TARGET].mean() * 100, 1),
                "groups_segment_pct": round((part["market_segment"] == "Groups").mean() * 100, 1),
                "non_refund_pct": round((part["deposit_type"] == "Non Refund").mean() * 100, 1),
            }
            for name, part in parts.items()
        },
        orient="index",
    )
