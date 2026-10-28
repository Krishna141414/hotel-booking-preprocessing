"""Tests for src.eda aggregations and a smoke test for src.visualize."""

from pathlib import Path

import pandas as pd
import pytest

from src import eda, visualize


@pytest.fixture
def bookings() -> pd.DataFrame:
    """Eight bookings over two hotels, two months and several lead times."""
    return pd.DataFrame(
        {
            "is_canceled": [1, 0, 1, 1, 0, 0, 0, 1],
            "hotel": ["City Hotel"] * 4 + ["Resort Hotel"] * 4,
            "arrival_month_num": [1, 1, 7, 7, 1, 1, 7, 7],
            "adr": [80.0, 100.0, 120.0, 140.0, 50.0, 70.0, 200.0, 180.0],
            "lead_time": [0, 5, 20, 100, 200, 400, 3, 60],
            "deposit_type": ["No Deposit"] * 6 + ["Non Refund"] * 2,
            "market_segment": ["Online TA"] * 5 + ["Groups"] * 3,
            "country": ["PRT", "PRT", "PRT", "GBR", "GBR", "FRA", "PRT", "ESP"],
        }
    )


def test_cancellation_rate_by_counts_and_rates(bookings: pd.DataFrame) -> None:
    """Rates are percentages and groups are sorted by booking count."""
    table = eda.cancellation_rate_by(bookings, "deposit_type")
    assert table.index.tolist() == ["No Deposit", "Non Refund"]
    assert table.loc["No Deposit", "bookings"] == 6
    assert table.loc["No Deposit", "cancel_rate_pct"] == 50.0
    assert table.loc["Non Refund", "cancel_rate_pct"] == 50.0


def test_cancellation_rate_by_respects_order(bookings: pd.DataFrame) -> None:
    """A given order is kept and unknown labels are skipped."""
    table = eda.cancellation_rate_by(bookings, "hotel", order=["Resort Hotel", "Spa", "City Hotel"])
    assert table.index.tolist() == ["Resort Hotel", "City Hotel"]


def test_cancellation_by_lead_time_keeps_chronological_order(bookings: pd.DataFrame) -> None:
    """Every lead-time group appears in order; 0-7 days holds 3 bookings here."""
    table = eda.cancellation_by_lead_time(bookings)
    assert table.index.astype(str).tolist() == ["0-7", "8-30", "31-90", "91-180", "181-365", "365+"]
    assert table.loc["0-7", "bookings"] == 3


def test_monthly_by_hotel(bookings: pd.DataFrame) -> None:
    """Twelve months per hotel; months without data are NaN, rates in percent."""
    table = eda.monthly_by_hotel(bookings, "cancel_rate_pct")
    assert table.shape == (12, 2)
    assert table.loc["Jan", "City Hotel"] == 50.0
    assert table.loc["Jul", "City Hotel"] == 100.0
    assert table["City Hotel"].isna().sum() == 10
    assert eda.monthly_by_hotel(bookings, "adr").loc["Jul", "Resort Hotel"] == 190.0


def test_monthly_by_hotel_rejects_unknown_metric(bookings: pd.DataFrame) -> None:
    """Only the two supported metrics are accepted."""
    with pytest.raises(ValueError, match="metric"):
        eda.monthly_by_hotel(bookings, "revenue")


def test_duplicates_profile() -> None:
    """Removed duplicates and kept rows are profiled separately."""
    raw = pd.DataFrame(
        {
            "is_canceled": [1, 1, 0],
            "market_segment": ["Groups", "Groups", "Direct"],
            "deposit_type": ["Non Refund", "Non Refund", "No Deposit"],
        }
    )
    profile = eda.duplicates_profile(raw)
    assert profile.loc["Removed duplicates", "rows"] == 1
    assert profile.loc["Removed duplicates", "groups_segment_pct"] == 100.0
    assert profile.loc["Kept rows", "cancel_rate_pct"] == 50.0


def test_figure_is_saved(bookings: pd.DataFrame, tmp_path: Path) -> None:
    """A figure renders with the shared style and is written as a PNG."""
    fig = visualize.plot_cancellation_by_lead_time(bookings)
    path = visualize.save_figure(fig, "lead_time", folder=tmp_path)
    assert path.exists() and path.stat().st_size > 0
