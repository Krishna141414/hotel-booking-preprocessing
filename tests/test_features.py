"""Tests for the feature-engineering functions in src.features."""

import pandas as pd
import pytest

from src import config, features


@pytest.fixture
def clean_bookings() -> pd.DataFrame:
    """Three cleaned bookings with known, hand-checkable feature values."""
    return pd.DataFrame(
        {
            "stays_in_weekend_nights": [2, 0, 1],
            "stays_in_week_nights": [3, 0, 1],
            "adults": [2, 1, 2],
            "children": [1, 0, 0],
            "babies": [0, 0, 1],
            "adr": [90.0, 0.0, 100.0],
            "arrival_date_year": [2016, 2017, 2015],
            "arrival_date_month": ["July", "January", "December"],
            "arrival_date_day_of_month": [4, 1, 25],
            "country": ["PRT", "GBR", "Unknown"],
            "previous_cancellations": [0, 1, 3],
            "previous_bookings_not_canceled": [0, 3, 0],
        }
    )


def test_add_stay_features(clean_bookings: pd.DataFrame) -> None:
    """Nights and guests are summed; babies count as children for the flag."""
    result = features.add_stay_features(clean_bookings)
    assert result["total_nights"].tolist() == [5, 0, 2]
    assert result["total_guests"].tolist() == [3, 1, 3]
    assert result["has_children"].tolist() == [1, 0, 1]


def test_add_price_features(clean_bookings: pd.DataFrame) -> None:
    """Per-person price excludes babies; stay value = ADR x nights."""
    result = features.add_price_features(features.add_stay_features(clean_bookings))
    assert result["adr_per_person"].tolist() == [30.0, 0.0, 50.0]
    assert result["stay_value"].tolist() == [450.0, 0.0, 200.0]


def test_add_price_features_rejects_zero_paying_guests() -> None:
    """A booking with only babies would divide by zero and is rejected."""
    df = pd.DataFrame({"adr": [50.0], "adults": [0], "children": [0], "total_nights": [1]})
    with pytest.raises(ValueError, match="paying guests"):
        features.add_price_features(df)


def test_add_arrival_features(clean_bookings: pd.DataFrame) -> None:
    """Month number, weekday and season are derived from the split date columns."""
    result = features.add_arrival_features(clean_bookings)
    assert result["arrival_month_num"].tolist() == [7, 1, 12]
    # 2016-07-04 Monday, 2017-01-01 Sunday, 2015-12-25 Friday
    assert result["arrival_weekday"].tolist() == [0, 6, 4]
    assert result["arrival_season"].tolist() == ["Summer", "Winter", "Winter"]


def test_add_arrival_features_rejects_bad_month(clean_bookings: pd.DataFrame) -> None:
    """Unknown month names raise a clear error."""
    bad = clean_bookings.assign(arrival_date_month=["Jul", "January", "December"])
    with pytest.raises(ValueError, match="Jul"):
        features.add_arrival_features(bad)


def test_add_guest_history_features(clean_bookings: pd.DataFrame) -> None:
    """Domestic flag uses PRT; cancel ratio is 0 for guests with no history."""
    result = features.add_guest_history_features(clean_bookings)
    assert result["is_domestic"].tolist() == [1, 0, 0]
    assert result["previous_cancel_ratio"].tolist() == [0.0, 0.25, 1.0]


def test_engineer_features_adds_all_configured_columns(clean_bookings: pd.DataFrame) -> None:
    """The full run adds every engineered feature and keeps the row count."""
    result = features.engineer_features(clean_bookings)
    assert set(config.ENGINEERED_FEATURES) <= set(result.columns)
    assert len(result) == len(clean_bookings)


def test_engineer_features_requires_inputs() -> None:
    """Missing input columns raise a clear error."""
    with pytest.raises(ValueError, match="Missing columns"):
        features.engineer_features(pd.DataFrame({"adr": [1.0]}))
