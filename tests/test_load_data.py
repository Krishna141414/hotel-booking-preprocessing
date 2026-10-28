"""Tests for src.load_data."""

from pathlib import Path

import pandas as pd
import pytest

from src.load_data import data_quality_snapshot, load_raw_data, missing_summary


@pytest.fixture
def raw_csv(tmp_path: Path) -> Path:
    """Write a tiny CSV that mimics the mirror's quirks ("NULL" and "NA" strings)."""
    content = (
        "is_canceled,adults,children,babies,adr,agent,country\n"
        "0,2,0,0,100.0,9,PRT\n"
        "1,2,NA,0,-6.4,NULL,NULL\n"
        "0,0,0,0,0.0,NULL,GBR\n"
        "0,2,0,0,100.0,9,PRT\n"
    )
    path = tmp_path / "bookings.csv"
    path.write_text(content)
    return path


def test_load_raw_data_treats_null_string_as_missing(raw_csv: Path) -> None:
    """The literal string "NULL" must become NaN, not a category called "NULL"."""
    df = load_raw_data(raw_csv)
    assert df["agent"].isna().sum() == 2
    assert df["country"].isna().sum() == 1
    assert df["children"].isna().sum() == 1
    assert "NULL" not in df["country"].astype(str).tolist()


def test_load_raw_data_missing_file_raises(tmp_path: Path) -> None:
    """A clear error is raised when the CSV is absent."""
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "nope.csv")


def test_missing_summary_lists_only_affected_columns(raw_csv: Path) -> None:
    """Columns without missing values are excluded and the worst comes first."""
    summary = missing_summary(load_raw_data(raw_csv))
    assert list(summary.index) == ["agent", "children", "country"]
    assert summary.loc["agent", "missing_pct"] == 50.0


def test_data_quality_snapshot_counts_known_issues(raw_csv: Path) -> None:
    """Duplicates, zero-guest bookings and negative ADR are counted correctly."""
    snapshot = data_quality_snapshot(load_raw_data(raw_csv))
    assert snapshot["duplicate_rows"] == 1
    assert snapshot["zero_guest_bookings"] == 1
    assert snapshot["negative_adr"] == 1
    assert snapshot["cancellation_rate_pct"] == 25.0


def test_data_quality_snapshot_requires_columns() -> None:
    """Missing required columns raise a clear ValueError."""
    with pytest.raises(ValueError, match="Missing required columns"):
        data_quality_snapshot(pd.DataFrame({"adr": [1.0]}))
