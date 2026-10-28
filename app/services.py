"""Data layer for the web app.

Everything here calls the project's ``src/`` functions - no preprocessing logic
is duplicated. Page payloads are computed once at start-up and stored as small
JSON-ready dictionaries, so the server does not keep full DataFrames in memory.
"""

import io
import logging
import math
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src import config, eda
from src.cleaning import before_after_summary, clean_data
from src.features import engineer_features
from src.load_data import column_overview, data_quality_snapshot, load_raw_data
from src.pipeline import load_artifacts, prepare_new_bookings

logger = logging.getLogger(__name__)

STEP_EXPLANATIONS: dict[str, str] = {
    "Drop zero-guest bookings": "0 adults, 0 children and 0 babies is not a real stay.",
    "Drop negative ADR": "A nightly price cannot be negative.",
    "Drop implausible counts": "Entry errors such as 55 adults or 10 babies (gap rule, not IQR).",
    "Drop exact duplicates": "Copies would leak between train and test. Mostly group blocks.",
    "company -> has_company flag": "94% empty: missing means 'no company', kept as a 0/1 flag.",
    "Fill agent + has_agent flag": "Missing means 'booked directly'. Filled with 0 + flag.",
    "Fill country 'Unknown'": "An honest label instead of guessing the most common country.",
    "Fill children 0": "4 rows; 0 is the most common value by far.",
    "meal Undefined -> SC": "Both labels mean 'no meal package' per the dataset paper.",
    "Cap extreme ADR": "Values above the 99.9th percentile capped; one 5,400 rate was an error.",
}


def _clean_number(value: object) -> object:
    """Convert numpy/pandas scalars to plain JSON values (NaN -> None).

    Args:
        value: Any scalar.

    Returns:
        A JSON-serialisable value.
    """
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _display(value: object) -> str:
    """Format a table value for display (thousands separators, at most 2 decimals).

    Args:
        value: Number or text.

    Returns:
        The formatted text.
    """
    value = _clean_number(value)
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.0f}" if value.is_integer() else f"{value:,.2f}"
    return str(value)


def _series_payload(table: pd.DataFrame) -> dict[str, list]:
    """Turn a cancellation table into chart-ready lists.

    Args:
        table: Output of an ``eda.cancellation_rate_*`` function.

    Returns:
        ``{"labels", "rate", "bookings"}``.
    """
    return {
        "labels": [str(label) for label in table.index],
        "rate": [_clean_number(v) for v in table["cancel_rate_pct"]],
        "bookings": [int(v) for v in table["bookings"]],
    }


def _monthly_payload(table: pd.DataFrame) -> dict[str, Any]:
    """Turn a month x hotel table into chart-ready lists.

    Args:
        table: Output of ``eda.monthly_by_hotel``.

    Returns:
        ``{"labels", "series": {hotel: values}}``.
    """
    return {
        "labels": table.index.tolist(),
        "series": {hotel: [_clean_number(v) for v in table[hotel]] for hotel in table.columns},
    }


def build_eda_payload(featured: pd.DataFrame) -> dict[str, Any]:
    """Compute every Explore-page chart for one slice of the data.

    Args:
        featured: Cleaned + engineered bookings (optionally filtered to one hotel).

    Returns:
        JSON-ready chart data.
    """
    requests = eda.cancellation_rate_by(featured, "total_of_special_requests").sort_index()
    return {
        "bookings": len(featured),
        "overall_rate": round(float(featured[config.TARGET].mean() * 100), 1),
        "lead_time": _series_payload(eda.cancellation_by_lead_time(featured)),
        "monthly_cancel": _monthly_payload(eda.monthly_by_hotel(featured, "cancel_rate_pct")),
        "monthly_adr": _monthly_payload(eda.monthly_by_hotel(featured, "adr")),
        "deposit": _series_payload(eda.cancellation_rate_ranked(featured, "deposit_type")),
        "segment": _series_payload(eda.cancellation_rate_ranked(featured, "market_segment")),
        "countries": _series_payload(eda.top_countries(featured)),
        "requests": _series_payload(requests),
    }


def _records(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame (with its index) into JSON-ready row dictionaries.

    Args:
        table: Any small DataFrame.

    Returns:
        One dictionary per row.
    """
    rows = table.reset_index().to_dict(orient="records")
    return [{key: _clean_number(value) for key, value in row.items()} for row in rows]


@dataclass
class AppData:
    """Everything the pages need, computed once."""

    overview: dict[str, Any]
    cleaning: dict[str, Any]
    eda: dict[str, dict[str, Any]]
    features: dict[str, Any]
    sample_raw: pd.DataFrame
    preprocessor: Any
    metadata: dict[str, Any]


def build_app_data() -> AppData:
    """Load the raw data once and compute every page payload with ``src/`` functions.

    Returns:
        The app's data, ready to render.
    """
    raw = load_raw_data()
    preprocessor, metadata = load_artifacts()
    cleaned = clean_data(raw)
    featured = engineer_features(cleaned.data)

    snapshot = data_quality_snapshot(raw)
    overview_table = column_overview(raw).rename_axis("column")
    overview = {
        "raw_rows": len(raw),
        "raw_columns": raw.shape[1],
        "clean_rows": len(cleaned.data),
        "missing_before": int(raw.isna().sum().sum()),
        "duplicates": snapshot["duplicate_rows"],
        "cancel_rate_raw": snapshot["cancellation_rate_pct"],
        "cancel_rate_clean": round(float(cleaned.data[config.TARGET].mean() * 100), 2),
        "n_inputs": len(metadata["input_features"]),
        "n_outputs": len(metadata["output_features"]),
        "n_train": metadata["rows"]["train"],
        "n_test": metadata["rows"]["test"],
        "hotels": raw["hotel"].value_counts().to_dict(),
        "columns": _records(overview_table.astype({"example": str})),
    }

    summary = before_after_summary(raw, cleaned.data).map(_display).rename_axis("metric")
    steps = cleaned.report.rename_axis("step")
    cleaning = {
        "before_after": _records(summary),
        "steps": [
            {**row, "why": STEP_EXPLANATIONS.get(row["step"], "")} for row in _records(steps)
        ],
        "duplicates": _records(eda.duplicates_profile(raw).rename_axis("group")),
        "adr_cap": cleaned.adr_cap,
        "adr": {
            "raw": {k: _clean_number(v) for k, v in raw["adr"].describe().round(2).items()},
            "clean": {
                k: _clean_number(v) for k, v in cleaned.data["adr"].describe().round(2).items()
            },
        },
    }

    eda_payloads = {
        name: build_eda_payload(
            featured if name == config.APP_HOTEL_FILTERS[0] else featured[featured["hotel"] == name]
        )
        for name in config.APP_HOTEL_FILTERS
    }

    features = {
        "engineered": config.ENGINEERED_FEATURES,
        "numeric": metadata["numeric_features"],
        "binary": metadata["binary_features"],
        "categorical": metadata["categorical_features"],
        "log": metadata["log_features"],
        "dropped": metadata["dropped_features"],
        "n_outputs": len(metadata["output_features"]),
        "adr_cap": metadata["adr_cap"],
    }
    sample = raw.sample(config.APP_SAMPLE_ROWS, random_state=config.RANDOM_STATE)
    logger.info("Web app data ready (%s raw rows summarised)", f"{len(raw):,}")
    return AppData(overview, cleaning, eda_payloads, features, sample, preprocessor, metadata)


# ---------------------------------------------------------------------------
# Processing uploaded files
# ---------------------------------------------------------------------------
@dataclass
class ProcessedFile:
    """A processed upload waiting to be downloaded."""

    name: str
    model_ready_csv: bytes
    clean_csv: bytes
    summary: dict[str, Any] = field(default_factory=dict)


class ResultStore:
    """Small thread-safe in-memory store that keeps only the newest results."""

    def __init__(self, max_items: int = config.APP_RESULT_CACHE_SIZE) -> None:
        """Create an empty store.

        Args:
            max_items: How many processed files to keep before evicting the oldest.
        """
        self._items: OrderedDict[str, ProcessedFile] = OrderedDict()
        self._lock = threading.Lock()
        self._max_items = max_items

    def add(self, item: ProcessedFile) -> str:
        """Store a result and return its download token.

        Args:
            item: The processed file.

        Returns:
            A random token used in the download URL.
        """
        token = uuid.uuid4().hex
        with self._lock:
            self._items[token] = item
            while len(self._items) > self._max_items:
                self._items.popitem(last=False)
        return token

    def get(self, token: str) -> ProcessedFile | None:
        """Return a stored result, or ``None`` if it expired or never existed.

        Args:
            token: Download token.

        Returns:
            The processed file or ``None``.
        """
        with self._lock:
            return self._items.get(token)


def read_uploaded_csv(content: bytes) -> pd.DataFrame:
    """Parse an uploaded CSV with the same missing-value rules as the raw data.

    Args:
        content: Raw bytes of the uploaded file.

    Returns:
        The parsed DataFrame.

    Raises:
        ValueError: If the file is empty or not a readable CSV.
    """
    if not content.strip():
        raise ValueError("The file is empty.")
    try:
        return pd.read_csv(io.BytesIO(content), na_values=config.MISSING_TOKENS, low_memory=False)
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"Could not read the file as CSV ({error}).") from error


def process_bookings(raw: pd.DataFrame, data: AppData, name: str) -> ProcessedFile:
    """Run the saved pipeline on uploaded raw bookings.

    Args:
        raw: Uploaded bookings in the original raw format.
        data: App data holding the fitted preprocessor and metadata.
        name: Original file name (used for the download names).

    Returns:
        The processed file with both CSV outputs and a summary.
    """
    model_ready, cleaning = prepare_new_bookings(raw, data.preprocessor, data.metadata)
    preview_columns = model_ready.columns[: config.APP_PREVIEW_ROWS + 2]
    summary = {
        "file": name,
        "rows_in": len(raw),
        "rows_out": len(model_ready),
        "columns_in": raw.shape[1],
        "columns_out": model_ready.shape[1],
        "has_target": config.TARGET in model_ready.columns,
        "missing_in": int(raw.isna().sum().sum()),
        "missing_out": int(model_ready.isna().sum().sum()),
        "steps": _records(cleaning.report.rename_axis("step")),
        "preview_columns": preview_columns.tolist(),
        "preview": _records(
            model_ready.loc[: config.APP_PREVIEW_ROWS - 1, preview_columns].round(3)
        )
        if not model_ready.empty
        else [],
    }
    for row in summary["preview"]:
        row.pop("index", None)
    return ProcessedFile(
        name=name,
        model_ready_csv=model_ready.to_csv(index=False, float_format="%.6g").encode("utf-8"),
        clean_csv=cleaning.data.to_csv(index=False).encode("utf-8"),
        summary=summary,
    )
