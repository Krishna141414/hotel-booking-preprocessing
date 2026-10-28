"""Central configuration: paths, dataset source, column groups and constants.

Every other module imports from here so there are no hardcoded paths or
magic numbers scattered around the codebase.
"""

import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
RAW_DATA_PATH: Path = RAW_DATA_DIR / "hotel_bookings.csv"
CLEAN_DATA_PATH: Path = PROCESSED_DATA_DIR / "hotel_bookings_clean.csv"

MODELS_DIR: Path = PROJECT_ROOT / "models"
FIGURES_DIR: Path = PROJECT_ROOT / "reports" / "figures"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

# ---------------------------------------------------------------------------
# Dataset source (public TidyTuesday mirror of Antonio, Almeida & Nunes, 2019)
# ---------------------------------------------------------------------------
_MIRROR_PATH: str = "data/2020/2020-02-11/hotels.csv"
DATA_URLS: tuple[str, ...] = (
    f"https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/{_MIRROR_PATH}",
    f"https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/{_MIRROR_PATH}",
)
DOWNLOAD_TIMEOUT_SECONDS: int = 60

EXPECTED_ROWS: int = 119_390
EXPECTED_COLUMNS: int = 32

# The mirror writes missing values in some columns as the literal string "NULL".
MISSING_TOKENS: list[str] = ["NULL"]

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------
TARGET: str = "is_canceled"

# Known only after the stay/cancellation happens -> would leak the target.
LEAKAGE_COLUMNS: list[str] = ["reservation_status", "reservation_status_date"]

# Numeric IDs that are really categories (agent #9 is not "bigger" than agent #1).
ID_AS_CATEGORY_COLUMNS: list[str] = ["agent", "company"]

GUEST_COLUMNS: list[str] = ["adults", "children", "babies"]

# ---------------------------------------------------------------------------
# Cleaning rules (reasoning in docs/decisions.md, section 2)
# ---------------------------------------------------------------------------
# Agent IDs start at 1, so 0 is a safe "booked without an agent" code.
NO_AGENT_ID: int = 0
UNKNOWN_COUNTRY: str = "Unknown"
# Per the dataset paper, "Undefined" and "SC" both mean no meal package.
MEAL_REPLACEMENTS: dict[str, str] = {"Undefined": "SC"}

# Highest value seen in a meaningful number of bookings, just before a clear gap
# in the frequency table (e.g. babies: 0, 1, 2, then 9 and 10). Above = entry error.
MAX_PLAUSIBLE_COUNTS: dict[str, int] = {
    "adults": 4,
    "children": 3,
    "babies": 2,
    "required_car_parking_spaces": 3,
}

# ADR above this quantile is capped (winsorised) to the quantile value.
ADR_CAP_QUANTILE: float = 0.999

# ---------------------------------------------------------------------------
# Feature engineering (reasoning in docs/decisions.md, section 3)
# ---------------------------------------------------------------------------
# Both hotels are in Portugal, so PRT = domestic guest.
DOMESTIC_COUNTRY: str = "PRT"

# Meteorological seasons (northern hemisphere), keyed by month number.
SEASON_BY_MONTH: dict[int, str] = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}  # fmt: skip

# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
# Lead-time groups in days: same week, month, quarter, half-year, year, longer.
LEAD_TIME_BINS: list[float] = [-1, 7, 30, 90, 180, 365, float("inf")]
LEAD_TIME_LABELS: list[str] = ["0-7", "8-30", "31-90", "91-180", "181-365", "365+"]
TOP_N_COUNTRIES: int = 10
# Categories smaller than this share of bookings are left out of rate charts (too noisy).
EDA_MIN_GROUP_SHARE: float = 0.001
FIGURE_DPI: int = 150

ENGINEERED_FEATURES: list[str] = [
    "total_nights",
    "total_guests",
    "has_children",
    "adr_per_person",
    "stay_value",
    "arrival_month_num",
    "arrival_weekday",
    "arrival_season",
    "is_domestic",
    "previous_cancel_ratio",
]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT: str = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"
LOG_DATE_FORMAT: str = "%H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with the project's standard format.

    Args:
        level: Minimum log level to display.
    """
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, force=True)


# ---------------------------------------------------------------------------
# Encoding + scaling (reasoning in docs/decisions.md, section 5)
# ---------------------------------------------------------------------------
PREPROCESSOR_PATH: Path = MODELS_DIR / "preprocessor.joblib"
METADATA_PATH: Path = MODELS_DIR / "preprocessing_metadata.json"
TRAIN_PROCESSED_PATH: Path = PROCESSED_DATA_DIR / "train_processed.csv"
TEST_PROCESSED_PATH: Path = PROCESSED_DATA_DIR / "test_processed.csv"

TEST_SIZE: float = 0.2
# One-hot categories seen in < 1% of training rows are merged into "infrequent".
MIN_CATEGORY_FREQUENCY: float = 0.01
# log1p is applied when a column's skew is above this ...
LOG_SKEW_THRESHOLD: float = 1.0
# ... and log1p shrinks |skew| to at most this fraction of the original.
LOG_SKEW_MAX_RATIO: float = 0.5

# Candidate model inputs, before feature selection. The hard-leakage columns
# (reservation_status*) are never candidates.
NUMERIC_FEATURES: list[str] = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "total_nights",
    "adults",
    "children",
    "babies",
    "total_guests",
    "adr",
    "adr_per_person",
    "stay_value",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "previous_cancel_ratio",
    "booking_changes",
    "days_in_waiting_list",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "arrival_month_num",
    "arrival_weekday",
]
BINARY_FEATURES: list[str] = [
    "is_repeated_guest",
    "has_company",
    "has_agent",
    "has_children",
    "is_domestic",
]
CATEGORICAL_FEATURES: list[str] = [
    "hotel",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
    "arrival_date_month",
    "arrival_season",
    "country",
    "agent",
]
