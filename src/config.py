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
