"""Encoding and scaling with a scikit-learn ColumnTransformer/Pipeline.

The data is split into train and test sets *before* anything is fitted, and the
preprocessor only ever sees the training rows during ``fit``. The fitted
preprocessor and a metadata file (feature lists, ADR cap, sizes) are saved so
new data (e.g. a CSV uploaded to the web app) is transformed exactly the same way.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config
from src.cleaning import CleaningResult, clean_data
from src.features import engineer_features

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureGroups:
    """Model input columns, grouped by how they are preprocessed."""

    numeric: list[str]
    binary: list[str]
    categorical: list[str]

    @property
    def all(self) -> list[str]:
        """Every input column, in transformer order."""
        return [*self.numeric, *self.binary, *self.categorical]

    @classmethod
    def from_config(cls) -> "FeatureGroups":
        """Build the groups from the lists in ``src/config.py``."""
        return cls(
            numeric=list(config.NUMERIC_FEATURES),
            binary=list(config.BINARY_FEATURES),
            categorical=list(config.CATEGORICAL_FEATURES),
        )


@dataclass
class PreprocessingResult:
    """Fitted preprocessor plus the model-ready train and test sets."""

    preprocessor: ColumnTransformer
    train: pd.DataFrame
    test: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


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
        raise ValueError(f"Missing model input columns: {missing}")


def split_data(
    df: pd.DataFrame,
    groups: FeatureGroups,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into stratified train/test sets, keeping only model input columns.

    Args:
        df: Cleaned + engineered bookings with the target column.
        groups: Columns to keep as model inputs.
        test_size: Share of rows held out for testing.
        random_state: Seed for a reproducible split.

    Returns:
        ``(X_train, X_test, y_train, y_test)``.
    """
    _require(df, [*groups.all, config.TARGET])
    x_train, x_test, y_train, y_test = train_test_split(
        df[groups.all],
        df[config.TARGET],
        test_size=test_size,
        random_state=random_state,
        stratify=df[config.TARGET],
    )
    logger.info(
        "Split: %s train / %s test rows (stratified, cancel rate %.1f%% / %.1f%%)",
        f"{len(x_train):,}",
        f"{len(x_test):,}",
        y_train.mean() * 100,
        y_test.mean() * 100,
    )
    return x_train, x_test, y_train, y_test


def choose_log_columns(
    x_train: pd.DataFrame,
    numeric: list[str],
    skew_threshold: float = config.LOG_SKEW_THRESHOLD,
    max_ratio: float = config.LOG_SKEW_MAX_RATIO,
) -> list[str]:
    """Pick numeric columns that ``log1p`` clearly de-skews (training data only).

    A column qualifies when its skew is above ``skew_threshold`` and ``log1p``
    shrinks the absolute skew to at most ``max_ratio`` of the original. Columns
    with negative values are never log-transformed.

    Args:
        x_train: Training features.
        numeric: Candidate numeric columns.
        skew_threshold: Minimum skew to consider a transform.
        max_ratio: Required improvement (after / before absolute skew).

    Returns:
        Columns to log-transform, in the original order.
    """
    chosen = []
    for column in numeric:
        values = x_train[column]
        if (values < 0).any():
            continue
        before, after = values.skew(), np.log1p(values).skew()
        if before > skew_threshold and abs(after) <= max_ratio * abs(before):
            chosen.append(column)
    logger.info("log1p applied to %d columns: %s", len(chosen), chosen)
    return chosen


def to_string(values: pd.DataFrame) -> pd.DataFrame:
    """Cast categories to text so numeric IDs (e.g. agent 9) are treated as labels.

    A named module-level function (not a lambda) so the fitted pipeline can be
    saved with joblib.

    Args:
        values: Categorical input columns.

    Returns:
        The same columns as strings.
    """
    return pd.DataFrame(values).astype(str)


def build_preprocessor(groups: FeatureGroups, log_columns: list[str]) -> ColumnTransformer:
    """Build the (unfitted) preprocessing ColumnTransformer.

    - skewed numeric: median impute -> log1p -> standard scale
    - other numeric: median impute -> standard scale
    - binary 0/1 flags: most-frequent impute, no scaling
    - categorical: most-frequent impute -> text -> one-hot (rare -> "infrequent")

    Imputers are a safety net for new data; the cleaned dataset has no gaps.

    Args:
        groups: Model input columns by type.
        log_columns: Numeric columns that get ``log1p`` first.

    Returns:
        A ColumnTransformer that outputs a pandas DataFrame.
    """
    plain_numeric = [c for c in groups.numeric if c not in log_columns]
    log_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    numeric_pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("to_string", FunctionTransformer(to_string, feature_names_out="one-to-one")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=config.MIN_CATEGORY_FREQUENCY,
                    sparse_output=False,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("log", log_pipeline, log_columns),
            ("num", numeric_pipeline, plain_numeric),
            ("bin", SimpleImputer(strategy="most_frequent"), groups.binary),
            ("cat", categorical_pipeline, groups.categorical),
        ],
        remainder="drop",
    )
    return preprocessor.set_output(transform="pandas")


def _with_target(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Attach the target column to transformed features.

    Args:
        features: Transformed feature matrix.
        target: Target values aligned with ``features``.

    Returns:
        Features plus the target as the last column.
    """
    return features.assign(**{config.TARGET: target.to_numpy()})


def fit_preprocessing(
    featured: pd.DataFrame, groups: FeatureGroups, adr_cap: float
) -> PreprocessingResult:
    """Split, fit the preprocessor on train only, and transform both sets.

    Args:
        featured: Cleaned + engineered bookings with the target.
        groups: Model input columns by type.
        adr_cap: ADR cap used during cleaning (stored for new data).

    Returns:
        Fitted preprocessor, model-ready train/test frames and metadata.
    """
    x_train, x_test, y_train, y_test = split_data(featured, groups)
    log_columns = choose_log_columns(x_train, groups.numeric)
    preprocessor = build_preprocessor(groups, log_columns)

    train_features = preprocessor.fit_transform(x_train)  # fit on train ONLY
    test_features = preprocessor.transform(x_test)
    logger.info(
        "Preprocessor fitted on train: %d input columns -> %d model features",
        len(groups.all),
        train_features.shape[1],
    )

    metadata: dict[str, Any] = {
        "target": config.TARGET,
        "test_size": config.TEST_SIZE,
        "random_state": config.RANDOM_STATE,
        "adr_cap": adr_cap,
        "numeric_features": groups.numeric,
        "log_features": log_columns,
        "binary_features": groups.binary,
        "categorical_features": groups.categorical,
        "input_features": groups.all,
        "output_features": train_features.columns.tolist(),
        "rows": {"train": len(x_train), "test": len(x_test)},
        "cancel_rate_pct": {
            "train": round(float(y_train.mean() * 100), 2),
            "test": round(float(y_test.mean() * 100), 2),
        },
        "versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__},
    }
    return PreprocessingResult(
        preprocessor=preprocessor,
        train=_with_target(train_features, y_train),
        test=_with_target(test_features, y_test),
        metadata=metadata,
    )


def save_artifacts(
    result: PreprocessingResult,
    preprocessor_path: Path = config.PREPROCESSOR_PATH,
    metadata_path: Path = config.METADATA_PATH,
) -> None:
    """Save the fitted preprocessor (joblib) and its metadata (JSON).

    Args:
        result: Output of :func:`fit_preprocessing`.
        preprocessor_path: Where to write the joblib file.
        metadata_path: Where to write the JSON metadata.
    """
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.preprocessor, preprocessor_path)
    metadata_path.write_text(json.dumps(result.metadata, indent=2), encoding="utf-8")
    logger.info("Saved %s and %s", preprocessor_path.name, metadata_path.name)


def save_model_ready(
    result: PreprocessingResult,
    train_path: Path = config.TRAIN_PROCESSED_PATH,
    test_path: Path = config.TEST_PROCESSED_PATH,
) -> None:
    """Write the model-ready train and test sets to CSV.

    Args:
        result: Output of :func:`fit_preprocessing`.
        train_path: Output path for the training set.
        test_path: Output path for the test set.
    """
    for frame, path in ((result.train, train_path), (result.test, test_path)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, float_format="%.6g")
        logger.info("Saved %s (%s rows x %s columns)", path.name, f"{len(frame):,}", frame.shape[1])


def load_artifacts(
    preprocessor_path: Path = config.PREPROCESSOR_PATH,
    metadata_path: Path = config.METADATA_PATH,
) -> tuple[ColumnTransformer, dict[str, Any]]:
    """Load the fitted preprocessor and its metadata.

    Args:
        preprocessor_path: joblib file written by :func:`save_artifacts`.
        metadata_path: JSON file written by :func:`save_artifacts`.

    Returns:
        ``(preprocessor, metadata)``.

    Raises:
        FileNotFoundError: If either file is missing (run ``python main.py``).
    """
    for path in (preprocessor_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"{path.name} not found - run `python main.py` first.")
    preprocessor = joblib.load(preprocessor_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return preprocessor, metadata


def prepare_new_bookings(
    raw: pd.DataFrame, preprocessor: ColumnTransformer, metadata: dict[str, Any]
) -> tuple[pd.DataFrame, CleaningResult]:
    """Clean, engineer and transform new raw bookings with the *fitted* artifacts.

    Nothing is re-fitted: the reference ADR cap and the saved preprocessor are
    reused, so new rows are treated exactly like the training data.

    Args:
        raw: Bookings in the original raw format (target column optional).
        preprocessor: Fitted preprocessor from :func:`load_artifacts`.
        metadata: Metadata from :func:`load_artifacts`.

    Returns:
        ``(model_ready, cleaning_result)``; ``model_ready`` includes the target
        column when the input had one.

    Raises:
        ValueError: If no valid rows remain after cleaning.
    """
    cleaning = clean_data(raw, adr_cap=metadata["adr_cap"])
    if cleaning.data.empty:
        raise ValueError("No valid rows left after cleaning - check the uploaded data.")
    featured = engineer_features(cleaning.data)
    _require(featured, metadata["input_features"])
    model_ready = preprocessor.transform(featured[metadata["input_features"]])
    if config.TARGET in featured.columns:
        model_ready = _with_target(model_ready, featured[config.TARGET])
    return model_ready.reset_index(drop=True), cleaning
