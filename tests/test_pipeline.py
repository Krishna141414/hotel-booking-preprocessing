"""Tests for src.pipeline (split, encoding, scaling, saved artifacts)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config, pipeline

GROUPS = pipeline.FeatureGroups(
    numeric=["lead_time", "adr"],
    binary=["has_agent"],
    categorical=["hotel", "country", "agent"],
)


@pytest.fixture
def featured() -> pd.DataFrame:
    """400 synthetic engineered bookings with a skewed column and a rare country.

    Country "ZZZ" appears exactly 3 times (0.75%), below the 1% one-hot threshold.
    """
    rng = np.random.default_rng(config.RANDOM_STATE)
    n = 400
    countries = rng.choice(["PRT", "GBR"], n)
    countries[:3] = "ZZZ"
    return pd.DataFrame(
        {
            "lead_time": rng.exponential(60, n).round(),
            "adr": rng.normal(100, 20, n).round(2),
            "has_agent": rng.integers(0, 2, n),
            "hotel": rng.choice(["City Hotel", "Resort Hotel"], n),
            "country": countries,
            "agent": rng.choice([0, 9, 240], n),
            "is_canceled": rng.integers(0, 2, n),
        }
    )


def test_split_data_is_stratified_and_complete(featured: pd.DataFrame) -> None:
    """Rows are split 80/20 with the same cancel rate and no overlap."""
    x_train, x_test, y_train, y_test = pipeline.split_data(featured, GROUPS)
    assert len(x_train) == 320 and len(x_test) == 80
    assert set(x_train.index).isdisjoint(x_test.index)
    assert abs(y_train.mean() - y_test.mean()) < 0.02
    assert list(x_train.columns) == GROUPS.all


def test_choose_log_columns_picks_only_skewed(featured: pd.DataFrame) -> None:
    """The exponential column is chosen; the normal one is not."""
    assert pipeline.choose_log_columns(featured, ["lead_time", "adr"]) == ["lead_time"]


def test_choose_log_columns_skips_negative_values() -> None:
    """Columns with negative values are never log-transformed."""
    df = pd.DataFrame({"x": [-1.0, 0, 0, 0, 0, 0, 100]})
    assert pipeline.choose_log_columns(df, ["x"]) == []


def test_preprocessor_is_fitted_on_train_only(featured: pd.DataFrame) -> None:
    """Scaler statistics equal the training-set statistics, not the full data's."""
    result = pipeline.fit_preprocessing(featured, GROUPS, adr_cap=300.0)
    scaler = result.preprocessor.named_transformers_["num"].named_steps["scale"]
    x_train, *_ = pipeline.split_data(featured, GROUPS)
    assert scaler.mean_[0] == pytest.approx(x_train["adr"].mean())
    assert scaler.mean_[0] != pytest.approx(featured["adr"].mean())


def test_fit_preprocessing_outputs(featured: pd.DataFrame) -> None:
    """Outputs are numeric, gap-free, carry the target and record metadata."""
    result = pipeline.fit_preprocessing(featured, GROUPS, adr_cap=300.0)
    assert result.train.isna().sum().sum() == 0
    assert config.TARGET in result.train.columns
    assert result.metadata["adr_cap"] == 300.0
    assert result.metadata["log_features"] == ["lead_time"]
    assert result.metadata["output_features"] == result.train.columns[:-1].tolist()


def test_rare_and_unknown_categories_go_to_infrequent(featured: pd.DataFrame) -> None:
    """Rare categories are merged; unseen ones map to 'infrequent' instead of crashing."""
    preprocessor = pipeline.build_preprocessor(GROUPS, log_columns=[])
    output = preprocessor.fit_transform(featured[GROUPS.all])
    assert "cat__country_ZZZ" not in output.columns
    assert output["cat__country_infrequent_sklearn"].sum() == 3
    unseen = preprocessor.transform(featured.head(1)[GROUPS.all].assign(country="NEW"))
    assert unseen["cat__country_infrequent_sklearn"].iloc[0] == 1.0


def test_artifacts_round_trip(featured: pd.DataFrame, tmp_path: Path) -> None:
    """Saved and reloaded artifacts produce identical output."""
    result = pipeline.fit_preprocessing(featured, GROUPS, adr_cap=300.0)
    paths = {"preprocessor_path": tmp_path / "p.joblib", "metadata_path": tmp_path / "m.json"}
    pipeline.save_artifacts(result, **paths)
    preprocessor, metadata = pipeline.load_artifacts(**paths)
    sample = featured[GROUPS.all].head(5)
    pd.testing.assert_frame_equal(
        preprocessor.transform(sample), result.preprocessor.transform(sample)
    )
    assert metadata == result.metadata


def test_load_artifacts_missing_file(tmp_path: Path) -> None:
    """A clear error tells the user to run the pipeline first."""
    with pytest.raises(FileNotFoundError, match="main.py"):
        pipeline.load_artifacts(tmp_path / "a.joblib", tmp_path / "b.json")


def test_prepare_new_bookings_on_real_rows(tmp_path: Path) -> None:
    """Raw rows -> clean -> features -> fitted transform, end to end."""
    from src.cleaning import clean_data
    from src.features import engineer_features
    from src.load_data import load_raw_data

    raw = load_raw_data().sample(3_000, random_state=config.RANDOM_STATE)
    cleaned = clean_data(raw)
    groups = pipeline.FeatureGroups.from_config()
    result = pipeline.fit_preprocessing(
        engineer_features(cleaned.data), groups, adr_cap=cleaned.adr_cap
    )
    new_raw = raw.head(200).drop(columns=config.TARGET)
    model_ready, cleaning = pipeline.prepare_new_bookings(
        new_raw, result.preprocessor, result.metadata
    )
    assert list(model_ready.columns) == result.metadata["output_features"]
    assert len(model_ready) == len(cleaning.data)
    assert model_ready.isna().sum().sum() == 0
