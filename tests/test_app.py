"""Tests for the Flask web app (pages, API, uploads, downloads)."""

import io

import pandas as pd
import pytest
from flask import Flask
from flask.testing import FlaskClient

from app.app import create_app
from app.services import AppData, ResultStore, build_app_data
from src import config


@pytest.fixture(scope="module")
def app_data() -> AppData:
    """Build the real app data once for all tests in this module."""
    return build_app_data()


@pytest.fixture(scope="module")
def app(app_data: AppData) -> Flask:
    """App instance using the shared data."""
    flask_app = create_app(app_data)
    flask_app.testing = True
    return flask_app


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """Test client."""
    return app.test_client()


@pytest.mark.parametrize("path", ["/", "/cleaning", "/explore", "/process", "/healthz"])
def test_pages_load(client: FlaskClient, path: str) -> None:
    """Every page renders successfully."""
    assert client.get(path).status_code == 200


def test_overview_shows_real_numbers(client: FlaskClient) -> None:
    """The overview page shows the pipeline's row counts."""
    html = client.get("/").get_data(as_text=True)
    assert "119,390" in html and "87,208" in html


def test_unknown_page_is_404(client: FlaskClient) -> None:
    """Unknown URLs get the custom 404 page."""
    response = client.get("/nope")
    assert response.status_code == 404
    assert "not in the dataset" in response.get_data(as_text=True)


@pytest.mark.parametrize("hotel", config.APP_HOTEL_FILTERS)
def test_eda_api(client: FlaskClient, hotel: str) -> None:
    """Chart data is returned for each filter with matching chart lengths."""
    data = client.get("/api/eda", query_string={"hotel": hotel}).get_json()
    assert data["bookings"] > 0
    assert len(data["lead_time"]["labels"]) == len(data["lead_time"]["rate"]) == 6
    assert len(data["monthly_cancel"]["labels"]) == 12


def test_eda_api_filters_by_hotel(client: FlaskClient) -> None:
    """A hotel filter returns only that hotel's monthly series."""
    data = client.get("/api/eda", query_string={"hotel": "Resort Hotel"}).get_json()
    assert list(data["monthly_cancel"]["series"]) == ["Resort Hotel"]


def test_eda_api_rejects_unknown_filter(client: FlaskClient) -> None:
    """An unknown hotel filter is a clear 400 error."""
    response = client.get("/api/eda", query_string={"hotel": "Spa"})
    assert response.status_code == 400
    assert "Unknown hotel filter" in response.get_json()["error"]


def test_process_sample_and_download(client: FlaskClient, app_data: AppData) -> None:
    """The sample is processed and both CSVs can be downloaded."""
    payload = client.post("/api/process-sample").get_json()
    summary = payload["summary"]
    assert summary["rows_in"] == config.APP_SAMPLE_ROWS
    assert 0 < summary["rows_out"] <= summary["rows_in"]
    assert summary["missing_out"] == 0
    assert summary["columns_out"] == len(app_data.metadata["output_features"]) + 1  # + target

    model_ready = client.get(f"/download/{payload['token']}/model-ready")
    assert model_ready.mimetype == "text/csv"
    frame = pd.read_csv(io.BytesIO(model_ready.data))
    assert len(frame) == summary["rows_out"]
    assert client.get(f"/download/{payload['token']}/clean").status_code == 200


def test_upload_without_target(client: FlaskClient, app_data: AppData) -> None:
    """An uploaded file without the target column is processed (no target in output)."""
    csv = app_data.sample_raw.head(50).drop(columns=config.TARGET).to_csv(index=False)
    response = client.post(
        "/api/process", data={"file": (io.BytesIO(csv.encode()), "new_bookings.csv")}
    )
    summary = response.get_json()["summary"]
    assert response.status_code == 200
    assert summary["has_target"] is False
    assert summary["columns_out"] == len(app_data.metadata["output_features"])


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("notes.txt", b"hello", "Only .csv"),
        ("empty.csv", b"", "empty"),
        ("wrong.csv", b"a,b\n1,2\n", "Missing required columns"),
    ],
)
def test_upload_errors_are_clear(
    client: FlaskClient, filename: str, content: bytes, message: str
) -> None:
    """Bad uploads return a 400 with a readable message instead of crashing."""
    response = client.post("/api/process", data={"file": (io.BytesIO(content), filename)})
    assert response.status_code == 400
    assert message in response.get_json()["error"]


def test_missing_file_field(client: FlaskClient) -> None:
    """Posting without a file asks the user to choose one."""
    response = client.post("/api/process", data={})
    assert response.status_code == 400


def test_unknown_download_token_is_404(client: FlaskClient) -> None:
    """Expired or invented tokens return 404."""
    assert client.get("/download/abc/model-ready").status_code == 404


def test_sample_csv_download(client: FlaskClient) -> None:
    """The raw sample can be downloaded and has the raw schema."""
    frame = pd.read_csv(io.BytesIO(client.get("/sample.csv").data))
    assert len(frame) == config.APP_SAMPLE_ROWS
    assert config.TARGET in frame.columns


def test_result_store_evicts_oldest() -> None:
    """Only the newest results are kept."""
    from app.services import ProcessedFile

    store = ResultStore(max_items=2)
    tokens = [
        store.add(ProcessedFile(name=str(i), model_ready_csv=b"", clean_csv=b"")) for i in range(3)
    ]
    assert store.get(tokens[0]) is None
    assert store.get(tokens[2]) is not None
