"""Flask application: pages, JSON API and file downloads.

Run locally:   flask --app app.app run      (or: python -m app.app)
Production:    gunicorn app.app:app         (settings in gunicorn.conf.py)
"""

import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from flask import Flask, Response, abort, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from app.services import AppData, ResultStore, build_app_data, process_bookings, read_uploaded_csv
from src import config

logger = logging.getLogger(__name__)

NAV_PAGES: list[tuple[str, str]] = [
    ("overview", "Overview"),
    ("cleaning", "Cleaning"),
    ("explore", "Explore"),
    ("process", "Process data"),
]
DOWNLOAD_KINDS: dict[str, str] = {"model-ready": "model_ready", "clean": "clean"}


def _csv_response(content: bytes, filename: str) -> Response:
    """Build a CSV file download response.

    Args:
        content: CSV bytes.
        filename: Suggested file name.

    Returns:
        The Flask response.
    """
    return Response(
        content,
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _error(message: str, status: int = 400) -> tuple[Response, int]:
    """Build a JSON error response.

    Args:
        message: Human-readable explanation.
        status: HTTP status code.

    Returns:
        ``(response, status)``.
    """
    return jsonify({"error": message}), status


def create_app(data: AppData | None = None) -> Flask:
    """Create the Flask app.

    Args:
        data: Pre-built app data (tests pass their own); built from the dataset if ``None``.

    Returns:
        The configured Flask application.
    """
    config.setup_logging()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.APP_MAX_UPLOAD_MB * 1024 * 1024
    data = data or build_app_data()
    results = ResultStore()

    @app.context_processor
    def inject_layout() -> dict[str, object]:
        """Make navigation and shared numbers available to every template."""
        return {"nav_pages": NAV_PAGES, "overview": data.overview}

    # -- Pages ----------------------------------------------------------------
    @app.get("/")
    def overview() -> str:
        """Dataset overview and pipeline summary."""
        return render_template("overview.html", page="overview", features=data.features)

    @app.get("/cleaning")
    def cleaning() -> str:
        """Before/after cleaning statistics and the per-step report."""
        return render_template("cleaning.html", page="cleaning", cleaning=data.cleaning)

    @app.get("/explore")
    def explore() -> str:
        """Interactive EDA charts."""
        return render_template("explore.html", page="explore", filters=config.APP_HOTEL_FILTERS)

    @app.get("/process")
    def process() -> str:
        """Upload a CSV and download the preprocessed output."""
        return render_template(
            "process.html",
            page="process",
            features=data.features,
            sample_rows=config.APP_SAMPLE_ROWS,
            max_mb=config.APP_MAX_UPLOAD_MB,
        )

    # -- API ------------------------------------------------------------------
    @app.get("/api/eda")
    def api_eda() -> Response | tuple[Response, int]:
        """Chart data for one hotel filter."""
        hotel = request.args.get("hotel", config.APP_HOTEL_FILTERS[0])
        if hotel not in data.eda:
            return _error(f"Unknown hotel filter. Use one of: {', '.join(data.eda)}.")
        return jsonify(data.eda[hotel])

    def _process_and_store(
        raw_name: str, frame_loader: Callable[[], pd.DataFrame]
    ) -> Response | tuple[Response, int]:
        """Process the DataFrame returned by ``frame_loader`` and store the result."""
        try:
            raw = frame_loader()
            processed = process_bookings(raw, data, raw_name)
        except ValueError as error:
            return _error(str(error))
        token = results.add(processed)
        return jsonify({"token": token, "summary": processed.summary})

    @app.post("/api/process")
    def api_process() -> Response | tuple[Response, int]:
        """Process an uploaded raw CSV with the saved pipeline."""
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            return _error("Choose a CSV file to upload.")
        if Path(upload.filename).suffix.lower() != ".csv":
            return _error("Only .csv files are supported.")
        content = upload.read()
        return _process_and_store(upload.filename, lambda: read_uploaded_csv(content))

    @app.post("/api/process-sample")
    def api_process_sample() -> Response | tuple[Response, int]:
        """Process the built-in sample of raw bookings."""
        return _process_and_store("sample_bookings.csv", lambda: data.sample_raw.copy())

    # -- Downloads ------------------------------------------------------------
    @app.get("/download/<token>/<kind>")
    def download(token: str, kind: str) -> Response:
        """Download a processed result (model-ready or cleaned CSV)."""
        item = results.get(token)
        if item is None or kind not in DOWNLOAD_KINDS:
            abort(404)
        stem = Path(item.name).stem
        content = item.model_ready_csv if kind == "model-ready" else item.clean_csv
        return _csv_response(content, f"{stem}_{DOWNLOAD_KINDS[kind]}.csv")

    @app.get("/sample.csv")
    def sample_csv() -> Response:
        """Download raw sample rows to try the upload flow."""
        return _csv_response(
            data.sample_raw.to_csv(index=False).encode("utf-8"), "sample_bookings.csv"
        )

    @app.get("/healthz")
    def healthz() -> tuple[str, int]:
        """Health check for the hosting platform."""
        return "ok", 200

    # -- Errors ---------------------------------------------------------------
    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_: RequestEntityTooLarge) -> tuple[Response, int]:
        """Reject uploads above the size limit with a clear message."""
        return _error(f"File is larger than {config.APP_MAX_UPLOAD_MB} MB.", 413)

    @app.errorhandler(404)
    def not_found(_: Exception) -> tuple[str, int]:
        """Friendly 404 page."""
        return render_template("404.html", page=""), 404

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=False)
