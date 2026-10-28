"""Pre-compute the web app's page data: ``python -m app.build_cache``.

Runs the heavy work (load the full dataset, clean, engineer, aggregate) once at
build time so the web server only reads small files. ``main.py`` calls this too.
"""

from app.services import write_cache
from src import config

if __name__ == "__main__":
    config.setup_logging()
    write_cache()
