"""Gunicorn settings for production (read automatically by `gunicorn app.app:app`)."""

import os

# Hosts such as Render pass the port to listen on in $PORT.
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# One worker keeps memory low on free hosting (the app holds the fitted pipeline and
# pre-computed page data); threads handle concurrent requests.
workers = 1
threads = 4

# Build the app (load data, compute page payloads) once, before serving.
preload_app = True

# Processing the full 119k-row file takes a few seconds; leave headroom.
timeout = 120

accesslog = "-"
errorlog = "-"
