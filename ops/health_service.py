"""Operational companion: health checks the running Streamlit HTTP service."""
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from flask import Flask, jsonify
from ops.observability import instrument


def create_app():
    app = Flask(__name__)
    instrument(app)

    @app.get("/health")
    def health():
        try:
            with urlopen("http://127.0.0.1:8501/_stcore/health", timeout=2) as response:
                healthy = response.status == 200
        except (HTTPError, URLError, TimeoutError, OSError):
            healthy = False
        return jsonify(status="ok" if healthy else "unavailable", dependency="streamlit"), 200 if healthy else 503

    return app


app = create_app()
