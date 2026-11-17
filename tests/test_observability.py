import json
import logging
from flask import Flask
from ops.observability import instrument


def test_metrics_cover_success_client_errors_server_errors_and_latency():
    app = Flask("test_observability")
    instrument(app)

    @app.get("/ok")
    def ok():
        return "ok"

    @app.get("/fail")
    def fail():
        raise RuntimeError("test failure")

    client = app.test_client()
    assert client.get("/ok?token=secret").status_code == 200
    assert client.get("/unknown-secret").status_code == 404
    assert client.get("/fail").status_code == 500
    response = client.get("/metrics")
    text = response.text
    assert 'http_requests_total{method="GET",route="/ok",status="200"} 1.0' in text
    assert 'http_errors_total{route="/fail"} 1.0' in text
    assert 'http_errors_total{route="unmatched"} 1.0' in text
    assert 'http_request_duration_seconds_count{route="/ok"} 1.0' in text
    assert 'secret' not in text
    assert 'route="/metrics"' not in text
    assert response.content_type.startswith("text/plain")


def test_json_log_does_not_capture_sensitive_request_fields():
    app = Flask("test_redaction")
    instrument(app)
    events = []

    class Capture(logging.Handler):
        def emit(self, record):
            events.append(json.loads(record.getMessage()))

    logger = logging.getLogger("test_redaction.requests")
    handler = Capture()
    logger.addHandler(handler)
    try:
        app.test_client().get("/secret-path?token=secret", headers={"Authorization": "Bearer secret"})
    finally:
        logger.removeHandler(handler)
    assert events[0]["route"] == "unmatched"
    assert events[0]["status"] == 404
    assert "secret" not in json.dumps(events)
