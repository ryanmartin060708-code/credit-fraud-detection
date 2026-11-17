"""Request telemetry with bounded labels and no request payloads or credentials."""
import json
import logging
import sys
import time
from datetime import datetime, timezone

from flask import Response, g, request
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST


def instrument(app):
    registry = CollectorRegistry()
    requests = Counter("http_requests_total", "Completed requests", ["method", "route", "status"], registry=registry)
    errors = Counter("http_errors_total", "HTTP responses with status >= 400", ["route"], registry=registry)
    latency = Histogram("http_request_duration_seconds", "Request duration", ["route"], registry=registry)
    logger = logging.getLogger(app.name + ".requests")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))

    @app.before_request
    def start():
        g.request_start = time.perf_counter()

    @app.after_request
    def finish(response):
        route = request.url_rule.rule if request.url_rule else "unmatched"
        # Scrapes do not count themselves; never label with raw paths or query strings.
        if route != "/metrics":
            duration = time.perf_counter() - g.request_start
            method = request.method if request.method in {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"} else "OTHER"
            requests.labels(method, route, str(response.status_code)).inc()
            latency.labels(route).observe(duration)
            if response.status_code >= 400:
                errors.labels(route).inc()
            logger.info(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "event": "http_request", "method": method, "route": route, "status": response.status_code, "duration_seconds": duration}))
        return response

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(registry), content_type=CONTENT_TYPE_LATEST)

    app.extensions["metrics_registry"] = registry
