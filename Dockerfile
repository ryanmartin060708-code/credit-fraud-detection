# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────
# Credit Card Fraud Detection — Streamlit Dashboard
# ─────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (cached layer)
COPY requirements.txt .
RUN --mount=type=secret,id=pip_ca,target=/run/secrets/pip_ca \
    if [ -f /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi; \
    pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Create artefact directories
RUN mkdir -p data models reports

# Expose Streamlit port
EXPOSE 8501

# Streamlit configuration
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Operational service checks Streamlit; no curl dependency.
COPY requirements-ops.txt .
RUN --mount=type=secret,id=pip_ca,target=/run/secrets/pip_ca \
    if [ -f /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi; \
    pip install --no-cache-dir -r requirements-ops.txt \
 && useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 9000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9000/health', timeout=3)"
CMD ["bash", "ops/start.sh"]
