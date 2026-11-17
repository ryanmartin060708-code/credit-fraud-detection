#!/usr/bin/env bash
# Supervise both processes; stop the container if either exits.
set -euo pipefail
streamlit run app/Home.py --server.port=8501 --server.address=0.0.0.0 &
dashboard_pid=$!
gunicorn --bind 0.0.0.0:9000 --workers 1 --threads 4 ops.health_service:app &
health_pid=$!
cleanup() {
  kill "$dashboard_pid" "$health_pid" 2>/dev/null || true
  wait "$dashboard_pid" "$health_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
wait -n "$dashboard_pid" "$health_pid"
exit 1
