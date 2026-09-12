#!/usr/bin/env bash
# AX-BPM Sidecar entrypoint.
# Add-on options are passed as env vars by Home Assistant (uppercase).
set -e

export AXBPM_API_TOKEN="${API_TOKEN:-}"
export AXBPM_MAX_ANALYZE_SECONDS="${MAX_ANALYZE_SECONDS:-60}"
export AXBPM_INTRA_OP_THREADS="${INTRA_OP_THREADS:-2}"
export AXBPM_DATA_DIR="${AXBPM_DATA_DIR:-/data}"

mkdir -p "$AXBPM_DATA_DIR"

exec python -m uvicorn ax_bpm_sidecar.api:app \
    --host 0.0.0.0 --port 8099 --log-level info