#!/usr/bin/env bash
# AX BPM Analyzer entrypoint.
#
# Home Assistant passes add-on options via /data/options.json (NOT env
# vars). Env vars remain supported as fallbacks for dev runs
# (addon/Dockerfile.dev usage: -e AXBPM_API_TOKEN=...).
set -e

OPTIONS_FILE="${AXBPM_OPTIONS_FILE:-/data/options.json}"

read_option() {
    # read_option <env_var_name> <json_key> <default>
    # Precedence: existing env var (dev) > options.json (HA add-on) > default.
    local _name="$1" _key="$2" _default="$3" _value=""
    if [ -n "${!_name:-}" ]; then
        return
    fi
    if [ -f "$OPTIONS_FILE" ]; then
        _value="$(python -c "
import json, sys
try:
    with open('$OPTIONS_FILE') as f:
        data = json.load(f)
except Exception:
    sys.exit(0)
value = data.get('$_key')
if value is not None:
    print(value)
" 2>/dev/null || true)"
    fi
    if [ -n "$_value" ]; then
        printf -v "$_name" '%s' "$_value"
        export "$_name"
    else
        printf -v "$_name" '%s' "$_default"
        export "$_name"
    fi
}

read_option AXBPM_API_TOKEN api_token ""
read_option AXBPM_MAX_ANALYZE_SECONDS max_analyze_seconds "60"
read_option AXBPM_INTRA_OP_THREADS intra_op_threads "2"
export AXBPM_DATA_DIR="${AXBPM_DATA_DIR:-/data}"

mkdir -p "$AXBPM_DATA_DIR"

exec python -m uvicorn ax_bpm_analyzer.api:app \
    --host 0.0.0.0 --port 8099 --log-level info
