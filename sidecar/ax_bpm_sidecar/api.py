"""FastAPI service: POST /analyze (auth'd) + GET /health (open).

- /analyze: Bearer shared-secret (hmac.compare_digest), MAX_UPLOAD_BYTES
  → 413 before decode, post-decode truncation to MAX_ANALYZE_SECONDS,
  single inference worker with queue depth 1 (busy → 503 immediately),
  hard per-request timeout, malformed audio → 422.
- /health: unauthenticated (auto-detect + config-flow status line need
  it), per-model state, partial availability.
- Empty configured token → /analyze answers 401 with a logged hint (once).
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from . import config as cfg
from .decode import decode
from .inference import InferenceEngine
from .models import ModelManager

_LOGGER = logging.getLogger(__name__)

_TOKEN_HINT_LOGGED = False

# Module-level default so FastAPI doesn't see a call in the argument
# default (ruff B008); semantics identical to inline File(...).
_FILE_PARAM = File(...)



class AppState:
    """Module-level service state (populated by the lifespan)."""

    def __init__(self) -> None:
        self.models = ModelManager()
        self.engine = InferenceEngine(self.models)
        self.worker_lock = asyncio.Lock()


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.environ.get("AXBPM_SKIP_BOOTSTRAP"):
        state.models = ModelManager()
        state.models.verify_existing()
        state.engine = InferenceEngine(state.models)
        # Download missing models + load sessions. Any failure here
        # degrades health — the service still serves /health (and 503 on
        # /analyze), it never crashes the container.
        try:
            state.models.ensure_all()
            state.engine.load_sessions()
        except Exception as err:  # noqa: BLE001 — degraded, never crash
            _LOGGER.error("AX-BPM sidecar: startup degraded: %s", err)
    yield


app = FastAPI(title="AX-BPM sidecar", version="1.0.0", lifespan=lifespan)


def _check_token(request: Request) -> None:
    """Bearer shared-secret check (constant-time). /health stays open."""
    global _TOKEN_HINT_LOGGED
    expected = cfg.API_TOKEN
    if not expected:
        if not _TOKEN_HINT_LOGGED:
            _LOGGER.warning(
                "AX-BPM sidecar: no api_token configured — /analyze returns "
                "401 until a token is set in the add-on configuration"
            )
            _TOKEN_HINT_LOGGED = True
        raise HTTPException(status_code=401, detail="server has no api_token configured")
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    provided = auth[len("Bearer "):]
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="invalid token")


@app.get("/health")
async def health() -> JSONResponse:
    """Per-model health. Unauthenticated (auto-detect + diagnostics)."""
    return JSONResponse(
        {
            "status": state.models.status,
            "models_loaded": state.models.states,
        }
    )


@app.post("/analyze")
async def analyze(request: Request, file: UploadFile = _FILE_PARAM) -> JSONResponse:
    _check_token(request)

    # Input cap BEFORE reading the whole body into memory (OOM guard).
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > cfg.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="upload too large")

    data = await file.read()
    if len(data) > cfg.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="upload too large")
    if not data:
        raise HTTPException(status_code=422, detail="empty upload")

    if state.models.status == "error":
        raise HTTPException(status_code=503, detail="models unavailable")

    # Single inference worker, queue depth 1: busy → 503 immediately.
    if state.worker_lock.locked():
        raise HTTPException(status_code=503, detail="busy")

    async with state.worker_lock:
        try:
            payload = await asyncio.wait_for(
                asyncio.to_thread(_analyze_sync, data),
                timeout=cfg.ANALYZE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=503, detail="analysis timeout") from None
        except ValueError as err:
            raise HTTPException(status_code=422, detail=f"undecodable audio: {err}") from None
    if payload is None:
        raise HTTPException(status_code=503, detail="effnet model unavailable")
    return JSONResponse(payload)


def _analyze_sync(data: bytes) -> dict | None:
    """Blocking analysis (runs in a worker thread via asyncio.to_thread)."""
    decoded = decode(data)
    if decoded is None:
        raise ValueError("no decoder could handle the input")
    samples, sr = decoded
    if sr != cfg.SAMPLE_RATE:
        raise ValueError(f"decoder returned {sr} Hz, expected {cfg.SAMPLE_RATE}")

    # Post-decode truncation (latency guard on aarch64).
    max_samples = int(cfg.MAX_ANALYZE_SECONDS * cfg.SAMPLE_RATE)
    if len(samples) > max_samples:
        samples = samples[:max_samples]
    return state.engine.analyze(samples)


def main() -> None:
    """Entry point (run.sh calls this)."""
    import uvicorn

    uvicorn.run(app, host=cfg.HOST, port=cfg.PORT, log_level="info")


if __name__ == "__main__":
    main()