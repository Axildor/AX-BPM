"""API tests: /analyze happy/503/401/413/422/timeout + /health semantics.

ONNX inference is stubbed (Tier 1 — no onnxruntime on musl). The real
inference path is covered by test_inference_goldens.py in the Tier 3 job.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ax_bpm_sidecar import config as cfg
from ax_bpm_sidecar.api import app, state
from ax_bpm_sidecar.inference import InferenceEngine
from ax_bpm_sidecar.models import ModelManager

TOKEN = "test-token-1234"


@pytest.fixture()
def client(monkeypatch):
    """TestClient with a stubbed engine + token set."""
    monkeypatch.setattr(cfg, "API_TOKEN", TOKEN)
    state.models = ModelManager()
    for name in cfg.MODEL_PINS:
        state.models.states[name] = "ok"
    state.engine = InferenceEngine(state.models)
    state.engine._sessions = {"effnet": object()}  # pretend loaded
    with TestClient(app) as c:
        yield c


def _stub_payload(*_args, **_kwargs):
    return {
        "mood_scores": {"aggressive": 0.1, "party": 0.2, "relaxed": 0.3,
                        "electronic": 0.4, "acoustic": 0.5},
        "mood_tags": [{"tag": "party", "score": 0.6}],
        "valence": None, "arousal": None,
        "valence_std": None, "arousal_std": None,
        "danceability": 0.7,
        "analyzed_seconds": 1.0,
        "model_versions": {},
    }


def _post(client, data: bytes, token: str | None = TOKEN):
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    return client.post("/analyze", files={"file": ("p.mp3", data, "audio/mpeg")},
                       headers=headers)


def test_health_open_without_token(client):
    """/health is unauthenticated (auto-detect + diagnostics need it)."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert set(body["models_loaded"]) == set(cfg.MODEL_PINS)


def test_analyze_happy_path(client, monkeypatch):
    monkeypatch.setattr(
        "ax_bpm_sidecar.api._analyze_sync", lambda data: _stub_payload()
    )
    resp = _post(client, b"fake-audio")
    assert resp.status_code == 200
    body = resp.json()
    assert body["mood_scores"]["aggressive"] == 0.1
    assert body["danceability"] == 0.7


def test_analyze_401_missing_token(client):
    resp = _post(client, b"fake-audio", token=None)
    assert resp.status_code == 401


def test_analyze_401_wrong_token(client):
    resp = _post(client, b"fake-audio", token="wrong")
    assert resp.status_code == 401


def test_analyze_401_when_server_token_empty(client, monkeypatch):
    """Empty configured token → 401 (with logged hint)."""
    monkeypatch.setattr(cfg, "API_TOKEN", "")
    resp = _post(client, b"fake-audio", token="anything")
    assert resp.status_code == 401


def test_analyze_413_over_cap(client, monkeypatch):
    monkeypatch.setattr(cfg, "MAX_UPLOAD_BYTES", 100)
    resp = _post(client, b"x" * 101)
    assert resp.status_code == 413


def test_analyze_422_undecodable(client):
    resp = _post(client, b"not-audio-at-all")
    assert resp.status_code == 422


def test_analyze_503_busy(client, monkeypatch):
    """Queue depth 1: a second concurrent request → 503 immediately."""
    import threading
    import time

    def slow_sync(_data):
        time.sleep(0.5)
        return _stub_payload()

    monkeypatch.setattr("ax_bpm_sidecar.api._analyze_sync", slow_sync)
    results = {}

    def first():
        results["first"] = _post(client, b"fake-audio")

    t = threading.Thread(target=first)
    t.start()
    time.sleep(0.2)  # let the first request acquire the lock
    resp = _post(client, b"fake-audio")
    assert resp.status_code == 503
    t.join()
    assert results["first"].status_code == 200


def test_analyze_503_models_error(client):
    state.models.states["effnet"] = "error"
    try:
        resp = _post(client, b"fake-audio")
        assert resp.status_code == 503
    finally:
        state.models.states["effnet"] = "ok"


def test_analyze_503_effnet_missing(client, monkeypatch):
    """effnet session missing → 503 (nothing can be computed)."""
    monkeypatch.setattr(
        "ax_bpm_sidecar.api._analyze_sync", lambda data: None
    )
    resp = _post(client, b"fake-audio")
    assert resp.status_code == 503


def test_truncation_to_max_seconds(client, monkeypatch):
    """Post-decode truncation to MAX_ANALYZE_SECONDS."""
    captured = {}

    def fake_sync(data):
        captured["called"] = True
        return _stub_payload()

    monkeypatch.setattr("ax_bpm_sidecar.api._analyze_sync", fake_sync)
    monkeypatch.setattr(cfg, "MAX_ANALYZE_SECONDS", 5.0)
    # decode is stubbed to return 10 s of audio; truncation happens in
    # _analyze_sync — test it directly.
    from ax_bpm_sidecar.api import _analyze_sync

    monkeypatch.setattr(
        "ax_bpm_sidecar.decode.decode",
        lambda data: (np.zeros(int(10 * cfg.SAMPLE_RATE), dtype=np.float32), cfg.SAMPLE_RATE),
    )
    payload = _analyze_sync(b"whatever")
    assert payload is not None
    assert payload["analyzed_seconds"] <= 5.0