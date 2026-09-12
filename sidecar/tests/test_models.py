"""Model manager tests: verify-existing, download failure → degraded health,
sha256 mismatch, backoff. Downloads are stubbed (no network in tests)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ax_bpm_sidecar import config as cfg
from ax_bpm_sidecar.models import ModelManager


def _write_valid(tmp_path: Path, name: str) -> None:
    pin = cfg.MODEL_PINS[name]
    data = hashlib.sha256(name.encode()).digest() * 1024
    # Can't fake a real sha256 — instead write the pin's expected content by
    # patching the pin to match what we write.
    pin["size"] = len(data)
    pin["sha256"] = hashlib.sha256(data).hexdigest()
    (tmp_path / pin["filename"]).write_bytes(data)


def test_verify_existing_ok(tmp_path, monkeypatch):
    for name in ("effnet", "moodtheme"):
        _write_valid(tmp_path, name)
    mgr = ModelManager(tmp_path)
    mgr.verify_existing()
    assert mgr.states["effnet"] == "ok"
    assert mgr.states["moodtheme"] == "ok"
    assert mgr.states["danceability"] == "pending"


def test_verify_existing_corrupt(tmp_path):
    pin = cfg.MODEL_PINS["effnet"]
    (tmp_path / pin["filename"]).write_bytes(b"corrupted-data")
    mgr = ModelManager(tmp_path)
    mgr.verify_existing()
    assert mgr.states["effnet"] == "error"


def test_verify_existing_size_mismatch(tmp_path):
    pin = cfg.MODEL_PINS["effnet"]
    (tmp_path / pin["filename"]).write_bytes(b"x" * (pin["size"] + 1))
    mgr = ModelManager(tmp_path)
    mgr.verify_existing()
    assert mgr.states["effnet"] == "error"


def test_status_aggregation(tmp_path):
    mgr = ModelManager(tmp_path)
    for name in cfg.MODEL_PINS:
        mgr.states[name] = "ok"
    assert mgr.status == "ok"
    mgr.states["moodtheme"] = "error"
    assert mgr.status == "degraded"
    mgr.states["effnet"] = "error"
    assert mgr.status == "error"


def test_ensure_all_download_failure_degrades(tmp_path, monkeypatch):
    """Download failure → per-model error state, never raises."""
    monkeypatch.setattr(cfg, "DOWNLOAD_RETRIES", 2)
    monkeypatch.setattr(cfg, "DOWNLOAD_BACKOFF_SECONDS", 0.0)

    def fail(*_args, **_kwargs):
        return False

    mgr = ModelManager(tmp_path)
    monkeypatch.setattr(ModelManager, "_download", fail)
    mgr.ensure_all()
    assert all(state == "error" for state in mgr.states.values())
    assert mgr.status == "error"


def test_ensure_all_success(tmp_path, monkeypatch):
    """Successful download → ok state; existing valid files are kept."""
    for name in cfg.MODEL_PINS:
        _write_valid(tmp_path, name)
    mgr = ModelManager(tmp_path)

    def fail(*_args, **_kwargs):
        raise AssertionError("download must not be called for valid files")

    monkeypatch.setattr(ModelManager, "_download", fail)
    mgr.ensure_all()
    assert all(state == "ok" for state in mgr.states.values())
    assert mgr.status == "ok"


def test_download_retries_with_backoff(tmp_path, monkeypatch):
    """_download retries DOWNLOAD_RETRIES times with increasing backoff."""
    monkeypatch.setattr(cfg, "DOWNLOAD_RETRIES", 3)
    monkeypatch.setattr(cfg, "DOWNLOAD_BACKOFF_SECONDS", 0.0)
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    attempts: list[int] = []

    import urllib.request

    def fake_urlretrieve(url, tmp):
        attempts.append(1)
        raise OSError("network down")

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)
    mgr = ModelManager(tmp_path)
    pin = cfg.MODEL_PINS["danceability"]
    assert mgr._download("danceability", tmp_path / pin["filename"], pin) is False
    assert len(attempts) == 3
    assert len(sleeps) == 2  # no sleep after the final attempt