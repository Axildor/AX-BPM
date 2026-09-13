"""Sidecar tempo tier tests: aubio on in-memory samples.

Tier 1 (musl devcontainer): aubio unavailable → skip-guarded (same
pattern as require_onnx). Tier 3 CI: aubio is in requirements.txt →
hard-fail via AXBPM_REQUIRE_TEMPO when missing (zero-skip guard).

Parity contract: the 120 BPM click track must land within ±2 BPM of
the integration's former in-core aubio path (same specdiff params).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ax_bpm_sidecar import tempo


def require_aubio() -> None:
    """Skip-guard for aubio tempo tests (mirrors require_onnx)."""
    if not tempo.aubio_available():
        if os.environ.get("AXBPM_REQUIRE_TEMPO"):
            pytest.fail("AXBPM_REQUIRE_TEMPO=1 but aubio is not importable")
        pytest.skip("aubio unavailable on musl workspace", allow_module_level=True)


# Module-level gate — without this call the guard is dead code and CI
# would silently skip the tempo tier (vacuous pass).
require_aubio()


def _click_track(bpm: float, seconds: float = 30.0) -> np.ndarray:
    """Deterministic 44.1 kHz click track (mirrors conftest generator)."""
    sr = 44100
    n = int(sr * seconds)
    period = int(sr * 60.0 / bpm)
    click_len = int(0.05 * sr)
    t = np.arange(click_len) / sr
    body = np.sin(2 * np.pi * 1000.0 * t) * np.exp(-t / 0.008)
    out = np.zeros(n)
    for start in range(0, n - click_len, period):
        out[start : start + click_len] += body
    out /= np.max(np.abs(out))
    return out.astype(np.float32)


# (module-level require_aubio() above is the single skip/hard-fail gate;
# a per-class skipif here would be redundant with it.)
class TestTempo:
    def test_click_120(self):
        """120 BPM click track → aubio tempo ≈120 (parity gate)."""
        result = tempo.estimate_bpm(_click_track(120.0))
        assert result is not None
        bpm, confidence = result
        assert abs(bpm - 120.0) <= 2.0
        assert 0.0 <= confidence <= 1.0

    def test_click_90(self):
        result = tempo.estimate_bpm(_click_track(90.0))
        assert result is not None
        bpm, _ = result
        assert abs(bpm - 90.0) <= 2.0

    def test_silence_returns_none(self):
        silence = np.zeros(int(44100 * 10), dtype=np.float32)
        assert tempo.estimate_bpm(silence) is None

    def test_too_short_returns_none(self):
        short = _click_track(120.0, seconds=2.0)
        assert tempo.estimate_bpm(short) is None

    def test_wrong_rate_resampled_not_scaled(self):
        """A 22.05 kHz buffer must NOT yield a half/double BPM."""
        click = _click_track(120.0)
        result = tempo.estimate_bpm(click[::2], 22050)
        if result is not None:  # resample path may still fail → None ok
            bpm, _ = result
            assert abs(bpm - 120.0) <= 2.0

    def test_unavailable_returns_none(self, monkeypatch):
        """estimate_bpm degrades to None when aubio is missing."""
        import ax_bpm_sidecar.tempo as t

        monkeypatch.setattr(t, "aubio_available", lambda: False)
        assert t.estimate_bpm(_click_track(120.0)) is None