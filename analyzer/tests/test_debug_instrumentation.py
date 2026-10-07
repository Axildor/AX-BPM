"""Debug instrumentation tests (AXBPM_ANALYZE_DEBUG) + decode scale guard.

Tier 1: no onnxruntime, no real decode of exotic containers. The decode
scale-guard regression uses a mocked miniaudio module returning an
int16-range buffer — the exact production shape that produced the
degenerate mood signature.

Covers:
- pcm_stats correctness (abs-max/RMS/DC/first-1000 on known arrays).
- decode_traced reports the winning decoder; decode() contract unchanged.
- Debug inertness: without the env flag, /analyze payloads carry NO
  "debug" key and the inference engine collects nothing.
- Debug active: with the flag patched on, the payload carries the
  mel/pooling/PCM debug blocks and the cosine ring buffer fills.
- Scale-guard regression: an int16-range decode result must surface
  abs-max ≤ 1.0 after the guard (this test pins the Phase 3 fix; it
  FAILS against the current dead-dtype code, demonstrating the defect
  in Tier 1 before the Tier 2 run confirms it on real audio).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ax_bpm_analyzer import config as cfg
from ax_bpm_analyzer.decode import decode, decode_traced, pcm_stats
from ax_bpm_analyzer.inference import (
    _DEBUG_EMB_HISTORY,
    InferenceEngine,
)


# ---------------------------------------------------------------------------
# pcm_stats
# ---------------------------------------------------------------------------
def test_pcm_stats_known_array():
    sr = 44100
    x = np.zeros(sr, dtype=np.float32)
    x[0] = 0.5
    x[1] = -0.25
    stats = pcm_stats(x, sr, "miniaudio")
    assert stats["decoder"] == "miniaudio"
    assert stats["length_samples"] == sr
    assert stats["sample_rate"] == 44100
    assert stats["abs_max"] == 0.5
    assert stats["first_1000_min"] == -0.25
    assert stats["first_1000_max"] == 0.5
    assert stats["dc_offset"] == pytest.approx(0.25 / sr, abs=1e-5)
    assert stats["rms"] == pytest.approx(np.sqrt((0.25 + 0.0625) / sr), abs=1e-6)
    assert "mono" in stats["channels"]


def test_pcm_stats_int16_range_detected():
    """The Step B decision number: int16-range PCM shows abs-max ≈ 32768."""
    x = (np.linspace(-1.0, 1.0, 1000) * 32767).astype(np.float32)
    stats = pcm_stats(x, 44100, "miniaudio")
    assert stats["abs_max"] > 30000.0


# ---------------------------------------------------------------------------
# decode_traced / decode contract
# ---------------------------------------------------------------------------
def test_decode_traced_reports_decoder():
    """A decodable WAV reports the winning decoder; decode() stays 2-tuple."""
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes((np.sin(np.linspace(0, 100, 8000)) * 10000).astype(np.int16).tobytes())
    data = buf.getvalue()

    traced = decode_traced(data, sample_rate=16000)
    assert traced is not None
    samples, sr, decoder = traced
    assert decoder in ("miniaudio", "soundfile")
    assert sr == 16000
    assert samples.dtype == np.float32

    plain = decode(data, sample_rate=16000)
    assert plain is not None
    assert len(plain) == 2  # public contract unchanged
    np.testing.assert_array_equal(plain[0], samples)


def test_decode_traced_empty_input():
    assert decode_traced(b"") is None
    assert decode(b"") is None


# ---------------------------------------------------------------------------
# Debug inertness / activity in the inference engine
# ---------------------------------------------------------------------------
@pytest.fixture()
def engine():
    """Engine with a stubbed effnet session (no onnxruntime needed)."""
    m = types.SimpleNamespace(states={}, is_loaded=lambda n: False)
    eng = InferenceEngine(m)
    eng._sessions = {"effnet": object()}  # pretend loaded
    return eng


def _stub_effnet(monkeypatch, engine, dim=1280):
    """Replace _run with a deterministic pseudo-embedding generator."""

    def fake_run(name, patches):
        assert name == "effnet"
        n = patches.shape[0]
        # Embedding derived from patch content so distinct audio →
        # distinct embeddings (a stand-in for a healthy effnet).
        base = np.linspace(-1.0, 1.0, dim, dtype=np.float32)
        scale = float(np.mean(np.abs(patches))) + 1e-3
        return [np.tile(base * scale, (n, 1))]

    monkeypatch.setattr(engine, "_run", fake_run)


def test_debug_inert_without_flag(monkeypatch, engine):
    """Without AXBPM_ANALYZE_DEBUG, analyze() emits no debug key and the
    cosine ring buffer stays untouched."""
    monkeypatch.setattr(cfg, "ANALYZE_DEBUG", False)
    _stub_effnet(monkeypatch, engine)
    _DEBUG_EMB_HISTORY.clear()

    audio = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
    payload = engine.analyze(audio)
    assert payload is not None
    assert "debug" not in payload
    assert len(_DEBUG_EMB_HISTORY) == 0


def test_debug_active_collects_artifacts(monkeypatch, engine):
    """With the flag on: mel regime, patch count, pooled L2, cosine ring."""
    monkeypatch.setattr(cfg, "ANALYZE_DEBUG", True)
    _stub_effnet(monkeypatch, engine)
    _DEBUG_EMB_HISTORY.clear()

    audio = np.random.default_rng(1).standard_normal(16000 * 5).astype(np.float32) * 0.1
    payload = engine.analyze(audio)
    assert payload is not None
    dbg = payload["debug"]
    assert dbg["mel"]["patch_count"] >= 1
    assert dbg["mel"]["mel_frames"] > 0
    assert "logmel_global" in dbg["mel"]
    assert "l2_norm" in dbg["pooling"]
    assert len(_DEBUG_EMB_HISTORY) == 1

    # Second analysis → cosine_vs_previous present with one entry.
    audio2 = np.random.default_rng(2).standard_normal(16000 * 5).astype(np.float32) * 0.1
    payload2 = engine.analyze(audio2)
    assert payload2 is not None
    cos = payload2["debug"]["pooling"]["cosine_vs_previous"]
    assert len(cos) == 1
    assert -1.0 <= cos[0] <= 1.0


def test_debug_ring_buffer_bounded(monkeypatch, engine):
    monkeypatch.setattr(cfg, "ANALYZE_DEBUG", True)
    _stub_effnet(monkeypatch, engine)
    _DEBUG_EMB_HISTORY.clear()
    audio = np.random.default_rng(3).standard_normal(16000).astype(np.float32) * 0.1
    for _ in range(12):
        engine.analyze(audio)
    assert len(_DEBUG_EMB_HISTORY) <= 8


# ---------------------------------------------------------------------------
# Scale-guard regression (pins the Phase 3 fix; fails pre-fix)
# ---------------------------------------------------------------------------
def test_miniaudio_int16_output_scaled_to_float(monkeypatch):
    """A miniaudio decode returning int16-range samples must surface
    abs-max ≤ 1.0 — the production shape that produced the degenerate
    mood signature (dead dtype check pre-fix)."""
    fake_miniaudio = types.ModuleType("miniaudio")

    class SampleFormat:
        # Real wheel enum members (verified live): SIGNED16, FLOAT32, …
        SIGNED16 = 1
        FLOAT32 = 3

    class DecodedSoundFile:
        sample_format = SampleFormat.SIGNED16
        samples = (np.linspace(-1.0, 1.0, 16000) * 32767).astype(np.float32)

    fake_miniaudio.SampleFormat = SampleFormat
    # Production call shape: miniaudio.decode(data, nchannels=1, sample_rate=t)
    fake_miniaudio.decode = lambda data, **kwargs: DecodedSoundFile()
    monkeypatch.setitem(sys.modules, "miniaudio", fake_miniaudio)

    # Force the miniaudio decoder to win (soundfile absent; the analyzer
    # chain has no ffmpeg tier — per-runtime asymmetry, owner decision
    # 2026-10-07).
    monkeypatch.setattr(
        "ax_bpm_analyzer.decode._decode_with_soundfile", lambda d, t: None
    )

    result = decode_traced(b"fake-mp3-bytes", sample_rate=16000)
    assert result is not None
    samples, sr, decoder = result
    assert decoder == "miniaudio"
    # THE REGRESSION: unscaled int16 range would be ≈ 32767 here.
    assert float(np.max(np.abs(samples))) <= 1.0, (
        "int16-range PCM leaked unscaled into the analysis path — "
        f"abs-max={float(np.max(np.abs(samples)))}"
    )
