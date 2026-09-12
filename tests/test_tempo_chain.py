"""Tests for the local tempo analysis chain (tempo_numpy, decode, analyzer)."""

from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
from ax_bpm.analyzer import AubioAnalyzer, aubio_available
from ax_bpm.decode import decode_available, decode_mono
from ax_bpm.tempo_numpy import estimate_bpm

SR = 22050


def _make_click_wav(bpm: float, seconds: float = 20.0) -> str:
    """Synthetic click track with a quiet bass line, as a WAV file."""
    t = np.arange(int(SR * seconds)) / SR
    click = np.zeros_like(t)
    beat = 60.0 / bpm
    for k in range(int(seconds / beat)):
        idx = int(k * beat * SR)
        if idx + 400 < len(click):
            click[idx:idx + 400] += (
                np.sin(2 * np.pi * 1000 * np.arange(400) / SR)
                * np.exp(-np.arange(400) / 80)
            )
    click += 0.15 * np.sin(2 * np.pi * 55 * t)
    click = (click / np.max(np.abs(click)) * 0.9 * 32767).astype(np.int16)
    path = tempfile.mktemp(suffix=".wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(click.tobytes())
    return path


def _wav_to_mp3(wav_path: str) -> str:
    mp3_path = tempfile.mktemp(suffix=".mp3")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", wav_path, "-b:a", "128k", mp3_path],
        check=True,
        capture_output=True,
    )
    return mp3_path


class TestEstimateBpm:
    def test_click_track_accuracy(self):
        for bpm_true in (70, 90, 120, 128, 174):
            t = np.arange(int(SR * 20)) / SR
            click = np.zeros_like(t)
            beat = 60.0 / bpm_true
            for k in range(int(20 / beat)):
                idx = int(k * beat * SR)
                if idx + 400 < len(click):
                    click[idx:idx + 400] += (
                        np.sin(2 * np.pi * 1000 * np.arange(400) / SR)
                        * np.exp(-np.arange(400) / 80)
                    )
            click += 0.15 * _bass(t)
            samples = click / np.max(np.abs(click))
            result = estimate_bpm(samples, SR)
            assert result is not None
            bpm, _confidence = result
            # Accept exact, half, or double — octave errors are corrected
            # downstream by math.apply_octave_disambiguation.
            ratios = {
                bpm / bpm_true,
                bpm / (bpm_true / 2),
                bpm / (bpm_true * 2),
            }
            assert any(0.95 <= r <= 1.05 for r in ratios), (
                f"true={bpm_true}, got={bpm:.1f}"
            )

    def test_silence_returns_none(self):
        result = estimate_bpm(np.zeros(SR * 10, dtype=np.float32), SR)
        assert result is None

    def test_too_short_returns_none(self):
        result = estimate_bpm(np.random.randn(SR // 2).astype(np.float32), SR)
        assert result is None


def _bass(t):
    return np.sin(2 * np.pi * 55 * t)


class TestDecodeChain:
    def test_decode_available(self):
        assert decode_available() is True

    def test_decode_mono_wav_via_ffmpeg(self):
        wav = _make_click_wav(120)
        try:
            result = decode_mono(wav)
            assert result is not None
            samples, sr = result
            assert samples.size > 0
            assert sr == 22050
        finally:
            Path(wav).unlink(missing_ok=True)

    def test_decode_mono_mp3(self):
        wav = _make_click_wav(128)
        mp3 = _wav_to_mp3(wav)
        try:
            result = decode_mono(mp3)
            assert result is not None
            samples, sr = result
            assert samples.size > 0
            assert sr == 22050
        finally:
            Path(wav).unlink(missing_ok=True)
            Path(mp3).unlink(missing_ok=True)

    def test_decode_missing_file(self):
        assert decode_mono("/nonexistent/file.mp3") is None


class TestAnalyzerChain:
    def test_aubio_available_with_decoder(self):
        assert aubio_available(None) is True

    def test_chain_returns_bpm_and_backend(self):
        wav = _make_click_wav(128)
        mp3 = _wav_to_mp3(wav)
        try:
            analyzer = AubioAnalyzer()
            bpm = analyzer._analyze_sync(mp3)
            assert bpm is not None
            assert 120 <= bpm <= 140
            assert analyzer.last_backend in {"aubio", "aubio_cli", "numpy"}
        finally:
            Path(wav).unlink(missing_ok=True)
        Path(mp3).unlink(missing_ok=True)

    def test_chain_on_missing_file(self):
        analyzer = AubioAnalyzer()
        assert analyzer._analyze_sync("/nonexistent/file.mp3") is None