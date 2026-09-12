#!/usr/bin/env python3
"""Deterministic synthetic clip generator for AX-BPM Phase 0 golden validation.

Generates 4 fixed 30-second 44.1 kHz mono WAV clips. The clips are pure
synthesis (no randomness beyond a fixed seed) so that the golden workflow
and PR CI regenerate bit-identical audio from this committed script.

Clips:
  1. click_90bpm   - click track at 90 BPM (impulse train + decay envelope)
  2. click_160bpm  - click track at 160 BPM
  3. pink_noise    - low-pass filtered noise (spectral content, no rhythm)
  4. chirp_sweep   - logarithmic sine sweep 40 Hz -> 12 kHz

Usage:
    python generate_clips.py <output_dir>

Writes: <output_dir>/<name>.wav for each clip.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 44100
DURATION = 30.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)
SEED = 20260912  # fixed: date the golden set was frozen


def _write_wav(path: Path, samples: np.ndarray) -> None:
    """Write float64 samples in [-1, 1] as 16-bit PCM mono WAV."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def click_track(bpm: float) -> np.ndarray:
    """Impulse train with exponentially decaying 1 kHz body per click."""
    rng = np.random.default_rng(SEED)
    out = np.zeros(N_SAMPLES)
    period = int(SAMPLE_RATE * 60.0 / bpm)
    click_len = int(0.05 * SAMPLE_RATE)  # 50 ms body
    t = np.arange(click_len) / SAMPLE_RATE
    body = np.sin(2 * np.pi * 1000.0 * t) * np.exp(-t / 0.008)
    for start in range(0, N_SAMPLES - click_len, period):
        out[start : start + click_len] += body
    # tiny dither so the signal is not exactly zero between clicks
    out += rng.normal(0.0, 1e-6, N_SAMPLES)
    peak = np.max(np.abs(out))
    return out / peak


def filtered_noise() -> np.ndarray:
    """White noise low-passed with a 4th-order Butterworth at 2 kHz."""
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(SEED + 1)
    sos = butter(4, 2000.0, btype="low", fs=SAMPLE_RATE, output="sos")
    out = sosfilt(sos, rng.normal(0.0, 1.0, N_SAMPLES))
    return out / np.max(np.abs(out))


def chirp_sweep() -> np.ndarray:
    """Logarithmic sine sweep 40 Hz -> 12 kHz over the full 30 s."""
    f0, f1 = 40.0, 12000.0
    t = np.arange(N_SAMPLES) / SAMPLE_RATE
    # log chirp phase: integral of f0 * (f1/f0)^(t/T)
    beta = np.log(f1 / f0) / DURATION
    phase = 2 * np.pi * f0 * (np.exp(beta * t) - 1.0) / beta
    out = 0.8 * np.sin(phase)
    # fade in/out to avoid boundary clicks
    fade = int(0.01 * SAMPLE_RATE)
    out[:fade] *= np.linspace(0, 1, fade)
    out[-fade:] *= np.linspace(1, 0, fade)
    return out


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = {
        "click_90bpm": click_track(90.0),
        "click_160bpm": click_track(160.0),
        "pink_noise": filtered_noise(),
        "chirp_sweep": chirp_sweep(),
    }
    for name, samples in clips.items():
        path = out_dir / f"{name}.wav"
        _write_wav(path, samples)
        print(f"wrote {path} ({samples.shape[0]} samples)")


if __name__ == "__main__":
    main()