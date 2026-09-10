"""Pure-Python tempo estimation (NumPy only) — the guaranteed floor.

Pipeline: STFT → spectral-flux onset envelope → autocorrelation → peak pick
in [MIN_BPM, MAX_BPM]. Returns (bpm, confidence) where confidence is the
prominence of the winning autocorrelation peak relative to the runner-up.

This is deliberately simple: its dominant failure mode (octave errors) is
exactly what the octave-disambiguation stage in math.py corrects. It exists
so local BPM analysis works out-of-the-box on installs where neither aubio
nor Essentia can be installed (HA OS / HA Container, musl, no compiler).

Executor only — never call from the event loop.
"""

from __future__ import annotations

import logging

import numpy as np

_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 22050
FRAME_SIZE = 1024
HOP_SIZE = 512

MIN_BPM = 40.0
MAX_BPM = 240.0


def estimate_bpm(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> tuple[float, float] | None:
    """Estimate (bpm, confidence) from mono float samples in [-1, 1].

    Confidence is the ratio of the winning autocorrelation peak to the
    strongest peak in the complementary lag region (roughly the octave
    competitor). Values near 1.0 mean an ambiguous octave; large values mean
    a clear winner.
    """
    if samples is None or len(samples) < sample_rate:  # need >= 1 s of audio
        return None

    x = samples.astype(np.float32)
    # Remove DC and normalize — cheap and makes flux thresholds stable.
    x = x - np.mean(x)
    peak = np.max(np.abs(x))
    if peak < 1e-4:  # silence
        return None
    x = x / peak

    # --- STFT magnitude spectrogram (Hann window) ---
    window = np.hanning(FRAME_SIZE)
    n_frames = 1 + (len(x) - FRAME_SIZE) // HOP_SIZE
    if n_frames < 8:
        return None
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, FRAME_SIZE),
        strides=(x.strides[0] * HOP_SIZE, x.strides[0]),
    ).copy()
    spec = np.abs(np.fft.rfft(frames * window, axis=1))  # (n_frames, bins)

    # --- Spectral flux (half-wave-rectified frame-to-frame difference) ---
    flux = np.maximum(np.diff(spec, axis=0), 0.0).sum(axis=1)
    if len(flux) < 8:
        return None
    flux = flux - flux.mean()
    std = flux.std()
    if std < 1e-9:
        return None
    flux = flux / std

    # --- Autocorrelation over the tempo-relevant lag range ---
    # Lag in frames: bpm -> seconds-per-beat -> frames.
    fps = sample_rate / HOP_SIZE  # envelope frame rate
    min_lag = int(fps * 60.0 / MAX_BPM)
    max_lag = int(fps * 60.0 / MIN_BPM) + 1
    max_lag = min(max_lag, len(flux) - 1)
    if max_lag <= min_lag:
        return None

    ac = np.correlate(flux, flux, mode="full")[len(flux) - 1:]
    ac = ac[: max_lag + 1]
    if min_lag >= len(ac):
        return None
    ac = ac / (ac[0] + 1e-12)  # normalize by zero-lag energy

    # Subtract the mean so a flat envelope doesn't produce spurious peaks.
    ac = ac - ac[min_lag:].mean()

    region = ac[min_lag:max_lag]
    if len(region) == 0 or np.max(region) <= 0:
        return None

    # Candidate peaks: local maxima above a fraction of the global max.
    # Weighted selection with a perceptual prior (listeners perceive tempo
    # mostly in ~[80, 180] BPM; log2-Gaussian centered at 120) reduces
    # octave errors at the source, before the disambiguation stage.
    global_max = float(np.max(region))
    if global_max <= 0:
        return None
    peak_thresh = 0.3 * global_max
    candidates: list[int] = []
    for i in range(1, len(region) - 1):
        if (
            region[i] >= region[i - 1]
            and region[i] >= region[i + 1]
            and region[i] >= peak_thresh
        ):
            candidates.append(i + min_lag)
    if not candidates:
        candidates = [int(np.argmax(region)) + min_lag]

    def perceptual_weight(lag: int) -> float:
        bpm_c = 60.0 * fps / lag
        return float(np.exp(-((np.log2(bpm_c / 120.0)) ** 2) / (2 * 0.75 ** 2)))

    best_idx = max(
        candidates,
        key=lambda lag: float(ac[lag]) * perceptual_weight(lag),
    )
    best = float(ac[best_idx])
    if best <= 0:
        return None

    bpm = 60.0 * fps / best_idx

    # Confidence: winner vs the strongest competitor in the octave region
    # (half or double the winning lag).
    competitor = 0.0
    for factor in (0.5, 2.0):
        comp_idx = int(round(best_idx * factor))
        lo, hi = comp_idx - max(1, comp_idx // 20), comp_idx + max(1, comp_idx // 20)
        lo = max(lo, min_lag)
        hi = min(hi, max_lag)
        if hi > lo:
            competitor = max(competitor, float(np.max(ac[lo:hi])))
    confidence = best / competitor if competitor > 0 else 2.0

    return bpm, min(confidence, 10.0)