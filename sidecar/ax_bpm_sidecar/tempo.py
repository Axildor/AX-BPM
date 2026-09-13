"""aubio tempo detection on pre-decoded samples — the sidecar tempo tier.

Port of the integration's former `analyzer.py::_analyze_with_package`
(specdiff, win 1024 / hop 512, 44.1 kHz) to operate on in-memory float32
arrays instead of `aubio.source` — the sidecar decodes ONCE (decode.py)
and feeds aubio hop-sized chunks directly, exactly like the reference
demo loop (`samples, read = src(); is_beat = tempo(samples)`).

Parity contract: same method, same win/hop, same samplerate, same
median-IBI aggregation as the integration's aubio package path, so the
click-track golden test reproduces the old in-core accuracy.

aubio raises on malformed input; every failure degrades to None — the
payload simply omits `bpm` (never 0), mirroring the mood_scores
atomicity philosophy: a missing signal is never read as a value.
"""

from __future__ import annotations

import itertools
import logging
import statistics

import numpy as np

_LOGGER = logging.getLogger(__name__)

# Identical to the integration's aubio package parameters (parity).
WIN_S = 1024
HOP_S = 512
SAMPLERATE = 44100

# Minimum audio length: aubio needs several windows to lock a tempo;
# below ~5 s the median IBI is noise. Previews are 30 s — plenty.
MIN_SECONDS = 5.0


def aubio_available() -> bool:
    """True when the aubio package imports (tempo tier active)."""
    try:
        import aubio  # noqa: F401
    except ImportError:
        return False
    return True


def estimate_bpm(samples: np.ndarray, sample_rate: int = SAMPLERATE) -> tuple[float, float] | None:
    """Estimate (bpm, confidence) from mono float samples in [-1, 1].

    Confidence is the interquartile spread of the beat intervals
    relative to the median (lower = steadier beat). Returns None when
    aubio is unavailable, the clip is too short/silent, or fewer than
    two beats are found — callers treat None as "no tempo opinion".
    """
    if not aubio_available():
        return None
    import aubio

    if samples is None or len(samples) < sample_rate * MIN_SECONDS:
        return None

    x = np.asarray(samples, dtype=np.float32).flatten()
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak < 1e-4:  # silence
        return None
    if sample_rate != SAMPLERATE:
        # aubio tempo is rate-sensitive; the caller is expected to
        # provide 44.1 kHz audio (decode.py output). Resample here only
        # as a safety net so a wrong-rate buffer never silently
        # produces a scaled BPM.
        from scipy.signal import resample_poly

        g = np.gcd(int(sample_rate), SAMPLERATE)
        x = resample_poly(x, SAMPLERATE // g, int(sample_rate) // g).astype(
            np.float32
        )

    try:
        tempo_detector = aubio.tempo("specdiff", WIN_S, HOP_S, SAMPLERATE)
        beats: list[float] = []
        total = 0
        for start in range(0, len(x) - HOP_S + 1, HOP_S):
            chunk = x[start : start + HOP_S]
            is_beat = tempo_detector(chunk)
            if is_beat:
                beats.append(tempo_detector.get_last_s())
            total += len(chunk)
        _LOGGER.debug(
            "aubio tempo: %d beats over %.1f s", len(beats), total / SAMPLERATE
        )
    except Exception as err:  # noqa: BLE001 — aubio raises on bad input
        _LOGGER.debug("aubio tempo analysis failed: %s", err)
        return None

    if len(beats) < 2:
        return None
    intervals = [b - a for a, b in itertools.pairwise(beats)]
    median_ibi = statistics.median(intervals)
    if median_ibi <= 0:
        return None
    bpm = 60.0 / median_ibi
    if not 30.0 <= bpm <= 300.0:
        return None

    # Confidence: 1 - normalized IQR of intervals (steady beat → ~1.0).
    if len(intervals) >= 4:
        q1, q3 = np.percentile(intervals, [25, 75])
        spread = (q3 - q1) / median_ibi
    else:
        spread = 0.5  # too few intervals to judge — middling confidence
    confidence = float(max(0.0, min(1.0, 1.0 - spread)))
    return bpm, confidence