"""Audio decode chain: miniaudio → soundfile.

Per-runtime decoder asymmetry (owner decision, 2026-10-07): the
integration's decode copy (custom_components/ax_bpm/decode.py) KEEPS an
ffmpeg-binary tier — HA core ships the binary and the integration
manifest installs no wheels, so ffmpeg is the NumPy BPM floor's only
decoder there. The analyzer image installs ONLY requirements.txt
(miniaudio + soundfile wheels) and never runs apt, so an ffmpeg tier
here could only silently 422 — it is removed. The scale handling
(sample_format source-of-truth branch + belt-and-braces guard) is
decoder-agnostic and stays structurally identical in both copies.

Each decoder is probed lazily; a decoder that fails to import or errors
falls through to the next.

Since the aubio tempo tier moved in-process (tempo.py), decode() is
parameterized by target rate: the API layer decodes ONCE at 44.1 kHz
(aubio tempo is rate-sensitive — parity with the integration's former
in-core aubio path), runs tempo on it, then downsample()s to 16 kHz for
the log-mel front end. One decode, two consumers.

NOTE (bit-exactness scope): production decode/resample is NOT essentia's
MonoLoader resampleQuality=1 path — live-audio outputs are
tolerance-bounded, never claimed bit-exact. The golden clips bypass this
chain entirely (they are synthesized + resampled via libsamplerate in the
test harness).
"""

from __future__ import annotations

import logging

import numpy as np

from . import config as cfg

_LOGGER = logging.getLogger(__name__)


def _resample_mono(mono: np.ndarray, sr: int, target: int) -> np.ndarray:
    """Rational-factor resample to the target rate (scipy, quality poly)."""
    if sr == target:
        return np.asarray(mono, dtype=np.float32)
    import math

    from scipy.signal import resample_poly

    g = math.gcd(int(sr), target)
    return resample_poly(mono, target // g, int(sr) // g).astype(np.float32)


def _decode_with_miniaudio(
    data: bytes, target_rate: int
) -> tuple[np.ndarray, int] | None:
    """Decode via the miniaudio wheel (bundled dr_mp3/dr_wav, no system deps)."""
    try:
        import miniaudio
    except ImportError:
        return None  # absent decoder — expected, debug-level
    try:
        decoded = miniaudio.decode(data, nchannels=1, sample_rate=target_rate)
        samples = np.asarray(decoded.samples, dtype=np.float32)
        # Root fix (mood-degeneracy): the decoder's sample_format is the
        # source of truth. The old `samples.dtype == np.int16` check was
        # dead code — np.asarray(..., dtype=np.float32) coerces at
        # construction, so the check could never fire — and dr_mp3's
        # SIGNED16 output leaked unscaled (abs-max ≈ 32768) into the
        # mel front end, shifting logmel +9–10 and collapsing mood heads.
        if getattr(decoded, "sample_format", None) == (
            miniaudio.SampleFormat.SIGNED16
        ):
            samples = samples / 32768.0
        # Belt-and-braces only: any decoder that still leaks an int16-range
        # buffer is rescaled here (never the primary mechanism).
        if samples.size and float(np.max(np.abs(samples))) > 1.5:
            samples = samples / 32768.0
        return samples, target_rate
    except Exception as err:  # noqa: BLE001 — any decode failure falls through
        # Fall-through hygiene (mood-degeneracy closeout): an AVAILABLE
        # decoder that raises must warn loudly — the SIGNED_INT16
        # AttributeError was silently swallowed here, invisibly disabling
        # miniaudio (ffmpeg won that run). "Graceful degradation" means
        # fail-to-fallback LOUDLY.
        _LOGGER.warning("miniaudio decode failed (falling through): %s", err)
        return None


def _decode_with_soundfile(
    data: bytes, target_rate: int
) -> tuple[np.ndarray, int] | None:
    """Decode via soundfile (libsndfile) + scipy resample_poly."""
    try:
        import io

        import soundfile as sf
    except ImportError:
        return None  # absent decoder — expected, debug-level
    try:
        samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        mono = samples.mean(axis=1)
        return _resample_mono(mono, int(sr), target_rate), target_rate
    except Exception as err:  # noqa: BLE001 — any decode failure falls through
        _LOGGER.warning("soundfile decode failed (falling through): %s", err)
        return None


def decode(
    data: bytes, sample_rate: int | None = None
) -> tuple[np.ndarray, int] | None:
    """Decode audio bytes → (mono float32 samples, sample_rate).

    sample_rate None → the front end's 16 kHz (existing behavior);
    44100 → the aubio tempo tier's preferred rate. Returns None when no
    decoder can handle the input (caller → 422).
    """
    result = decode_traced(data, sample_rate)
    return None if result is None else (result[0], result[1])


def decode_traced(
    data: bytes, sample_rate: int | None = None
) -> tuple[np.ndarray, int, str] | None:
    """decode() + the winning decoder's name as a third tuple element.

    Debug instrumentation only (AXBPM_ANALYZE_DEBUG): the API layer uses
    this to report WHICH decoder produced the PCM. The public decode()
    contract is unchanged.
    """
    if not data:
        return None
    target = int(sample_rate) if sample_rate else cfg.SAMPLE_RATE
    for name, decoder in (
        ("miniaudio", _decode_with_miniaudio),
        ("soundfile", _decode_with_soundfile),
    ):
        result = decoder(data, target)
        if result is not None and len(result[0]) > 0:
            return result[0], result[1], name
    return None


def pcm_stats(samples: np.ndarray, sr: int, decoder: str) -> dict:
    """PCM diagnostics for one decoded buffer (debug mode only).

    Reports the Step B decision-table numbers: length, rate, abs-max,
    RMS, DC offset, first-1000-samples min/max. Channel handling is
    noted (all three decoders downmix to mono — nchannels=1 / mean(axis=1)
    / -ac 1 — there is no interleave path).
    """
    x = np.asarray(samples, dtype=np.float64)
    head = x[:1000]
    return {
        "decoder": decoder,
        "length_samples": int(x.size),
        "sample_rate": int(sr),
        "channels": "mono (downmixed at decode: nchannels=1 / mean(axis=1) / -ac 1)",
        "abs_max": round(float(np.max(np.abs(x))) if x.size else 0.0, 6),
        "rms": round(float(np.sqrt(np.mean(x * x))) if x.size else 0.0, 6),
        "dc_offset": round(float(np.mean(x)) if x.size else 0.0, 6),
        "first_1000_min": round(float(head.min()) if head.size else 0.0, 6),
        "first_1000_max": round(float(head.max()) if head.size else 0.0, 6),
    }


def downsample(
    samples: np.ndarray, sr: int, target: int | None = None
) -> np.ndarray:
    """Rational-factor downsample (44.1 kHz tempo buffer → 16 kHz front end)."""
    target = target or cfg.SAMPLE_RATE
    return _resample_mono(np.asarray(samples), int(sr), target)
