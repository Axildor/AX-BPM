"""Audio decode chain: miniaudio → soundfile → ffmpeg binary.

Mirrors the integration's decode.py chain, targeting 16 kHz mono float32
(the analyzer front end's input spec). Each decoder is probed lazily and
cached; a decoder that fails to import or errors falls through to the
next. The ffmpeg path uses the binary present in most container images.

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
import shutil
import tempfile
from pathlib import Path

import numpy as np

from . import config as cfg

_LOGGER = logging.getLogger(__name__)

_FFMPEG_TIMEOUT = 30.0


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
        return None
    try:
        decoded = miniaudio.decode(data, nchannels=1, sample_rate=target_rate)
        samples = np.asarray(decoded.samples, dtype=np.float32)
        if samples.dtype == np.int16:
            samples = samples / 32768.0
        return samples, target_rate
    except Exception as err:  # noqa: BLE001 — any decode failure falls through
        _LOGGER.debug("miniaudio decode failed: %s", err)
        return None


def _decode_with_soundfile(
    data: bytes, target_rate: int
) -> tuple[np.ndarray, int] | None:
    """Decode via soundfile (libsndfile) + scipy resample_poly."""
    try:
        import io

        import soundfile as sf
    except ImportError:
        return None
    try:
        samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        mono = samples.mean(axis=1)
        return _resample_mono(mono, int(sr), target_rate), target_rate
    except Exception as err:  # noqa: BLE001 — any decode failure falls through
        _LOGGER.debug("soundfile decode failed: %s", err)
        return None


def _decode_with_ffmpeg(
    data: bytes, target_rate: int
) -> tuple[np.ndarray, int] | None:
    """Decode via the ffmpeg binary (present in most container images)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    tmp = Path(tempfile.mkstemp(suffix=".audio", prefix="axbpm_")[1])
    try:
        tmp.write_bytes(data)
        import subprocess

        proc = subprocess.run(
            [
                ffmpeg,
                "-v", "quiet",
                "-i", str(tmp),
                "-ac", "1",
                "-ar", str(target_rate),
                "-f", "f32le",
                "-",
            ],
            capture_output=True,
            timeout=_FFMPEG_TIMEOUT,
            check=False,  # returncode inspected below (fall-through decoder)
        )
        if proc.returncode != 0 or not proc.stdout:
            return None
        samples = np.frombuffer(proc.stdout, dtype=np.float32)
        return samples, target_rate
    except Exception as err:  # noqa: BLE001 — any decode failure falls through
        _LOGGER.debug("ffmpeg decode failed: %s", err)
        return None
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def decode(
    data: bytes, sample_rate: int | None = None
) -> tuple[np.ndarray, int] | None:
    """Decode audio bytes → (mono float32 samples, sample_rate).

    sample_rate None → the front end's 16 kHz (existing behavior);
    44100 → the aubio tempo tier's preferred rate. Returns None when no
    decoder can handle the input (caller → 422).
    """
    if not data:
        return None
    target = int(sample_rate) if sample_rate else cfg.SAMPLE_RATE
    for decoder in (_decode_with_miniaudio, _decode_with_soundfile, _decode_with_ffmpeg):
        result = decoder(data, target)
        if result is not None and len(result[0]) > 0:
            return result
    return None


def downsample(
    samples: np.ndarray, sr: int, target: int | None = None
) -> np.ndarray:
    """Rational-factor downsample (44.1 kHz tempo buffer → 16 kHz front end)."""
    target = target or cfg.SAMPLE_RATE
    return _resample_mono(np.asarray(samples), int(sr), target)