"""Audio decode chain: miniaudio → soundfile → ffmpeg binary.

Mirrors the integration's decode.py chain, targeting 16 kHz mono float32
(the sidecar front end's input spec). Each decoder is probed lazily and
cached; a decoder that fails to import or errors falls through to the
next. The ffmpeg path uses the binary present in most container images.

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


def _decode_with_miniaudio(data: bytes) -> tuple[np.ndarray, int] | None:
    """Decode via the miniaudio wheel (bundled dr_mp3/dr_wav, no system deps)."""
    try:
        import miniaudio
    except ImportError:
        return None
    try:
        decoded = miniaudio.decode(data, nchannels=1, sample_rate=cfg.SAMPLE_RATE)
        samples = np.asarray(decoded.samples, dtype=np.float32)
        if samples.dtype == np.int16:
            samples = samples / 32768.0
        return samples, cfg.SAMPLE_RATE
    except Exception as err:  # noqa: BLE001 — any decode failure falls through
        _LOGGER.debug("miniaudio decode failed: %s", err)
        return None


def _decode_with_soundfile(data: bytes) -> tuple[np.ndarray, int] | None:
    """Decode via soundfile (libsndfile) + scipy resample_poly."""
    try:
        import io

        import soundfile as sf
        from scipy.signal import resample_poly
    except ImportError:
        return None
    try:
        samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        mono = samples.mean(axis=1)
        if sr != cfg.SAMPLE_RATE:
            import math

            g = math.gcd(int(sr), cfg.SAMPLE_RATE)
            mono = resample_poly(
                mono, cfg.SAMPLE_RATE // g, int(sr) // g
            ).astype(np.float32)
        return mono.astype(np.float32), cfg.SAMPLE_RATE
    except Exception as err:  # noqa: BLE001 — any decode failure falls through
        _LOGGER.debug("soundfile decode failed: %s", err)
        return None


def _decode_with_ffmpeg(data: bytes) -> tuple[np.ndarray, int] | None:
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
                "-ar", str(cfg.SAMPLE_RATE),
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
        return samples, cfg.SAMPLE_RATE
    except Exception as err:  # noqa: BLE001 — any decode failure falls through
        _LOGGER.debug("ffmpeg decode failed: %s", err)
        return None
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def decode(data: bytes) -> tuple[np.ndarray, int] | None:
    """Decode audio bytes → (mono float32 samples, sample_rate).

    Returns None when no decoder can handle the input (caller → 422).
    """
    if not data:
        return None
    for decoder in (_decode_with_miniaudio, _decode_with_soundfile, _decode_with_ffmpeg):
        result = decoder(data)
        if result is not None and len(result[0]) > 0:
            return result
    return None