"""MP3 decode fallback chain: miniaudio → soundfile → ffmpeg binary.

Returns mono float32 samples at a fixed sample rate, ready for tempo_numpy.
Each decoder is probed lazily and cached; a decoder that fails to import or
errors on a file simply falls through to the next one. The ffmpeg path uses
the binary present in official HA images (and most systems), so decoding is
effectively always available.

Executor only — never call from the event loop.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile

import numpy as np

from .const import DECODE_SAMPLE_RATE

_LOGGER = logging.getLogger(__name__)

_FFMPEG_TIMEOUT = 30.0


def _decode_with_miniaudio(path: str) -> tuple[np.ndarray, int] | None:
    """Decode via the miniaudio wheel (if installed)."""
    try:
        import miniaudio  # noqa: PLC0415
    except ImportError:
        return None
    try:
        decoded = miniaudio.decode_file(path, nchannels=1, sample_rate=DECODE_SAMPLE_RATE)
        samples = np.asarray(decoded.samples, dtype=np.float32)
        if samples.dtype == np.int16:
            samples = samples / 32768.0
        elif samples.max(initial=0) > 1.5:  # int8-ish or int32 payload
            samples = samples / 32768.0
        return samples, DECODE_SAMPLE_RATE
    except Exception as err:  # noqa: BLE001 — any decode failure falls through
        _LOGGER.debug("miniaudio decode failed for %s: %s", path, err)
        return None


def _decode_with_soundfile(path: str) -> tuple[np.ndarray, int] | None:
    """Decode via the soundfile wheel (libsndfile bundled, MP3-capable)."""
    try:
        import soundfile as sf  # noqa: PLC0415
    except ImportError:
        return None
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1).astype(np.float32)
        if sr != DECODE_SAMPLE_RATE:
            mono = _resample_linear(mono, sr, DECODE_SAMPLE_RATE)
            sr = DECODE_SAMPLE_RATE
        return mono, sr
    except Exception as err:  # noqa: BLE001
        _LOGGER.debug("soundfile decode failed for %s: %s", path, err)
        return None


def _decode_with_ffmpeg(path: str) -> tuple[np.ndarray, int] | None:
    """Decode via the ffmpeg binary (present in official HA images)."""
    binary = shutil.which("ffmpeg")
    if not binary:
        return None
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as raw:
        raw_path = raw.name
    try:
        proc = subprocess.run(
            [
                binary, "-v", "error", "-i", path,
                "-f", "f32le", "-acodec", "pcm_f32le",
                "-ac", "1", "-ar", str(DECODE_SAMPLE_RATE),
                "-y", raw_path,
            ],
            capture_output=True,
            timeout=_FFMPEG_TIMEOUT,
            check=False,
        )
        if proc.returncode != 0:
            _LOGGER.debug("ffmpeg decode failed for %s: %s", path, proc.stderr[-300:])
            return None
        samples = np.fromfile(raw_path, dtype=np.float32)
        if samples.size == 0:
            return None
        return samples, DECODE_SAMPLE_RATE
    except (OSError, subprocess.SubprocessError) as err:
        _LOGGER.debug("ffmpeg decode error for %s: %s", path, err)
        return None
    finally:
        try:
            import os  # noqa: PLC0415

            os.unlink(raw_path)
        except OSError:
            pass


def _resample_linear(samples: np.ndarray, target_sr: int) -> np.ndarray:
    """Cheap linear-interpolation resample (good enough for tempo analysis)."""
    if samples.size < 2:
        return samples
    n_target = int(round(len(samples) * target_sr / DECODE_SAMPLE_RATE))
    if n_target < 2:
        return samples
    src_idx = np.linspace(0.0, len(samples) - 1, num=n_target)
    return np.interp(src_idx, np.arange(len(samples)), samples).astype(np.float32)


_DECODERS = (_decode_with_miniaudio, _decode_with_soundfile, _decode_with_ffmpeg)


def decode_available() -> bool:
    """True when at least one decoder can plausibly run right now."""
    try:
        import miniaudio  # noqa: F401, PLC0415

        return True
    except ImportError:
        pass
    try:
        import soundfile  # noqa: F401, PLC0415

        return True
    except ImportError:
        pass
    return shutil.which("ffmpeg") is not None


def decode_mono(path: str) -> tuple[np.ndarray, int] | None:
    """Decode an audio file to (mono float32 samples, sample_rate).

    Tries each decoder in order; returns None when all fail.
    """
    for decoder in _DECODERS:
        result = decoder(path)
        if result is not None and result[0].size > 0:
            return result
    return None