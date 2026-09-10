"""Local tempo analysis — pluggable analyzer chain.

Chain (first success wins):
1. aubio Python package (executor) — best classic accuracy
2. External `aubio` CLI binary (config option `aubio_binary`)
3. Essentia RhythmExtractor2013 (if the essentia wheel is installed)
4. Pure-Python NumPy estimator (tempo_numpy) — always available floor

Decoding to PCM is handled by decode.py (miniaudio → soundfile → ffmpeg);
aubio and Essentia decode files themselves, so the decode chain only backs
the NumPy floor.

BPM convention: 60 / median(inter-beat interval) for beat-based analyzers;
direct tempo output for Essentia/NumPy. A missing analyzer must never delay
or block the sensor: callers get None.
"""

from __future__ import annotations

import asyncio
import logging
import statistics

from .const import ANALYSIS_TIMEOUT
from .decode import decode_available, decode_mono
from .tempo_numpy import estimate_bpm

_LOGGER = logging.getLogger(__name__)


def aubio_available(binary: str | None = None) -> bool:
    """True when local tempo analysis can run at all.

    With the NumPy floor, analysis is available whenever any decoder is
    available; aubio/Essentia are optional accuracy upgrades.
    """
    if decode_available():
        return True
    try:
        import aubio  # noqa: F401, PLC0415

        return True
    except ImportError:
        return bool(binary)


def _analyze_with_package(path: str) -> float | None:
    """In-process aubio tempo detection (executor only)."""
    try:
        import aubio  # noqa: PLC0415
    except ImportError:
        return None

    win_s = 1024
    hop_s = 512
    samplerate = 44100

    try:
        src = aubio.source(path, samplerate, hop_s)
        samplerate = src.samplerate
        tempo_detector = aubio.tempo("specdiff", win_s, hop_s, samplerate)

        beats: list[float] = []
        total_frames = 0
        while True:
            samples, read = src()
            is_beat = tempo_detector(samples)
            if is_beat:
                beats.append(tempo_detector.get_last_s())
            total_frames += read
            if read < hop_s:
                break
    except Exception as err:  # aubio raises on malformed files
        _LOGGER.debug("aubio package analysis failed for %s: %s", path, err)
        return None

    if len(beats) < 2:
        return None
    intervals = [b - a for a, b in zip(beats, beats[1:])]
    median_ibi = statistics.median(intervals)
    if median_ibi <= 0:
        return None
    return 60.0 / median_ibi


def _analyze_with_cli(binary: str, path: str) -> float | None:
    """External `aubio tempo` CLI: parse beat timestamps from stdout."""
    try:
        import subprocess  # noqa: PLC0415

        proc = subprocess.run(
            [binary, "beat", path],
            capture_output=True,
            text=True,
            timeout=ANALYSIS_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as err:
        _LOGGER.debug("aubio CLI failed (%s): %s", binary, err)
        return None

    if proc.returncode != 0:
        return None

    beats: list[float] = []
    for line in proc.stdout.splitlines():
        try:
            beats.append(float(line.strip()))
        except ValueError:
            continue
    if len(beats) < 2:
        return None
    intervals = [b - a for a, b in zip(beats, beats[1:])]
    median_ibi = statistics.median(intervals)
    if median_ibi <= 0:
        return None
    return 60.0 / median_ibi


def _analyze_with_essentia(path: str) -> float | None:
    """Essentia RhythmExtractor2013 tempo (executor only)."""
    try:
        import essentia.standard as es  # noqa: PLC0415
    except ImportError:
        return None
    try:
        audio = es.MonoLoader(filename=path, sampleRate=22050)()
        if audio.size == 0:
            return None
        extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, _beats, _confidence, _ext, _intervals = extractor(audio)
        bpm = float(bpm)
        if bpm <= 0:
            return None
        return bpm
    except Exception as err:  # noqa: BLE001 — any failure falls through
        _LOGGER.debug("essentia tempo failed for %s: %s", path, err)
        return None


def _analyze_with_numpy(path: str) -> float | None:
    """NumPy floor: decode via decode.py, then tempo_numpy.estimate_bpm."""
    decoded = decode_mono(path)
    if decoded is None:
        return None
    samples, sample_rate = decoded
    result = estimate_bpm(samples, sample_rate)
    if result is None:
        return None
    bpm, confidence = result
    _LOGGER.debug(
        "numpy tempo for %s: %.1f BPM (confidence %.2f)", path, bpm, confidence
    )
    return bpm


class AubioAnalyzer:
    """Tempo analyzer with a pluggable chain: aubio → CLI → essentia → numpy."""

    def __init__(self, aubio_binary: str | None = None) -> None:
        self._binary = aubio_binary or None
        self._availability_logged = False

    def _log_availability_once(self) -> None:
        """Log the analyzer strategy once, on first use."""
        if self._availability_logged:
            return
        self._availability_logged = True
        try:
            import aubio  # noqa: F401, PLC0415

            _LOGGER.info("AX BPM: aubio package available (local tempo analysis active)")
            return
        except ImportError:
            pass
        try:
            import essentia  # noqa: F401, PLC0415

            _LOGGER.info("AX BPM: essentia available (local tempo analysis active)")
            return
        except ImportError:
            pass
        if decode_available():
            _LOGGER.info(
                "AX BPM: using built-in NumPy tempo estimator (install the "
                "'aubio' or 'essentia' package for higher accuracy)"
            )
        elif self._binary:
            _LOGGER.info(
                "AX BPM: no decoder — using external aubio binary %s", self._binary
            )
        else:
            _LOGGER.warning(
                "AX BPM: no tempo analyzer or decoder available — tracks "
                "where Deezer reports bpm: 0 will resolve to unknown"
            )

    async def get_bpm(self, hass, path: str) -> float | None:
        """Analyze a preview file. Returns BPM or None. Never blocks the loop."""
        self._log_availability_once()
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, self._analyze_sync, path),
                timeout=ANALYSIS_TIMEOUT,
            )
        except asyncio.TimeoutError:
            _LOGGER.debug("tempo analysis timed out for %s", path)
            return None

    def _analyze_sync(self, path: str) -> float | None:
        self.last_backend = None
        for name, analyzer in (
            ("aubio", _analyze_with_package),
            ("aubio_cli", lambda p: _analyze_with_cli(self._binary, p) if self._binary else None),
            ("essentia", _analyze_with_essentia),
            ("numpy", _analyze_with_numpy),
        ):
            bpm = analyzer(path)
            if bpm:
                self.last_backend = name
                return bpm
        return None