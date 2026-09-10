"""aubio tempo analysis.

Computes BPM as 60 / median(inter-beat interval) of detected beats.

Two modes, tried in order:
1. Python `aubio` package (run in the executor — never the event loop).
2. External `aubio` CLI binary (config option `aubio_binary`), via subprocess.

A missing analyzer must never delay or block the sensor: callers get None.
"""

from __future__ import annotations

import asyncio
import logging
import statistics

from .const import ANALYSIS_TIMEOUT

_LOGGER = logging.getLogger(__name__)


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


class AubioAnalyzer:
    """Tempo analyzer with package-first, CLI-fallback strategy."""

    def __init__(self, aubio_binary: str | None = None) -> None:
        self._binary = aubio_binary or None

    async def get_bpm(self, hass, path: str) -> float | None:
        """Analyze a preview file. Returns BPM or None. Never blocks the loop."""
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, self._analyze_sync, path),
                timeout=ANALYSIS_TIMEOUT,
            )
        except asyncio.TimeoutError:
            _LOGGER.debug("aubio analysis timed out for %s", path)
            return None

    def _analyze_sync(self, path: str) -> float | None:
        bpm = _analyze_with_package(path)
        if bpm:
            return bpm
        if self._binary:
            return _analyze_with_cli(self._binary, path)
        return None