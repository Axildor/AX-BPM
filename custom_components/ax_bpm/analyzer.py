"""Local tempo analysis — the NumPy floor.

Since the aubio tempo tier moved into the AX BPM Analyzer add-on
(tempo.py, served in the /analyze payload as `bpm`), the integration
ships ONLY the pure-NumPy estimator as its local analyzer. It works
out-of-the-box on installs where compiled wheels are unavailable
(HA OS / HA Container, musl, no compiler) — the analyzer add-on is the
accuracy upgrade when installed.

Decoding to PCM is handled by decode.py (miniaudio → soundfile → ffmpeg).

BPM convention: direct tempo output from tempo_numpy. A missing analyzer
must never delay or block the sensor: callers get None.
"""

from __future__ import annotations

import asyncio
import logging

from .const import ANALYSIS_TIMEOUT
from .decode import decode_available, decode_mono
from .tempo_numpy import estimate_bpm

_LOGGER = logging.getLogger(__name__)


def analyzer_available() -> bool:
    """True when local tempo analysis can run at all.

    With the NumPy floor, analysis is available whenever any decoder is
    available; the analyzer add-on (when installed) is the accuracy
    upgrade.
    """
    return decode_available()


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


class NumpyAnalyzer:
    """Local tempo analyzer — the guaranteed NumPy floor."""

    def __init__(self) -> None:
        self._availability_logged = False
        self.last_backend = "numpy"

    def _log_availability_once(self) -> None:
        """Log the analyzer strategy once, on first use."""
        if self._availability_logged:
            return
        self._availability_logged = True
        if decode_available():
            _LOGGER.info(
                "AX BPM: using built-in NumPy tempo estimator (install the "
                "AX BPM Analyzer add-on for aubio-grade accuracy + mood)"
            )
        else:
            _LOGGER.warning(
                "AX BPM: no decoder available — tracks where Deezer "
                "reports bpm: 0 will resolve to unknown"
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
        self.last_backend = "numpy"
        return _analyze_with_numpy(path)