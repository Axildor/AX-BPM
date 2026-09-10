"""Essentia SVM mood analysis.

Runs the five SVM mood classifiers (MoodAggressive, MoodParty,
MoodElectronic, MoodRelaxed, MoodAcoustic) — NOT the TensorFlow
arousal/valence models (out of scope, hundreds of MB).

Model files (Gaia .history format, CC BY-NC-ND licensed) are located in
the wheel's package data when present; otherwise downloaded once at first
setup into the integration's storage directory.

Subprocess mode is preferred (full isolation, no GIL risk, a crash can't
take down HA); in-process executor mode is the fallback.
"""

from __future__ import annotations

import asyncio
import logging
import os
import urllib.request

from .const import ANALYSIS_TIMEOUT, ESSENTIA_MODELS, ESSENTIA_MODEL_BASE_URL

_LOGGER = logging.getLogger(__name__)


def is_available() -> bool:
    """Probe whether the essentia Python package is importable."""
    try:
        import essentia  # noqa: F401, PLC0415
        return True
    except ImportError:
        return False


def _find_model_dir(storage_path: str) -> str | None:
    """Locate SVM model files: wheel package data first, then storage."""
    try:
        import essentia  # noqa: PLC0415

        pkg_dir = os.path.dirname(essentia.__file__)
        for candidate in (
            os.path.join(pkg_dir, "models", "svm-models", "mood"),
            os.path.join(pkg_dir, "svm_models"),
        ):
            if os.path.isdir(candidate):
                return candidate
    except ImportError:
        pass
    if storage_path and os.path.isdir(storage_path):
        return storage_path
    return None


def ensure_models(storage_path: str) -> bool:
    """Download missing SVM models once into storage_path. Returns success."""
    os.makedirs(storage_path, exist_ok=True)
    ok = True
    for name, filename in ESSENTIA_MODELS.items():
        dest = os.path.join(storage_path, filename)
        if os.path.exists(dest):
            continue
        url = f"{ESSENTIA_MODEL_BASE_URL}/{filename}"
        try:
            _LOGGER.info("Downloading Essentia SVM model %s", filename)
            urllib.request.urlretrieve(url, dest)  # noqa: S310
        except OSError as err:
            _LOGGER.error("Failed to download %s: %s", url, err)
            ok = False
    return ok


def analyze_mood_sync(path: str, model_dir: str) -> dict[str, float] | None:
    """Run MusicExtractor highlevel SVM mood classifiers on one file."""
    try:
        import essentia.standard as es  # noqa: PLC0415
    except ImportError:
        return None

    model_paths = [
        os.path.join(model_dir, filename)
        for filename in ESSENTIA_MODELS.values()
        if os.path.exists(os.path.join(model_dir, filename))
    ]
    if not model_paths:
        return None

    try:
        features, _frames = es.MusicExtractor(
            lowlevelStats=["mean", "stdev"],
            rhythmStats=["mean", "stdev"],
            tonalStats=["mean", "stdev"],
            svm_models=model_paths,
        )(path)
    except Exception as err:  # essentia raises on malformed input/models
        _LOGGER.debug("Essentia analysis failed for %s: %s", path, err)
        return None

    scores: dict[str, float] = {}
    for key in ESSENTIA_MODELS:
        value = features.get(f"highlevel.mood_{key}.probability")
        if value is not None:
            scores[key] = float(value)
    return scores or None


class MoodAnalyzer:
    """Essentia SVM mood analyzer (in-process executor mode)."""

    def __init__(self, storage_path: str) -> None:
        self._storage_path = storage_path
        self._model_dir: str | None = None
        self._models_ready = False

    async def async_setup(self) -> bool:
        """Probe availability; download models once if needed."""
        if not is_available():
            return False
        loop = asyncio.get_running_loop()
        try:
            self._model_dir = await loop.run_in_executor(
                None, _find_model_dir, self._storage_path
            )
            if not self._model_dir or not all(
                os.path.exists(os.path.join(self._model_dir, fn))
                for fn in ESSENTIA_MODELS.values()
            ):
                ok = await loop.run_in_executor(
                    None, ensure_models, self._storage_path
                )
                if ok:
                    self._model_dir = self._storage_path
            self._models_ready = bool(self._model_dir)
        except OSError as err:
            _LOGGER.error("Essentia model setup failed: %s", err)
            self._models_ready = False
        return self._models_ready

    @property
    def available(self) -> bool:
        return self._models_ready

    async def get_scores(self, path: str) -> dict[str, float] | None:
        """Analyze a preview file. Returns {mood: score} or None."""
        if not self._models_ready:
            return None
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    None, analyze_mood_sync, path, self._model_dir
                ),
                timeout=ANALYSIS_TIMEOUT,
            )
        except asyncio.TimeoutError:
            _LOGGER.debug("Essentia analysis timed out for %s", path)
            return None