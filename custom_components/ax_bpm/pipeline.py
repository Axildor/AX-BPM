"""Per-track BPM resolution pipeline.

Order: cache lookup → anonymous Deezer match (metadata bpm) → local
analysis (aubio tempo + Essentia mood) → octave disambiguation → publish.

Invariants:
- Single-flight per track (no duplicate lookups).
- Overall budget ~OVERALL_BUDGET seconds; on timeout publish from whatever
  stage completed, else unknown.
- Never publish 0: failure → None (sensor shows unknown).
- Mood correction only ever fires on the locally analyzed aubio estimate,
  never on Deezer metadata bpm.
- Graceful degradation: mood+genre → genre-only → raw.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from typing import Any, Callable

from . import math as bpm_math
from .analyzer import AubioAnalyzer
from .const import (
    ANALYSIS_TIMEOUT,
    CONF_GENRE_CORRECTION,
    CONF_MOOD_CORRECTION,
    OVERALL_BUDGET,
    SOURCE_AUBIO,
    SOURCE_DEEZER,
)
from .deezer import DeezerClient, parse_artist_title
from .mood import MoodAnalyzer
from .store import BpmCache, cache_key

_LOGGER = logging.getLogger(__name__)


class ResolutionResult:
    """Everything the sensor needs to publish one track's BPM."""

    def __init__(self, bpm: float, source: str, **attrs: Any) -> None:
        self.bpm = bpm
        self.source = source
        self.attrs = attrs


class BpmPipeline:
    """Orchestrates cache → Deezer → local analysis for track changes."""

    def __init__(
        self,
        hass,
        session,
        cache: BpmCache,
        options: dict[str, Any],
    ) -> None:
        self._hass = hass
        self._deezer = DeezerClient(session)
        self._cache = cache
        self._aubio = AubioAnalyzer(options.get("aubio_binary"))
        self._mood = MoodAnalyzer(
            os.path.join(hass.config.path("storage"), "ax_bpm", "models")
        )
        self._genre_correction = options.get(CONF_GENRE_CORRECTION, True)
        self._mood_correction = options.get(CONF_MOOD_CORRECTION, True)
        self._mood_ready = False
        self._lock = asyncio.Lock()
        self._current_token: str | None = None

    async def async_setup(self) -> None:
        """Probe Essentia availability (never blocks the sensor)."""
        if self._mood_correction:
            self._mood_ready = await self._mood.async_setup()
            if not self._mood_ready:
                _LOGGER.warning(
                    "Essentia unavailable — falling back to genre-only "
                    "octave disambiguation"
                )

    async def async_resolve(
        self,
        artist: str | None,
        title: str | None,
        duration: float | None,
    ) -> ResolutionResult | None:
        """Resolve BPM for one track. Returns None on failure (→ unknown)."""
        parsed = parse_artist_title(artist, title)
        if not parsed:
            return None
        track_artist, track_title = parsed
        token = f"{track_artist}|{track_title}|{duration}"

        # Single-flight: if a resolution for this track is already running,
        # wait for it and reuse its outcome via the cache.
        async with self._lock:
            self._current_token = token
            return await self._resolve_inner(
                track_artist, track_title, duration, token
            )

    async def _resolve_inner(
        self,
        artist: str,
        title: str,
        duration: float | None,
        token: str,
    ) -> ResolutionResult | None:
        deadline = time.monotonic() + OVERALL_BUDGET

        # 1. Cache lookup — publish immediately, no network, no analysis.
        key = cache_key(None, artist, title, duration)
        cached = self._cache.get(key)
        if cached and cached.get("bpm"):
            return ResolutionResult(cached["bpm"], **{
                "source": "cache",
                **{k: v for k, v in cached.items() if k != "bpm"},
            })

        # 2. Anonymous Deezer match.
        try:
            result = await asyncio.wait_for(
                self._resolve_deezer(artist, title, duration, key),
                timeout=max(0.1, deadline - time.monotonic()),
            )
        except asyncio.TimeoutError:
            result = None
        if result is not None:
            return result
        if time.monotonic() > deadline or self._current_token != token:
            return None

        # 3. Local analysis fallback (Deezer bpm == 0 or no confident match).
        try:
            return await asyncio.wait_for(
                self._resolve_local(artist, title, duration, key, deadline),
                timeout=max(0.1, deadline - time.monotonic()),
            )
        except asyncio.TimeoutError:
            _LOGGER.debug("Overall budget exceeded for %s - %s", artist, title)
            return None

    async def _resolve_deezer(
        self,
        artist: str,
        title: str,
        duration: float | None,
        key: str,
    ) -> ResolutionResult | None:
        match = await self._deezer.find_match(artist, title, duration)
        if not match or not match.get("id"):
            return None

        track = await self._deezer.get_track(match["id"])
        if not track:
            return None

        deezer_bpm = track.get("bpm") or 0.0
        isrc = track.get("isrc")
        album_id = (track.get("album") or {}).get("id")
        genres = (
            await self._deezer.get_album_genres(album_id) if album_id else []
        )

        # Deezer metadata bpm present → publish as-is, NEVER corrected.
        if deezer_bpm > 0:
            result = ResolutionResult(
                float(deezer_bpm),
                source=SOURCE_DEEZER,
                track=f"{artist} - {title}",
                isrc=isrc,
                deezer_track_id=match["id"],
                match_rank=match.get("rank"),
                bpm_raw=float(deezer_bpm),
                genre=", ".join(genres) if genres else None,
                genre_source="deezer_album" if genres else None,
                mood_scores=None,
                mood_label=None,
                intensity=None,
                calmness=None,
                octave_corrected=False,
                octave_rule=bpm_math.RULE_NONE,
                preview_analyzed=False,
            )
            await self._cache.async_put(
                cache_key(isrc, artist, title, duration),
                {"bpm": result.bpm, **result.attrs},
            )
            return result

        # bpm == 0 → remember isrc/genres for the local path via cache entry
        # metadata, but do not publish anything yet.
        self._last_match = {
            "isrc": isrc,
            "deezer_track_id": match["id"],
            "match_rank": match.get("rank"),
            "genres": genres,
            "preview_url": track.get("preview"),
        }
        return None

    async def _resolve_local(
        self,
        artist: str,
        title: str,
        duration: float | None,
        key: str,
        deadline: float,
    ) -> ResolutionResult | None:
        match = getattr(self, "_last_match", None) or {}
        preview_url = match.get("preview_url")
        if not preview_url:
            return None

        # Download the preview once — use immediately, never cache the URL.
        fd, tmp_path = tempfile.mkstemp(suffix=".mp3", prefix="ax_bpm_")
        os.close(fd)
        try:
            if not await self._deezer.download_preview(preview_url, tmp_path):
                return None

            # Run BOTH analyzers concurrently on the one preview.
            bpm_task = asyncio.ensure_future(self._aubio.get_bpm(self._hass, tmp_path))
            mood_task = None
            if self._mood_correction and self._mood_ready:
                mood_task = asyncio.ensure_future(
                    self._mood.get_scores(tmp_path)
                )
            remaining = max(0.1, deadline - time.monotonic())
            done, pending = await asyncio.wait(
                {t for t in (bpm_task, mood_task) if t},
                timeout=min(remaining, ANALYSIS_TIMEOUT * 2),
            )
            for task in pending:
                task.cancel()

            bpm_raw = bpm_task.result() if bpm_task in done else None
            if not bpm_raw or bpm_raw <= 0:
                return None

            mood_scores = (
                mood_task.result() if mood_task in done and not mood_task.cancelled() else None
            )
            if not self._mood_correction:
                mood_scores = None

            genres = match.get("genres") or []
            if not self._genre_correction:
                genres = []

            # Octave disambiguation — only on the local aubio estimate.
            disambiguated = bpm_math.apply_octave_disambiguation(
                bpm_raw, mood_scores, genres
            )

            mood_label = (
                max(mood_scores, key=mood_scores.get) if mood_scores else None
            )
            result = ResolutionResult(
                disambiguated["bpm"],
                source=SOURCE_AUBIO,
                track=f"{artist} - {title}",
                isrc=match.get("isrc"),
                deezer_track_id=match.get("deezer_track_id"),
                match_rank=match.get("match_rank"),
                bpm_raw=bpm_raw,
                genre=", ".join(genres) if genres else None,
                genre_source="deezer_album" if genres else None,
                mood_scores=mood_scores,
                mood_label=mood_label,
                intensity=disambiguated["intensity"],
                calmness=disambiguated["calmness"],
                octave_corrected=disambiguated["corrected"],
                octave_rule=disambiguated["rule"],
                preview_analyzed=True,
            )
            await self._cache.async_put(
                cache_key(match.get("isrc"), artist, title, duration),
                {"bpm": result.bpm, **result.attrs},
            )
            return result
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def make_result_from_cache(cached: dict[str, Any]) -> ResolutionResult:
    """Rebuild a ResolutionResult from a cache entry."""
    bpm = cached.pop("bpm")
    cached.pop("source", None)
    return ResolutionResult(bpm, source="cache", **cached)


# Convenience re-export for the sensor.
publish_callback = Callable[[ResolutionResult | None], None]