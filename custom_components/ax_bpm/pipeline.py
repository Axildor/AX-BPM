"""Per-track BPM resolution pipeline.

Order: cache lookup → anonymous Deezer match (metadata bpm) → local
analysis (aubio tempo + sidecar mood) → octave disambiguation → publish.

Invariants:
- Single-flight per track (no duplicate lookups).
- Overall budget ~OVERALL_BUDGET seconds; on timeout publish from whatever
  stage completed, else unknown.
- Never publish 0: failure → None (sensor shows unknown).
- Mood correction only ever fires on the locally analyzed aubio estimate,
  never on Deezer metadata bpm.
- Graceful degradation: mood+genre → genre-only → raw.
- Mood-enabled cache miss downloads the 30 s preview ONCE (regardless of
  Deezer bpm) and reuses the same buffer for the local BPM fallback AND
  the sidecar call; discarded after.
- Local path (bpm == 0): the sidecar call runs CONCURRENTLY with aubio,
  bounded by min(remaining budget, MOOD_TIMEOUT), so the first publish is
  already mood-corrected (user-approved deviation from the parent plan's
  "mood never delays the publish" invariant — no BPM flip-flop).
- Deezer path (bpm > 0): publish immediately; mood enrichment is a
  separate post-publish step (async_enrich) the sensor triggers — mood
  never delays this publish (and cannot change the BPM anyway).
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from collections.abc import Callable
from typing import Any

from . import math as bpm_math
from .analyzer import AubioAnalyzer
from .const import (
    ANALYSIS_TIMEOUT,
    CONF_MOOD_ANALYZER_URL,
    CONF_MOOD_API_TOKEN,
    CONF_OCTAVE_DISAMBIGUATION,
    EXPECTED_MOOD_SCORES,
    OCTAVE_GENRE_MOOD,
    OCTAVE_OFF,
    OVERALL_BUDGET,
    SOURCE_AUBIO,
    SOURCE_DEEZER,
    SOURCE_NUMPY,
)
from .deezer import DeezerClient, parse_artist_title
from .mood_client import MoodClient
from .store import BpmCache, cache_key

_LOGGER = logging.getLogger(__name__)

# Legacy cache fields dropped by the v1 → v2 migration (Essentia SVM era).
LEGACY_MOOD_FIELDS = ("mood_scores", "mood_label")


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
        self._mood = MoodClient(
            session,
            options.get(CONF_MOOD_ANALYZER_URL),
            options.get(CONF_MOOD_API_TOKEN),
        )
        self._octave_mode = options.get(CONF_OCTAVE_DISAMBIGUATION, "genre_only")
        self._mood_ready = False
        self._lock = asyncio.Lock()
        self._current_token: str | None = None

    async def async_setup(self) -> None:
        """Probe sidecar availability (never blocks the sensor)."""
        if self._octave_mode == OCTAVE_GENRE_MOOD:
            self._mood_ready = await self._mood.async_detect() is not None
            if not self._mood_ready:
                _LOGGER.info(
                    "Sidecar mood analyzer not detected — falling back to "
                    "genre-only octave disambiguation"
                )

    @property
    def mood_client(self) -> MoodClient:
        """Expose the mood client (sensor post-publish enrichment)."""
        return self._mood

    @property
    def mood_enabled(self) -> bool:
        """True when the dropdown selects Genre + mood."""
        return self._octave_mode == OCTAVE_GENRE_MOOD

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

    async def async_enrich(
        self,
        artist: str,
        title: str,
        duration: float | None,
        result: ResolutionResult,
    ) -> dict[str, Any] | None:
        """Post-publish mood enrichment (Deezer-metadata path only).

        Downloads the preview once, POSTs it to the sidecar, caches the
        mood payload per-ISRC alongside the BPM, and returns the mood
        attribute dict for the sensor's second state write. Returns None
        on any failure — never raises, never retries.
        """
        if not self.mood_enabled or not self._mood_ready:
            return None
        if result.source != SOURCE_DEEZER:
            return None  # local path already carries mood attrs

        try:
            preview = await self._deezer.download_preview_bytes(
                artist, title, duration
            )
        except Exception:
            _LOGGER.debug("Mood enrichment preview download failed", exc_info=True)
            return None
        if not preview:
            return None

        payload = await self._mood.async_analyze(preview)
        if not payload:
            return None

        mood_attrs = self._mood_attrs_from_payload(payload)
        # Cache mood alongside the BPM (per-ISRC when known).
        key = cache_key(result.attrs.get("isrc"), artist, title, duration)
        cached = self._cache.get(key) or {}
        await self._cache.async_put(
            key, {**cached, "bpm": result.bpm, **result.attrs, **mood_attrs}
        )
        return mood_attrs

    @staticmethod
    def _scores_from_tags(payload: dict[str, Any] | None) -> dict[str, float] | None:
        """Derive the five-signal mapping from sidecar mood tags."""
        if not payload:
            return None
        tags = payload.get("mood_tags") or []
        scores = {
            t.get("tag"): float(t.get("score", 0.0))
            for t in tags
            if isinstance(t, dict) and t.get("tag")
        }
        return scores or None

    @classmethod
    def _mood_attrs_from_payload(cls, payload: dict[str, Any]) -> dict[str, Any]:
        """Map a sidecar /analyze payload to sensor mood attributes.

        mood_scores is ATOMIC (Catch 1): consumed only when present AND
        complete over EXPECTED_MOOD_SCORES. A missing key read as 0.0
        would mean "maximally non-X" and bias octave gating toward
        intensity during partial failure — incomplete → genre-only
        fallback (the designed degradation path), never 0.0 substitution.
        """
        attrs: dict[str, Any] = {
            "mood_tags": payload.get("mood_tags") or [],
            "valence": payload.get("valence"),
            "arousal": payload.get("arousal"),
            "valence_std": payload.get("valence_std"),
            "arousal_std": payload.get("arousal_std"),
            "danceability": payload.get("danceability"),
            "analyzed_seconds": payload.get("analyzed_seconds"),
            "model_versions": payload.get("model_versions"),
            "source": "sidecar",
        }
        # Explicit five-signal dict from the dedicated mood heads — the
        # sidecar emits it only when ALL five heads are healthy.
        scores = payload.get("mood_scores")
        if not (
            isinstance(scores, dict)
            and EXPECTED_MOOD_SCORES.issubset(scores)
        ):
            scores = None
        if scores:
            intensity = (
                float(scores["aggressive"])
                + float(scores["party"])
                + float(scores["electronic"])
            ) / 3.0
            calmness = (
                float(scores["relaxed"]) + float(scores["acoustic"])
            ) / 2.0
            arousal = payload.get("arousal")
            if arousal is not None and abs(intensity - calmness) < 0.05:
                intensity = max(intensity, float(arousal))
                calmness = min(calmness, 1.0 - float(arousal))
            attrs["intensity"] = round(intensity, 3)
            attrs["calmness"] = round(calmness, 3)
            attrs["mood_scores"] = scores
        return attrs

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
            _LOGGER.debug(
                "AX BPM cache hit for %s - %s → %s BPM",
                artist, title, cached["bpm"],
            )
            return ResolutionResult(cached["bpm"], **{
                "source": "cache",
                **{k: v for k, v in cached.items() if k != "bpm"},
            })

        # 2. Anonymous Deezer match. Returns (result, pending_match) where
        # pending_match is the bpm==0 match metadata for the local path —
        # strictly per-resolution, never stored on the instance.
        try:
            result, match = await asyncio.wait_for(
                self._resolve_deezer(artist, title, duration, key),
                timeout=max(0.1, deadline - time.monotonic()),
            )
        except asyncio.TimeoutError:
            result, match = None, None
        if result is not None:
            _LOGGER.info(
                "AX BPM: %s - %s → %.1f BPM (source: %s)",
                artist, title, result.bpm, result.source,
            )
            return result
        if time.monotonic() > deadline or self._current_token != token:
            _LOGGER.info(
                "AX BPM: %s - %s → unresolved (budget exceeded or track "
                "changed before Deezer match)", artist, title,
            )
            return None

        # 3. Local analysis fallback (Deezer bpm == 0 or no confident match).
        # Only runs when THIS resolution produced a bpm==0 match; a failed
        # Deezer lookup must never reuse a previous track's match data.
        try:
            result = await asyncio.wait_for(
                self._resolve_local(
                    artist, title, duration, key, deadline, match
                ),
                timeout=max(0.1, deadline - time.monotonic()),
            )
        except asyncio.TimeoutError:
            _LOGGER.debug("Overall budget exceeded for %s - %s", artist, title)
            return None
        if result is None:
            _LOGGER.info(
                "AX BPM: %s - %s → unresolved (no Deezer match with bpm, "
                "and local analysis unavailable or failed)", artist, title,
            )
        else:
            _LOGGER.info(
                "AX BPM: %s - %s → %.1f BPM (source: %s)",
                artist, title, result.bpm, result.source,
            )
        return result

    async def _resolve_deezer(
        self,
        artist: str,
        title: str,
        duration: float | None,
        key: str,
    ) -> tuple[ResolutionResult | None, dict | None]:
        """Resolve via Deezer metadata.

        Returns (result, pending_match): result is the published resolution
        when Deezer reports a bpm; pending_match carries isrc/genres/preview
        for the local-analysis path when bpm == 0. Both are None on any
        failure — callers must never fall back to stale match data.
        """
        match = await self._deezer.find_match(artist, title, duration)
        if not match or not match.get("id"):
            return None, None

        track = await self._deezer.get_track(match["id"])
        if not track:
            return None, None

        deezer_bpm = track.get("bpm") or 0.0
        isrc = track.get("isrc")
        album_id = (track.get("album") or {}).get("id")
        genres = (
            await self._deezer.get_album_genres(album_id) if album_id else []
        )

        _LOGGER.debug(
            "Deezer track %s: bpm=%s, isrc=%s, preview=%s",
            match["id"], deezer_bpm, isrc, bool(track.get("preview")),
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
            return result, None

        # bpm == 0 → hand isrc/genres/preview to the local path for THIS
        # resolution only. Never stored on the instance: a later failed
        # lookup must not inherit a previous track's match data.
        pending = {
            "isrc": isrc,
            "deezer_track_id": match["id"],
            "match_rank": match.get("rank"),
            "genres": genres,
            "preview_url": track.get("preview"),
        }
        return None, pending

    async def _resolve_local(
        self,
        artist: str,
        title: str,
        duration: float | None,
        key: str,
        deadline: float,
        match: dict | None,
    ) -> ResolutionResult | None:
        match = match or {}
        preview_url = match.get("preview_url")
        if not preview_url:
            return None

        # Download the preview once — use immediately, never cache the URL.
        # mkstemp does filesystem I/O — keep it off the event loop.
        loop = asyncio.get_running_loop()
        fd, tmp_path = await loop.run_in_executor(
            None,
            lambda: tempfile.mkstemp(suffix=".mp3", prefix="ax_bpm_"),
        )
        os.close(fd)
        try:
            if not await self._deezer.download_preview(preview_url, tmp_path):
                return None

            # Run the tempo analyzer and (when enabled) the sidecar mood
            # call CONCURRENTLY on the one preview. The sidecar timeout is
            # bounded by the remaining budget so the first publish is
            # already mood-corrected; a slow sidecar degrades to genre-only.
            bpm_task = asyncio.ensure_future(self._aubio.get_bpm(self._hass, tmp_path))
            mood_task = None
            if self.mood_enabled and self._mood_ready:
                mood_task = asyncio.ensure_future(
                    self._mood.async_analyze_file(tmp_path)
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

            mood_payload = (
                mood_task.result()
                if mood_task in done and not mood_task.cancelled()
                else None
            )
            mood_scores = (
                mood_payload.get("mood_scores") if mood_payload else None
            ) or self._scores_from_tags(mood_payload)

            genres = match.get("genres") or []
            if self._octave_mode == OCTAVE_OFF:
                genres = []

            # Octave disambiguation — only on the local tempo estimate.
            disambiguated = bpm_math.apply_octave_disambiguation(
                bpm_raw, mood_scores, genres
            )

            mood_label = (
                max(mood_scores, key=mood_scores.get) if mood_scores else None
            )
            backend = getattr(self._aubio, "last_backend", None) or "aubio"
            source = {
                "aubio": SOURCE_AUBIO,
                "aubio_cli": SOURCE_AUBIO,
                "numpy": SOURCE_NUMPY,
            }.get(backend, SOURCE_AUBIO)
            result = ResolutionResult(
                disambiguated["bpm"],
                source=source,
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