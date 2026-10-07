"""Per-track BPM resolution pipeline.

Order: cache lookup → anonymous Deezer match (metadata bpm) → local
analysis (analyzer tempo+mood when healthy → NumPy floor) → octave
disambiguation → publish.

Invariants:
- Single-flight per track (no duplicate lookups).
- Overall budget ~OVERALL_BUDGET seconds; on timeout publish from whatever
  stage completed, else unknown.
- Never publish 0: failure → None (sensor shows unknown).
- Mood correction only ever fires on the locally analyzed estimate
  (analyzer aubio or NumPy floor), never on Deezer metadata bpm.
- Graceful degradation: analyzer(bpm+mood) → NumPy floor + genre-only → raw.
- Mood-enabled cache miss downloads the 30 s preview ONCE (regardless of
  Deezer bpm) and reuses the same buffer for the local BPM fallback AND
  the analyzer call; discarded after.
- Local path (bpm == 0): when the analyzer is healthy, ONE /analyze call
  provides BOTH aubio bpm and mood_scores (decode-once design in the
  analyzer); when it is not, the NumPy floor runs locally and the
  analyzer mood call (if any) runs concurrently, bounded by min(remaining
  budget, ANALYZER_TIMEOUT).
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
from .analyzer import NumpyAnalyzer
from .analyzer_client import AnalyzerClient
from .const import (
    ANALYSIS_TIMEOUT,
    CONF_ANALYZER_API_TOKEN,
    CONF_ANALYZER_URL,
    CONF_DISCOVERED_ANALYZER_URL,
    CONF_OCTAVE_DISAMBIGUATION,
    EXPECTED_MOOD_SCORES,
    OCTAVE_GENRE_MOOD,
    OCTAVE_OFF,
    OVERALL_BUDGET,
    RULE_MANUAL_DOUBLE,
    RULE_MANUAL_HALF,
    SOURCE_ANALYZER,
    SOURCE_DEEZER,
    SOURCE_NUMPY,
    SOURCE_OVERRIDE,
)
from .deezer import DeezerClient, parse_artist_title
from .itunes import ItunesClient, preview_suffix
from .overrides import BpmOverrideStore
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
        overrides: BpmOverrideStore | None = None,
    ) -> None:
        self._hass = hass
        self._deezer = DeezerClient(session)
        self._itunes = ItunesClient(session)
        self._cache = cache
        self._overrides = overrides
        self._analyzer = NumpyAnalyzer()
        self._analyzer_client = AnalyzerClient(
            session,
            options.get(CONF_ANALYZER_URL),
            options.get(CONF_ANALYZER_API_TOKEN),
            options.get(CONF_DISCOVERED_ANALYZER_URL),
        )
        self._octave_mode = options.get(CONF_OCTAVE_DISAMBIGUATION, "genre_only")
        self._mood_ready = False
        self._lock = asyncio.Lock()
        self._current_token: str | None = None

    async def async_setup(self) -> None:
        """Probe analyzer availability (never blocks the sensor)."""
        if self._octave_mode == OCTAVE_GENRE_MOOD:
            self._mood_ready = (
                await self._analyzer_client.async_detect() is not None
            )
            if not self._mood_ready:
                _LOGGER.info(
                    "Analyzer not detected — falling back to genre-only "
                    "octave disambiguation"
                )

    @property
    def analyzer_client(self) -> AnalyzerClient:
        """Expose the analyzer client (sensor post-publish enrichment)."""
        return self._analyzer_client

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

    @property
    def overrides(self) -> BpmOverrideStore | None:
        """The manual override store (None when not wired up)."""
        return self._overrides

    async def async_apply_correction(
        self,
        artist: str,
        title: str,
        current_bpm: float,
        factor: float,
    ) -> float:
        """Store a manual halve/double correction for this track.

        factor: 0.5 (halve) or 2.0 (double). Returns the corrected BPM.
        The correction is applied to the CURRENTLY PUBLISHED bpm (the
        value the user sees), not to any underlying raw estimate.
        """
        if factor not in (0.5, 2.0):
            raise ValueError(f"correction factor must be 0.5 or 2.0, got {factor}")
        corrected = current_bpm * factor
        rule = RULE_MANUAL_HALF if factor == 0.5 else RULE_MANUAL_DOUBLE
        if self._overrides is None:
            raise RuntimeError("override store not configured")
        await self._overrides.async_put(
            artist, title, corrected, rule, current_bpm
        )
        return corrected

    async def async_clear_override(self, artist: str, title: str) -> bool:
        """Remove the manual override for this track. True when one existed."""
        if self._overrides is None:
            return False
        return await self._overrides.async_remove(artist, title)

    async def async_clear_cache(self) -> int:
        """Wipe the BPM cache (overrides untouched). Returns entries removed."""
        return await self._cache.async_clear()

    async def async_clear_overrides(self) -> int:
        """Wipe ALL manual overrides. Returns entries removed."""
        if self._overrides is None:
            return 0
        return await self._overrides.async_clear()

    async def async_enrich(
        self,
        artist: str,
        title: str,
        duration: float | None,
        result: ResolutionResult,
    ) -> dict[str, Any] | None:
        """Post-publish mood enrichment (Deezer-metadata path only).

        Downloads the preview once, POSTs it to the analyzer, caches the
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

        payload = await self._analyzer_client.async_analyze(preview)
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
        """Derive the five-signal mapping from analyzer mood tags."""
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
        """Map an analyzer /analyze payload to sensor mood attributes.

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
            "source": "analyzer",
        }
        # Explicit five-signal dict from the dedicated mood heads — the
        # analyzer emits it only when ALL five heads are healthy.
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
            # mood_label (defect 3): argmax ONLY over the COMPLETE five-head
            # set (atomic rule). Deterministic tie-break via sorted order.
            # This is the single derivation — the local path merges this
            # function's output instead of hand-rolling its own label.
            five = {k: float(scores[k]) for k in sorted(EXPECTED_MOOD_SCORES)}
            attrs["mood_label"] = max(five, key=five.get)
        return attrs

    async def _resolve_inner(
        self,
        artist: str,
        title: str,
        duration: float | None,
        token: str,
    ) -> ResolutionResult | None:
        deadline = time.monotonic() + OVERALL_BUDGET

        # 0. Manual override — beats every automatic source. Checked
        # before the BPM cache so a user correction wins even over a
        # previously cached Deezer value. Keyed by artist|title only
        # (no duration bucket), so it survives duration drift.
        if self._overrides is not None:
            override = self._overrides.get(artist, title)
            if override and override.get("bpm"):
                _LOGGER.info(
                    "AX BPM override hit for %s - %s → %.1f BPM (%s)",
                    artist, title, override["bpm"], override.get("rule"),
                )
                return ResolutionResult(override["bpm"], **{
                    "source": SOURCE_OVERRIDE,
                    "octave_rule": override.get("rule"),
                    "original_bpm": override.get("original_bpm"),
                    "track": f"{artist} - {title}",
                })

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
        except TimeoutError:
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

        # 2b. iTunes fallback — preview/genre provider only (Apple exposes
        # no tempo). Fires when Deezer yielded no match at all OR a match
        # without a preview URL: the local path needs audio to analyze.
        # The pending Deezer match (bpm==0 metadata) wins when it already
        # carries a preview; iTunes only fills the gap.
        if not match or not match.get("preview_url"):
            try:
                itunes_match = await asyncio.wait_for(
                    self._itunes.find_match(artist, title, duration),
                    timeout=max(0.1, deadline - time.monotonic()),
                )
            except TimeoutError:
                itunes_match = None
            if itunes_match:
                _LOGGER.debug(
                    "iTunes fallback match for %s - %s (id=%s)",
                    artist, title, itunes_match.get("itunes_track_id"),
                )
                match = itunes_match

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
        except TimeoutError:
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

    async def _download_preview(self, preview_url: str, dest_path: str) -> bool:
        """Download a preview via the provider that produced the URL.

        iTunes preview URLs (audio-ssl.itunes.apple.com) must NOT be sent
        to the Deezer client — route by the match's provider field,
        defaulting to Deezer for legacy/unknown URLs.
        """
        if "itunes.apple.com" in preview_url or "audio-ssl.itunes" in preview_url:
            return await self._itunes.download_preview(preview_url, dest_path)
        return await self._deezer.download_preview(preview_url, dest_path)

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
        # mkstemp does filesystem I/O — keep it off the event loop. The
        # suffix follows the preview URL (.m4a for iTunes AAC previews,
        # .mp3 for Deezer) so the ffmpeg decode tier sniffs the container
        # correctly.
        loop = asyncio.get_running_loop()
        fd, tmp_path = await loop.run_in_executor(
            None,
            lambda: tempfile.mkstemp(
                suffix=preview_suffix(preview_url), prefix="ax_bpm_"
            ),
        )
        os.close(fd)
        try:
            if not await self._download_preview(preview_url, tmp_path):
                return None

            # Two-tier local analysis on the one preview:
            # 1. Analyzer healthy → ONE /analyze call returns BOTH the
            #    aubio bpm and the atomic mood_scores (decode-once design
            #    in the analyzer). Bounded by the remaining budget.
            # 2. Analyzer down/unneeded → NumPy floor locally; mood call
            #    (when enabled but tempo-less) runs concurrently.
            remaining = max(0.1, deadline - time.monotonic())
            bpm_raw: float | None = None
            mood_payload: dict[str, Any] | None = None

            if self.mood_enabled and self._mood_ready:
                analyzer_payload = await self._analyzer_client.async_analyze_file(
                    tmp_path
                )
                if analyzer_payload:
                    bpm_raw = analyzer_payload.get("bpm")
                    mood_payload = analyzer_payload

            if not bpm_raw or bpm_raw <= 0:
                # Analyzer tempo unavailable (not installed, degraded, or
                # tempo failed) → NumPy floor + optional concurrent mood.
                mood_task = None
                if self.mood_enabled and self._mood_ready and not mood_payload:
                    mood_task = asyncio.ensure_future(
                        self._analyzer_client.async_analyze_file(tmp_path)
                    )
                bpm_task = asyncio.ensure_future(
                    self._analyzer.get_bpm(self._hass, tmp_path)
                )
                done, pending = await asyncio.wait(
                    {t for t in (bpm_task, mood_task) if t},
                    timeout=min(remaining, ANALYSIS_TIMEOUT * 2),
                )
                for task in pending:
                    task.cancel()

                bpm_raw = bpm_task.result() if bpm_task in done else None
                mood_payload = (
                    mood_task.result()
                    if mood_task in done and not mood_task.cancelled()
                    else None
                )

            if not bpm_raw or bpm_raw <= 0:
                return None

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

            # Published mood attributes — ONE derivation (defects 1–3):
            # the local path merges the SAME _mood_attrs_from_payload the
            # enrichment path uses, replacing the hand-rolled
            # mood_scores/mood_label/intensity/calmness. mood_label argmax
            # fires only over the COMPLETE five-head set; published
            # intensity/calmness use the pinned /3, /2 formula (the merge
            # below lets the derivation override the genre-only defaults).
            # Octave GATING above still uses compute_signals (/4, /3)
            # internally — no re-thresholding.
            mood_attrs = (
                self._mood_attrs_from_payload(mood_payload)
                if mood_payload
                else {}
            )
            mood_attrs.pop("source", None)  # result-level source set below
            source = (
                SOURCE_ANALYZER
                if mood_payload is not None and mood_payload.get("bpm")
                else SOURCE_NUMPY
            )
            # bpm_engine (closeout): which engine produced the LOCAL tempo —
            # aubio (analyzer add-on) vs numpy (the floor). Deezer-metadata
            # BPM has no engine → attribute omitted. Semantics: this
            # describes the BPM SOURCE only; cached mood attributes can
            # outlive the current run ("mood from cache + BPM from floor"
            # is a legitimate mixed state, not a bug).
            attrs = {
                "bpm_engine": "aubio" if source == SOURCE_ANALYZER else "numpy",
                "track": f"{artist} - {title}",
                "isrc": match.get("isrc"),
                "deezer_track_id": match.get("deezer_track_id"),
                "itunes_track_id": match.get("itunes_track_id"),
                "provider": match.get("provider", "deezer"),
                "match_rank": match.get("match_rank"),
                "bpm_raw": bpm_raw,
                "genre": ", ".join(genres) if genres else None,
                "genre_source": (
                    f"{match.get('provider', 'deezer')}_album"
                    if genres
                    else None
                ),
                "intensity": disambiguated["intensity"],
                "calmness": disambiguated["calmness"],
                "octave_corrected": disambiguated["corrected"],
                "octave_rule": disambiguated["rule"],
                "preview_analyzed": True,
            }
            attrs.update(mood_attrs)
            result = ResolutionResult(
                disambiguated["bpm"], source=source, **attrs
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
