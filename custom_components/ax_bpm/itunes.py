"""Anonymous iTunes Search API client — preview/genre fallback provider.

The iTunes Search API (https://itunes.apple.com/search) exposes the Apple
catalog with NO authentication — the pragmatic alternative to the Apple
Music API, whose developer-token (JWT) requirement makes it a poor fit
for a HACS integration.

Role in the pipeline: PREVIEW + GENRE provider only. Apple exposes no
tempo anywhere in its public APIs, so this client never supplies a BPM —
it exists to close the gap where Deezer yields no match or a match
without a preview URL, giving the local analysis path audio to work with.

Verified live (2026-09-30):
- GET /search?term=...&entity=song&limit=N → {"resultCount": N, "results": [...]}
- Song fields: trackId, trackName, artistName, collectionName,
  previewUrl (.m4a AAC-LC), trackTimeMillis, primaryGenreName.
- No ISRC in search results (unlike Deezer /track).
- Preview download: HTTP 200, ~1 MB, "ISO Media, Apple iTunes ALAC/AAC-LC".
- Unofficial rate limit ~20 req/min/IP — only fires when Deezer fails,
  so normal operation sends it near-zero traffic.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

import aiohttp

from .const import (
    DURATION_TOLERANCE,
    ITUNES_API,
    ITUNES_GENRE_ALIASES,
    NETWORK_TIMEOUT,
)
from .deezer import _norm, clean_title

_LOGGER = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "AX-BPM-HomeAssistant/3.0 (+https://github.com/adix992/AX-BPM)",
}


def preview_suffix(url: str) -> str:
    """File extension for a preview URL (.m4a for iTunes, .mp3 for Deezer).

    The temp file must keep the right extension so the ffmpeg decode tier
    sniffs the container correctly (AAC in an m4a container vs MP3).
    """
    path = urlparse(url).path.lower()
    if path.endswith(".m4a") or path.endswith(".m4p"):
        return ".m4a"
    return ".mp3"


def map_genres(apple_genres: list[str]) -> list[str]:
    """Map Apple genre names onto the Deezer vocabulary.

    The octave-disambiguation whitelists (FAST_GENRES / SLOW_GENRES) are
    keyed on Deezer album-genre strings. Apple's vocabulary differs
    ("Dance", "Electronica", "Hip-Hop/Rap", ...). Conservative mapping:
    only aliases with an unambiguous tempo meaning are translated;
    unmapped genres pass through unchanged and simply don't gate.
    """
    mapped: list[str] = []
    for genre in apple_genres:
        if not genre:
            continue
        mapped.append(ITUNES_GENRE_ALIASES.get(genre.lower(), genre))
    return mapped


class ItunesClient:
    """Async client for the anonymous iTunes Search API."""

    def __init__(self, session: aiohttp.ClientSession) -> None:
        self._session = session

    async def _get_json(self, params: dict) -> dict | None:
        try:
            async with self._session.get(
                ITUNES_API,
                params=params,
                headers=_HEADERS,
                timeout=aiohttp.ClientTimeout(total=NETWORK_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    _LOGGER.debug("iTunes search returned HTTP %s", resp.status)
                    return None
                return await resp.json()
        except (aiohttp.ClientError, TimeoutError) as err:
            _LOGGER.debug("iTunes search request failed: %s", err)
            return None

    async def find_match(
        self,
        artist: str,
        title: str,
        duration: float | None,
    ) -> dict | None:
        """Search + candidate selection, mirroring DeezerClient.find_match.

        Returns a provider-neutral match dict:
        {preview_url, genres, itunes_track_id, provider} — or None.

        Same selection rules as the Deezer client: artist verification
        (normalized substring match) then a ±DURATION_TOLERANCE hard
        filter when media_duration is available. No rank field exists in
        the iTunes response, so candidates keep their search order
        (relevance order) — the first verified, duration-matched
        candidate wins.
        """
        want_artist = _norm(artist)
        for attempt_title in (title, clean_title(title)):
            if not attempt_title:
                continue
            data = await self._get_json({
                "term": f"{artist} {attempt_title}",
                "entity": "song",
                "limit": 25,
            })
            candidates = list((data or {}).get("results") or [])
            if not candidates:
                _LOGGER.debug(
                    "iTunes search: no results for %r", attempt_title
                )
                continue

            # Artist verification — same normalized substring rule as the
            # Deezer client (handles "feat. X" style variants).
            verified = [
                c
                for c in candidates
                if want_artist
                and (
                    want_artist in _norm(c.get("artistName"))
                    or _norm(c.get("artistName")) in want_artist
                )
            ]
            if duration is not None:
                filtered = [
                    c
                    for c in verified
                    if isinstance(c.get("trackTimeMillis"), (int, float))
                    and abs(c["trackTimeMillis"] / 1000.0 - duration)
                    <= DURATION_TOLERANCE
                ]
            else:
                filtered = verified
            if not filtered:
                _LOGGER.debug(
                    "iTunes: no verified/duration-matched candidate for "
                    "%s - %s (%d raw candidates)",
                    artist,
                    attempt_title,
                    len(candidates),
                )
                continue

            # Search order = relevance order; first match wins. A missing
            # previewUrl makes the candidate useless (the local path needs
            # audio) — skip to the next one.
            for best in filtered:
                preview_url = best.get("previewUrl")
                if not preview_url:
                    continue
                _LOGGER.debug(
                    "iTunes match: %s - %s (id=%s, dur=%s)",
                    best.get("artistName"),
                    best.get("trackName"),
                    best.get("trackId"),
                    best.get("trackTimeMillis"),
                )
                return {
                    "preview_url": preview_url,
                    "genres": map_genres(
                        [g for g in (best.get("primaryGenreName"),) if g]
                    ),
                    "itunes_track_id": best.get("trackId"),
                    "provider": "itunes",
                }
            _LOGGER.debug(
                "iTunes: %d verified candidates for %s - %s, none with a "
                "previewUrl",
                len(filtered),
                artist,
                attempt_title,
            )
        return None

    async def download_preview(self, preview_url: str, dest_path: str) -> bool:
        """Download the ~30s preview (m4a/AAC) to dest_path. Use immediately."""
        try:
            async with self._session.get(
                preview_url,
                headers=_HEADERS,
                timeout=aiohttp.ClientTimeout(total=NETWORK_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    _LOGGER.debug(
                        "iTunes preview download returned HTTP %s", resp.status
                    )
                    return False
                data = await resp.read()
        except (aiohttp.ClientError, TimeoutError) as err:
            _LOGGER.debug("iTunes preview download failed: %s", err)
            return False
        if not data:
            return False
        try:
            with open(dest_path, "wb") as f:
                f.write(data)
        except OSError as err:
            _LOGGER.debug("iTunes preview write failed: %s", err)
            return False
        return True
