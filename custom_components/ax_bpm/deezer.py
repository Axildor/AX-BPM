"""Anonymous Deezer API client.

All endpoints are public reads — no API key, no account, no OAuth.
Typically 3 requests per unseen track: search + track + album.

The signed preview URL is time-limited: download it immediately, never
cache the URL itself.
"""

from __future__ import annotations

import logging
import re
import urllib.parse

import aiohttp

from .const import DEEZER_API, DURATION_TOLERANCE, NETWORK_TIMEOUT

_LOGGER = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "HomeAssistant/AX-BPM (anonymous)"}

# Title suffixes stripped for the cleaned-title retry query.
_TITLE_NOISE = re.compile(
    r"\s*[\(\[](?:feat\.?|ft\.?|featuring|remix|radio edit|single version|"
    r"album version|extended|edit|version|remaster(?:ed)?(?:\s*\d{4})?|"
    r"live|bonus track)[^\)\]]*[\)\]]",
    re.IGNORECASE,
)
_FEAT_TAIL = re.compile(r"\s+(?:feat\.?|ft\.?|featuring)\s+.*$", re.IGNORECASE)


def _norm(text: str | None) -> str:
    """Light artist-name normalization: lowercase, collapse whitespace."""
    if not text:
        return ""
    return " ".join(text.lower().split())


def clean_title(title: str) -> str:
    """Strip (Remix)/(Radio Edit)/feat. suffixes for the retry query."""
    cleaned = _TITLE_NOISE.sub("", title)
    cleaned = _FEAT_TAIL.sub("", cleaned)
    return cleaned.strip()


class DeezerClient:
    """Async client for anonymous Deezer public endpoints."""

    def __init__(self, session: aiohttp.ClientSession) -> None:
        self._session = session

    async def _get_json(self, path: str, params: dict | None = None) -> dict | None:
        url = f"{DEEZER_API}{path}"
        try:
            async with self._session.get(
                url,
                params=params,
                headers=_HEADERS,
                timeout=aiohttp.ClientTimeout(total=NETWORK_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    _LOGGER.debug("Deezer %s returned HTTP %s", path, resp.status)
                    return None
                return await resp.json()
        except (aiohttp.ClientError, TimeoutError) as err:
            _LOGGER.debug("Deezer request failed for %s: %s", path, err)
            return None

    async def search_tracks(
        self, artist: str, title: str
    ) -> list[dict]:
        """Search Deezer for tracks. Returns data list.

        Tries the field-quoted query first (artist:"X" track:"Y"); Deezer
        has disabled the artist: field operator (returns 0 results since
        ~2026-09), so fall back to plain-text "{artist} {title}" when the
        quoted query yields nothing.
        """
        queries = [f'artist:"{artist}" track:"{title}"', f"{artist} {title}"]
        for query in queries:
            data = await self._get_json(
                "/search", {"q": query, "limit": 25}
            )
            results = list((data or {}).get("data") or [])
            if results:
                _LOGGER.debug(
                    "Deezer search %r → %d results", query, len(results)
                )
                return results
        _LOGGER.debug(
            "Deezer search: no results for %r (all query variants)", title
        )
        return []

    async def find_match(
        self,
        artist: str,
        title: str,
        duration: float | None,
    ) -> dict | None:
        """Search + candidate selection.

        Prefer duration within ±DURATION_TOLERANCE of media_duration (hard
        filter when available), then highest rank wins. Falls back to a
        cleaned title when the first attempt yields nothing usable.

        Plain-text fallback results can include wrong-artist tracks, so
        candidates are verified against the artist name (normalized
        substring match on the candidate's artist name) before ranking.
        """
        want_artist = _norm(artist)
        for attempt_title in (title, clean_title(title)):
            if not attempt_title:
                continue
            candidates = await self.search_tracks(artist, attempt_title)
            # Artist verification: the candidate's artist name must contain
            # the wanted artist (or vice versa) after normalization. Handles
            # "3 Doors Down" vs "3 Doors Down feat. X" style variants.
            verified = [
                c
                for c in candidates
                if want_artist
                and (
                    want_artist in _norm((c.get("artist") or {}).get("name"))
                    or _norm((c.get("artist") or {}).get("name"))
                    in want_artist
                )
            ]
            if duration is not None:
                filtered = [
                    c
                    for c in verified
                    if isinstance(c.get("duration"), (int, float))
                    and abs(c["duration"] - duration) <= DURATION_TOLERANCE
                ]
            else:
                filtered = verified
            if filtered:
                best = max(filtered, key=lambda c: c.get("rank") or 0)
                _LOGGER.debug(
                    "Deezer match: %s - %s (id=%s, dur=%s, rank=%s)",
                    (best.get("artist") or {}).get("name"),
                    best.get("title"),
                    best.get("id"),
                    best.get("duration"),
                    best.get("rank"),
                )
                return best
            _LOGGER.debug(
                "Deezer: no verified/duration-matched candidate for "
                "%s - %s (%d raw candidates)",
                artist,
                attempt_title,
                len(candidates),
            )
        return None

    async def get_track(self, track_id: int | str) -> dict | None:
        """GET /track/{id} → bpm, isrc, preview, album.id."""
        return await self._get_json(f"/track/{track_id}")

    async def get_album_genres(self, album_id: int | str) -> list[str]:
        """GET /album/{id} → genres.data[].name (album-level, coarse)."""
        data = await self._get_json(f"/album/{album_id}")
        if not data:
            return []
        genres = data.get("genres") or {}
        return [
            g.get("name")
            for g in (genres.get("data") or [])
            if g.get("name")
        ]

    async def download_preview(self, preview_url: str, dest_path: str) -> bool:
        """Download the ~30s preview MP3 to dest_path. Use immediately.

        Returns True on success. The URL is never cached by callers.
        """
        try:
            async with self._session.get(
                preview_url,
                headers=_HEADERS,
                timeout=aiohttp.ClientTimeout(total=NETWORK_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    _LOGGER.info(
                        "AX BPM: preview download failed (HTTP %s)", resp.status
                    )
                    return False
                with open(dest_path, "wb") as fh:
                    async for chunk in resp.content.iter_chunked(64 * 1024):
                        fh.write(chunk)
            _LOGGER.debug("AX BPM: preview downloaded to %s", dest_path)
            return True
        except (aiohttp.ClientError, TimeoutError, OSError) as err:
            _LOGGER.info("AX BPM: preview download failed: %s", err)
            return False


def parse_artist_title(media_artist: str | None, media_title: str | None) -> tuple[str, str] | None:
    """Fallback parse of "Artist - Title" when media_artist is missing."""
    if media_artist and media_title:
        return media_artist.strip(), media_title.strip()
    if media_title and " - " in media_title:
        artist, _, title = media_title.partition(" - ")
        if artist.strip() and title.strip():
            return artist.strip(), title.strip()
    return None