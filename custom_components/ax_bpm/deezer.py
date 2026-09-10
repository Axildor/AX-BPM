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
        """Field-quoted search: artist:"X" track:"Y". Returns data list."""
        query = f'artist:"{artist}" track:"{title}"'
        data = await self._get_json(
            "/search", {"q": query, "limit": 25}
        )
        if not data:
            return []
        return list(data.get("data") or [])

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
        """
        for attempt_title in (title, clean_title(title)):
            if not attempt_title:
                continue
            candidates = await self.search_tracks(artist, attempt_title)
            if duration is not None:
                filtered = [
                    c
                    for c in candidates
                    if isinstance(c.get("duration"), (int, float))
                    and abs(c["duration"] - duration) <= DURATION_TOLERANCE
                ]
            else:
                filtered = candidates
            if filtered:
                return max(
                    filtered, key=lambda c: c.get("rank") or 0
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
                    _LOGGER.debug("Preview download HTTP %s", resp.status)
                    return False
                with open(dest_path, "wb") as fh:
                    async for chunk in resp.content.iter_chunked(64 * 1024):
                        fh.write(chunk)
            return True
        except (aiohttp.ClientError, TimeoutError, OSError) as err:
            _LOGGER.debug("Preview download failed: %s", err)
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