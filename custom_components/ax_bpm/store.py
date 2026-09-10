"""Persistent BPM cache backed by the Home Assistant Store (.storage).

Key: ISRC when previously matched, else a hash of the normalized
"artist|title|duration bucket (±3s)". Survives restarts. A cache hit
publishes immediately — no network calls, no analysis.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import DURATION_TOLERANCE, SOURCE_CACHE

_LOGGER = logging.getLogger(__name__)

STORAGE_VERSION = 1
STORAGE_KEY = "ax_bpm_cache"


def normalize(text: str | None) -> str:
    """Lowercase, collapse whitespace, strip punctuation-ish noise."""
    if not text:
        return ""
    return " ".join(text.lower().split())


def duration_bucket(duration: float | None) -> int:
    """Bucket duration to ±DURATION_TOLERANCE granularity (None → 0)."""
    if duration is None:
        return 0
    return int(math.floor(duration / (2 * DURATION_TOLERANCE)))


def cache_key(
    isrc: str | None,
    artist: str | None,
    title: str | None,
    duration: float | None,
) -> str:
    """ISRC when known, else hash of normalized artist|title|duration bucket."""
    if isrc:
        return f"isrc:{isrc.upper()}"
    raw = f"{normalize(artist)}|{normalize(title)}|{duration_bucket(duration)}"
    return "hash:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


class BpmCache:
    """Async wrapper around a HA Store dict of cache_key → result dict."""

    def __init__(self, hass: HomeAssistant) -> None:
        self._store = Store(hass, STORAGE_VERSION, STORAGE_KEY)
        self._data: dict[str, dict[str, Any]] = {}

    async def async_load(self) -> None:
        data = await self._store.async_load()
        self._data = dict(data) if isinstance(data, dict) else {}

    def get(self, key: str) -> dict[str, Any] | None:
        entry = self._data.get(key)
        if not entry:
            return None
        result = dict(entry)
        result["source"] = SOURCE_CACHE
        return result

    async def async_put(self, key: str, result: dict[str, Any]) -> None:
        """Store a resolution result (without the volatile preview URL)."""
        entry = {k: v for k, v in result.items() if k != "preview_url"}
        self._data[key] = entry
        await self._store.async_save(self._data)