"""Persistent BPM cache backed by the Home Assistant Store (.storage).

Key: ISRC when previously matched, else a hash of the normalized
"artist|title|duration bucket (±3s)". Survives restarts. A cache hit
publishes immediately — no network calls, no analysis.

Schema v2 (Phase 1): legacy Essentia SVM mood fields (mood_scores,
mood_label) are dropped on load; BPM fields are kept. Sidecar mood
results are cached per-ISRC alongside BPM under the same keys.
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

STORAGE_VERSION = 2
STORAGE_KEY = "ax_bpm_cache"

# Legacy fields removed by the v1 → v2 migration.
LEGACY_MOOD_FIELDS = ("mood_scores", "mood_label")


def normalize(text: str | None) -> str:
    """Lowercase, collapse whitespace, strip punctuation-ish noise."""
    if not text:
        return ""
    return " ".join(text.lower().split())


def duration_bucket(duration: float | None) -> int:
    """Bucket duration to ±DURATION_TOLERANCE granularity (None → 0)."""
    if duration is None:
        return 0
    return math.floor(duration / (2 * DURATION_TOLERANCE))


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
        if not isinstance(data, dict):
            self._data = {}
            return
        # v1 → v2 migration: drop legacy SVM mood fields, keep BPM.
        migrated = False
        for entry in data.values():
            if not isinstance(entry, dict):
                continue
            for field in LEGACY_MOOD_FIELDS:
                if field in entry:
                    entry.pop(field, None)
                    migrated = True
        if migrated:
            _LOGGER.info(
                "AX BPM cache: migrated v1 → v2 (legacy SVM mood fields dropped)"
            )
        self._data = dict(data)

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