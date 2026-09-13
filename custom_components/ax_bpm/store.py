"""Persistent BPM cache backed by the Home Assistant Store (.storage).

Key: ISRC when previously matched, else a hash of the normalized
"artist|title|duration bucket (±3s)". Survives restarts. A cache hit
publishes immediately — no network calls, no analysis.

Schema v2 (Phase 1): legacy Essentia SVM mood fields (mood_scores,
mood_label) are dropped on load; BPM fields are kept. Analyzer mood
results are cached per-ISRC alongside BPM under the same keys.

The v1 → v2 migration runs at the Store level (_async_migrate_func
override): HA calls it when the on-disk version differs from
STORAGE_VERSION, BEFORE the data is returned to async_load. Without it,
loading a v1 file raises NotImplementedError from the Store base class
and the config entry fails to set up.
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


def strip_legacy_mood_fields(data: dict[str, Any]) -> bool:
    """Drop legacy SVM mood fields from every cache entry, in place.

    Returns True when anything was removed (for logging).
    """
    migrated = False
    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        for field in LEGACY_MOOD_FIELDS:
            if field in entry:
                entry.pop(field, None)
                migrated = True
    return migrated


class MigratingStore(Store[dict[str, Any]]):
    """Store that migrates older on-disk cache schemas at load time.

    HA's Store calls _async_migrate_func when the stored file's version
    differs from the declared version. The base implementation raises
    NotImplementedError, so the override here is REQUIRED for any
    version bump — a v1 file would otherwise crash async_setup_entry.
    """

    async def _async_migrate_func(
        self,
        old_major_version: int,
        old_minor_version: int,
        old_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Migrate v1 cache data (legacy SVM mood fields) to v2."""
        if old_major_version < STORAGE_VERSION and strip_legacy_mood_fields(old_data):
            _LOGGER.info(
                "AX BPM cache: migrated v%s → v%s (legacy SVM mood fields dropped)",
                old_major_version,
                STORAGE_VERSION,
            )
        return old_data


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
        self._store = MigratingStore(hass, STORAGE_VERSION, STORAGE_KEY)
        self._data: dict[str, dict[str, Any]] = {}

    async def async_load(self) -> None:
        data = await self._store.async_load()
        if not isinstance(data, dict):
            self._data = {}
            return
        # Defense-in-depth: the Store-level migration handles the v1 →
        # v2 bump, but sweep legacy fields here too in case a v2 file
        # somehow still contains them.
        if strip_legacy_mood_fields(data):
            _LOGGER.info(
                "AX BPM cache: legacy SVM mood fields dropped on load (v1 → v2)"
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