"""Persistent manual BPM + mood overrides backed by the HA Store.

Separate from the BPM cache (store.py) on purpose:

- Keyed by normalized "artist|title" ONLY — no duration bucket, no ISRC.
  An override must survive duration drift between plays and the
  ISRC-vs-hash key divergence in cache_key(): a Deezer resolution stores
  under isrc:KEY while the button handler only knows artist/title. A
  dedicated store sidesteps both.
- Checked BEFORE the BPM cache in the pipeline: an explicit user
  correction beats every automatic source, including a previously cached
  Deezer value.
- Independently clearable: clear_cache must never touch overrides and
  vice versa.

Value schema (v1, extended in 3.3.0 — optional fields, no migration):
  bpm:            corrected BPM (float, > 0) — present when the BPM is
                  overridden (manual_half / manual_double / manual_set)
  rule:           "manual_half" | "manual_double" | "manual_set"
  original_bpm:   the BPM the correction was applied to (float)
  mood_label:     one of the five mood heads — present when the mood is
                  overridden; published as one-hot synthetic scores
  created:        ISO-8601 UTC timestamp (first write for the track)

One entry per track may carry a BPM override, a mood override, or both;
async_set_bpm / async_set_mood merge into the existing entry.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import (
    EXPECTED_MOOD_SCORES,
    OVERRIDE_STORAGE_KEY,
    RULE_MANUAL_DOUBLE,
    RULE_MANUAL_HALF,
    RULE_MANUAL_SET,
)

_LOGGER = logging.getLogger(__name__)

OVERRIDE_VERSION = 1


def override_key(artist: str | None, title: str | None) -> str:
    """Stable key: sha1 of the normalized "artist|title" pair.

    Uses the same normalize() semantics as the BPM cache (lowercase,
    collapsed whitespace) so "Alicia  Keys" and "alicia keys" collide
    correctly. No duration component — an override applies to the song,
    not to one playback.
    """
    from .store import normalize  # local import: avoids a circular module

    raw = f"{normalize(artist)}|{normalize(title)}"
    return "ovr:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


class BpmOverrideStore:
    """Async wrapper around a HA Store dict of override_key → override dict."""

    def __init__(self, hass: HomeAssistant) -> None:
        self._store: Store[dict[str, Any]] = Store(
            hass, OVERRIDE_VERSION, OVERRIDE_STORAGE_KEY
        )
        self._data: dict[str, dict[str, Any]] = {}

    async def async_load(self) -> None:
        data = await self._store.async_load()
        self._data = dict(data) if isinstance(data, dict) else {}

    def get(self, artist: str | None, title: str | None) -> dict[str, Any] | None:
        """Return a copy of the override for this track, or None."""
        entry = self._data.get(override_key(artist, title))
        return dict(entry) if entry else None

    async def async_put(
        self,
        artist: str | None,
        title: str | None,
        bpm: float,
        rule: str,
        original_bpm: float,
    ) -> None:
        """Store a manual BPM correction for this track.

        Merges into an existing entry so a mood override survives a
        later BPM correction (and vice versa via async_set_mood).
        """
        if rule not in (RULE_MANUAL_HALF, RULE_MANUAL_DOUBLE, RULE_MANUAL_SET):
            raise ValueError(f"invalid override rule: {rule!r}")
        if not bpm or bpm <= 0:
            raise ValueError(f"override bpm must be > 0, got {bpm}")
        key = override_key(artist, title)
        entry = self._data.get(key) or {}
        entry.update({
            "bpm": float(bpm),
            "rule": rule,
            "original_bpm": float(original_bpm),
        })
        entry.setdefault("created", datetime.now(UTC).isoformat())
        self._data[key] = entry
        await self._store.async_save(self._data)
        _LOGGER.info(
            "AX BPM override stored: %s - %s → %.1f BPM (%s, was %.1f)",
            artist, title, bpm, rule, original_bpm,
        )

    async def async_set_mood(
        self, artist: str | None, title: str | None, label: str
    ) -> None:
        """Store a manual mood override for this track.

        Merges into an existing entry so a BPM override survives a
        later mood correction. The label is validated against the five
        mood heads; the pipeline publishes one-hot synthetic scores.
        """
        if label not in EXPECTED_MOOD_SCORES:
            raise ValueError(f"invalid mood label: {label!r}")
        key = override_key(artist, title)
        entry = self._data.get(key) or {}
        entry["mood_label"] = label
        entry.setdefault("created", datetime.now(UTC).isoformat())
        self._data[key] = entry
        await self._store.async_save(self._data)
        _LOGGER.info(
            "AX BPM mood override stored: %s - %s → %s", artist, title, label
        )

    async def async_remove(self, artist: str | None, title: str | None) -> bool:
        """Remove the override for this track. True when one existed."""
        key = override_key(artist, title)
        if key not in self._data:
            return False
        del self._data[key]
        await self._store.async_save(self._data)
        _LOGGER.info("AX BPM override removed: %s - %s", artist, title)
        return True

    async def async_clear(self) -> int:
        """Remove ALL overrides. Returns the number removed."""
        count = len(self._data)
        if count:
            self._data = {}
            await self._store.async_save(self._data)
            _LOGGER.info("AX BPM: cleared %d override(s)", count)
        return count

    def count(self) -> int:
        """Number of stored overrides (for logging / diagnostics)."""
        return len(self._data)
