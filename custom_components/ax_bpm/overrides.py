"""Persistent manual BPM overrides backed by the Home Assistant Store.

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

Value schema (v1):
  bpm:            corrected BPM (float, > 0)
  rule:           "manual_half" | "manual_double"
  original_bpm:   the BPM the correction was applied to (float)
  created:        ISO-8601 UTC timestamp
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import OVERRIDE_STORAGE_KEY, RULE_MANUAL_DOUBLE, RULE_MANUAL_HALF

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
        """Store a manual correction for this track."""
        if rule not in (RULE_MANUAL_HALF, RULE_MANUAL_DOUBLE):
            raise ValueError(f"invalid override rule: {rule!r}")
        if not bpm or bpm <= 0:
            raise ValueError(f"override bpm must be > 0, got {bpm}")
        entry = {
            "bpm": float(bpm),
            "rule": rule,
            "original_bpm": float(original_bpm),
            "created": datetime.now(UTC).isoformat(),
        }
        self._data[override_key(artist, title)] = entry
        await self._store.async_save(self._data)
        _LOGGER.info(
            "AX BPM override stored: %s - %s → %.1f BPM (%s, was %.1f)",
            artist, title, bpm, rule, original_bpm,
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
