"""Config-entry migration: v1 (sidecar/mood keys) → v2 (analyzer keys).

The naming overhaul renamed `mood_analyzer_url` → `analyzer_url` and
`mood_api_token` → `analyzer_api_token`. Existing installs must migrate
silently, preserving values, in BOTH `data` and `options`.

Kept in its own module (no homeassistant import at module level) so the
pure rename logic is unit-testable without a full HA runtime. The
`async_migrate_entry` symbol is re-exported from `__init__.py`, which is
where Home Assistant looks for it.
"""

from __future__ import annotations

import logging
from typing import Any

_LOGGER = logging.getLogger(__name__)

# v1 → v2 config-key renames (the "sidecar"/"mood" era keys).
KEY_RENAMES = {
    "mood_analyzer_url": "analyzer_url",
    "mood_api_token": "analyzer_api_token",
}

CURRENT_VERSION = 2


def rename_keys(mapping: dict[str, Any]) -> dict[str, Any]:
    """Rename legacy keys, preserving values.

    Iteration order means a pre-existing new key wins on collision (the
    old key is written first, then overwritten by the new one).
    """
    renamed: dict[str, Any] = {}
    for key, value in mapping.items():
        renamed[KEY_RENAMES.get(key, key)] = value
    return renamed


async def async_migrate_entry(hass, entry) -> bool:
    """Migrate a v1 entry to v2 (sidecar/mood keys → analyzer keys).

    v1 stored `mood_analyzer_url` / `mood_api_token`; v2 uses
    `analyzer_url` / `analyzer_api_token`. Values are preserved so
    existing installs keep working without reconfiguration.
    """
    if entry.version > CURRENT_VERSION:
        # Downgrade from a future version — nothing we can safely do.
        _LOGGER.error(
            "AX BPM: cannot migrate config entry from version %s to %s",
            entry.version,
            CURRENT_VERSION,
        )
        return False

    if entry.version < CURRENT_VERSION:
        _LOGGER.debug(
            "AX BPM: migrating config entry from v%s to v%s",
            entry.version,
            CURRENT_VERSION,
        )
        hass.config_entries.async_update_entry(
            entry,
            data=rename_keys(dict(entry.data)),
            options=rename_keys(dict(entry.options)),
            version=CURRENT_VERSION,
        )
        _LOGGER.info("AX BPM: config entry migrated to v%s", CURRENT_VERSION)

    return True
