"""Config-entry migration tests: v1 (sidecar/mood keys) → v2 (analyzer keys).

The v1 → v2 rename moved `mood_analyzer_url` → `analyzer_url` and
`mood_api_token` → `analyzer_api_token`. Existing installs must migrate
silently, preserving values, in BOTH `data` and `options`.

`async_migrate_entry` is called directly with a stub entry — the conftest
HA stubs make the module importable without a full HA runtime.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ax_bpm.const import (
    CONF_ANALYZER_API_TOKEN,
    CONF_ANALYZER_URL,
    CONF_DISCOVERED_ANALYZER_URL,
)
from ax_bpm.migration import async_migrate_entry, rename_keys


def _stub_entry(version: int, data: dict, options: dict | None = None):
    """Minimal ConfigEntry stand-in for the migration function."""
    entry = MagicMock()
    entry.version = version
    entry.data = data
    entry.options = options or {}
    return entry


def _stub_hass():
    """Hass stub capturing async_update_entry calls."""
    hass = MagicMock()
    hass.config_entries.async_update_entry = MagicMock()
    return hass


def test_rename_keys_maps_legacy_keys():
    assert rename_keys({"mood_analyzer_url": "http://x:8099"}) == {
        CONF_ANALYZER_URL: "http://x:8099"
    }
    assert rename_keys({"mood_api_token": "secret"}) == {
        CONF_ANALYZER_API_TOKEN: "secret"
    }


def test_rename_keys_leaves_current_keys_untouched():
    original = {
        CONF_ANALYZER_URL: "http://x:8099",
        CONF_ANALYZER_API_TOKEN: "secret",
        "media_player": "media_player.living_room",
        CONF_DISCOVERED_ANALYZER_URL: "http://discovered:8099",
    }
    assert rename_keys(original) == original


def test_rename_keys_new_key_wins_on_collision():
    """If both old and new keys exist, the new value is preserved."""
    renamed = rename_keys(
        {"mood_analyzer_url": "http://old:8099", CONF_ANALYZER_URL: "http://new:8099"}
    )
    assert renamed[CONF_ANALYZER_URL] == "http://new:8099"


@pytest.mark.asyncio
async def test_migrate_v1_rewrites_data_and_options():
    hass = _stub_hass()
    entry = _stub_entry(
        version=1,
        data={
            "media_player": "media_player.living_room",
            "mood_analyzer_url": "http://old:8099",
        },
        options={"mood_api_token": "secret"},
    )

    assert await async_migrate_entry(hass, entry) is True

    hass.config_entries.async_update_entry.assert_called_once()
    kwargs = hass.config_entries.async_update_entry.call_args.kwargs
    assert kwargs["version"] == 2
    # Values preserved under the new keys.
    assert kwargs["data"][CONF_ANALYZER_URL] == "http://old:8099"
    assert kwargs["data"]["media_player"] == "media_player.living_room"
    assert kwargs["options"][CONF_ANALYZER_API_TOKEN] == "secret"
    # Old keys gone.
    assert "mood_analyzer_url" not in kwargs["data"]
    assert "mood_api_token" not in kwargs["options"]


@pytest.mark.asyncio
async def test_migrate_v2_is_noop():
    """An already-migrated entry is left alone."""
    hass = _stub_hass()
    entry = _stub_entry(version=2, data={CONF_ANALYZER_URL: "http://x:8099"})

    assert await async_migrate_entry(hass, entry) is True
    hass.config_entries.async_update_entry.assert_not_called()


@pytest.mark.asyncio
async def test_migrate_future_version_refuses():
    """A downgrade from a future version fails setup rather than corrupting."""
    hass = _stub_hass()
    entry = _stub_entry(version=3, data={})

    assert await async_migrate_entry(hass, entry) is False
    hass.config_entries.async_update_entry.assert_not_called()
