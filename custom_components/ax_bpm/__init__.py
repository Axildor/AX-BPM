"""The AX BPM integration."""

from __future__ import annotations

import logging

import aiohttp
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir

from .analyzer import aubio_available
from .const import CONF_AUBIO_BINARY, CONF_MEDIA_PLAYER, DOMAIN, PLATFORMS
from .pipeline import BpmPipeline
from .store import BpmCache

_LOGGER = logging.getLogger(__name__)

ISSUE_NO_ANALYZER = "no_tempo_analyzer"


def _update_analyzer_issue(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Create/clear a repair issue for the missing tempo analyzer.

    With the built-in NumPy estimator + ffmpeg decode chain, local analysis
    is almost always available; this issue now only fires when no decoder
    and no analyzer exist at all (e.g. ffmpeg missing and no wheels).
    """
    if aubio_available(entry.options.get(CONF_AUBIO_BINARY)):
        ir.async_delete_issue(hass, DOMAIN, ISSUE_NO_ANALYZER)
    else:
        ir.async_create_issue(
            hass,
            DOMAIN,
            ISSUE_NO_ANALYZER,
            is_fixable=False,
            severity=ir.IssueSeverity.WARNING,
            translation_key=ISSUE_NO_ANALYZER,
            translation_placeholders={
                "entry_title": entry.title,
            },
            learn_more_url="https://github.com/adix992/AX-BPM#dependency-footprint",
        )


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up AX BPM from a config entry."""
    session = aiohttp.ClientSession()
    cache = BpmCache(hass)
    await cache.async_load()
    # Options override data; merged view drives the dropdown + URL fields
    # (legacy toggle entries migrate on the next options save).
    pipeline = BpmPipeline(
        hass, session, cache, {**entry.data, **entry.options}
    )
    await pipeline.async_setup()
    _update_analyzer_issue(hass, entry)

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {
        "session": session,
        "cache": cache,
        "pipeline": pipeline,
        "media_player": entry.data.get(CONF_MEDIA_PLAYER),
    }

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    return True


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload on options change."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded:
        ir.async_delete_issue(hass, DOMAIN, ISSUE_NO_ANALYZER)
        data = hass.data[DOMAIN].pop(entry.entry_id, {})
        session = data.get("session")
        if session:
            await session.close()
    return unloaded