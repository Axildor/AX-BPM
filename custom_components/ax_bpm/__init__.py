"""The AX BPM integration."""

from __future__ import annotations

import logging

import aiohttp
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import CONF_MEDIA_PLAYER, DOMAIN, PLATFORMS
from .pipeline import BpmPipeline
from .store import BpmCache

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up AX BPM from a config entry."""
    session = aiohttp.ClientSession()
    cache = BpmCache(hass)
    await cache.async_load()
    pipeline = BpmPipeline(hass, session, cache, dict(entry.options))
    await pipeline.async_setup()

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
        data = hass.data[DOMAIN].pop(entry.entry_id, {})
        session = data.get("session")
        if session:
            await session.close()
    return unloaded