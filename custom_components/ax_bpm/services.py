"""Service handlers for manual BPM/mood corrections and cache management.

Seven services, all operating on the configured media player's CURRENT
track (the same artist/title/duration the sensor resolves):

  ax_bpm.halve_bpm       — halve the currently published BPM, persist it
  ax_bpm.double_bpm      — double the currently published BPM, persist it
  ax_bpm.set_bpm         — set an exact BPM for the current track
  ax_bpm.set_mood        — set the mood (one of the five heads) for the
                           current track
  ax_bpm.clear_override  — remove the manual override (BPM AND mood)
                           for the current track
  ax_bpm.clear_cache     — wipe the BPM cache (overrides untouched)
  ax_bpm.clear_overrides — wipe ALL manual overrides

Design notes:
- Halve/double apply to the sensor's CURRENTLY PUBLISHED value (what the
  user sees), then persist to the override store. The pipeline's
  override-first check makes the correction stick across re-resolutions
  and restarts.
- set_bpm does NOT require a published BPM — the user supplies the
  value, so it also works (and rescues) tracks that resolve to unknown.
- set_mood stores one of the five mood heads; the pipeline publishes
  one-hot synthetic scores through the same formula as the analyzer
  path.
- When the sensor is unknown (no BPM published), halve/double are no-ops
  with a WARNING — never publish 0 or None.
- Buttons (button.py) call these same handlers via hass.services.async_call,
  so automations and button presses share one code path.
- After any mutation the sensor is asked to re-resolve the current track
  immediately (bypassing the debounce) so the correction is visible at
  once.
"""

from __future__ import annotations

import logging

import voluptuous as vol
from homeassistant.const import STATE_PLAYING
from homeassistant.core import HomeAssistant, ServiceCall

from .const import (
    DOMAIN,
    MOOD_LABELS,
)

_LOGGER = logging.getLogger(__name__)

SERVICE_HALVE_BPM = "halve_bpm"
SERVICE_DOUBLE_BPM = "double_bpm"
SERVICE_SET_BPM = "set_bpm"
SERVICE_SET_MOOD = "set_mood"
SERVICE_CLEAR_OVERRIDE = "clear_override"
SERVICE_CLEAR_CACHE = "clear_cache"
SERVICE_CLEAR_OVERRIDES = "clear_overrides"

# No parameters: the correction services act on the (single) configured
# entry's media player and its current track. Multi-entry installs get
# one set of services per entry via the entity/device target below.
SERVICE_SCHEMA = vol.Schema({})

# set_bpm: exact tempo. Range covers all plausible music tempos; values
# outside are user error (a 20–300 BPM window is generous).
SET_BPM_SCHEMA = vol.Schema({
    vol.Required("bpm"): vol.All(vol.Coerce(float), vol.Range(min=20, max=300)),
})

# set_mood: one of the five mood heads.
SET_MOOD_SCHEMA = vol.Schema({
    vol.Required("mood"): vol.In(MOOD_LABELS),
})

ATTR_MEDIA_PLAYER = "media_player"


def _entries(hass: HomeAssistant) -> list:
    """All loaded AX BPM config entries."""
    return list(hass.config_entries.async_entries(DOMAIN))


def _current_track(hass: HomeAssistant, media_player_id: str | None):
    """Read (artist, title, duration) from the media player state.

    Returns None when the player is missing or not playing.
    """
    if not media_player_id:
        return None
    state = hass.states.get(media_player_id)
    if state is None or state.state != STATE_PLAYING:
        return None
    artist = state.attributes.get("media_artist")
    title = state.attributes.get("media_title")
    if not artist or not title:
        return None
    return artist, title, state.attributes.get("media_duration")


def _republish(hass: HomeAssistant, media_player_id: str | None) -> None:
    """Ask every AX BPM sensor to re-resolve the current track now.

    The sensor listens for media_player state changes; nudging it with a
    synthetic state-changed event is fragile, so instead the sensor
    exposes a refresh callback registered in hass.data (set by sensor.py
    on async_added_to_hass).
    """
    for entry in _entries(hass):
        data = hass.data.get(DOMAIN, {}).get(entry.entry_id, {})
        refresh = data.get("refresh_sensor")
        if refresh:
            refresh()


async def async_setup_services(hass: HomeAssistant) -> None:
    """Register the AX BPM services (idempotent)."""
    if hass.services.has_service(DOMAIN, SERVICE_HALVE_BPM):
        return

    async def _correct(call: ServiceCall, factor: float) -> None:
        for entry in _entries(hass):
            data = hass.data.get(DOMAIN, {}).get(entry.entry_id)
            if not data:
                continue
            pipeline = data["pipeline"]
            media_player_id = data.get("media_player")
            track = _current_track(hass, media_player_id)
            if not track:
                _LOGGER.warning(
                    "AX BPM %s: media player %s is not playing a track "
                    "with artist/title metadata — nothing to correct",
                    call.service, media_player_id,
                )
                continue
            artist, title, _duration = track

            # The correction applies to the currently PUBLISHED value.
            # Read it from the sensor entity via the registered getter.
            get_bpm = data.get("get_sensor_bpm")
            current = get_bpm() if get_bpm else None
            if not current or current <= 0:
                _LOGGER.warning(
                    "AX BPM %s: no BPM published for %s - %s — "
                    "nothing to correct (unknown is never corrected)",
                    call.service, artist, title,
                )
                continue

            corrected = await pipeline.async_apply_correction(
                artist, title, float(current), factor
            )
            _LOGGER.info(
                "AX BPM %s: %s - %s %.1f → %.1f BPM",
                call.service, artist, title, current, corrected,
            )
        _republish(hass, None)

    async def _halve(call: ServiceCall) -> None:
        await _correct(call, 0.5)

    async def _double(call: ServiceCall) -> None:
        await _correct(call, 2.0)

    async def _set_bpm(call: ServiceCall) -> None:
        bpm = float(call.data["bpm"])
        for entry in _entries(hass):
            data = hass.data.get(DOMAIN, {}).get(entry.entry_id)
            if not data:
                continue
            pipeline = data["pipeline"]
            media_player_id = data.get("media_player")
            track = _current_track(hass, media_player_id)
            if not track:
                _LOGGER.warning(
                    "AX BPM set_bpm: media player %s is not playing a "
                    "track with artist/title metadata — nothing to set",
                    media_player_id,
                )
                continue
            artist, title, _duration = track
            await pipeline.async_set_bpm(artist, title, bpm)
            _LOGGER.info(
                "AX BPM set_bpm: %s - %s → %.1f BPM (manual_set)",
                artist, title, bpm,
            )
        _republish(hass, None)

    async def _set_mood(call: ServiceCall) -> None:
        label = call.data["mood"]
        for entry in _entries(hass):
            data = hass.data.get(DOMAIN, {}).get(entry.entry_id)
            if not data:
                continue
            pipeline = data["pipeline"]
            media_player_id = data.get("media_player")
            track = _current_track(hass, media_player_id)
            if not track:
                _LOGGER.warning(
                    "AX BPM set_mood: media player %s is not playing a "
                    "track with artist/title metadata — nothing to set",
                    media_player_id,
                )
                continue
            artist, title, _duration = track
            await pipeline.async_set_mood(artist, title, label)
            _LOGGER.info(
                "AX BPM set_mood: %s - %s → %s", artist, title, label
            )
        _republish(hass, None)

    async def _clear_override(call: ServiceCall) -> None:
        for entry in _entries(hass):
            data = hass.data.get(DOMAIN, {}).get(entry.entry_id)
            if not data:
                continue
            pipeline = data["pipeline"]
            media_player_id = data.get("media_player")
            track = _current_track(hass, media_player_id)
            if not track:
                _LOGGER.warning(
                    "AX BPM clear_override: media player %s is not "
                    "playing — nothing to clear", media_player_id,
                )
                continue
            artist, title, _duration = track
            removed = await pipeline.async_clear_override(artist, title)
            if removed:
                _LOGGER.info(
                    "AX BPM clear_override: removed override for %s - %s",
                    artist, title,
                )
            else:
                _LOGGER.info(
                    "AX BPM clear_override: no override existed for %s - %s",
                    artist, title,
                )
        _republish(hass, None)

    async def _clear_cache(call: ServiceCall) -> None:
        for entry in _entries(hass):
            data = hass.data.get(DOMAIN, {}).get(entry.entry_id)
            if not data:
                continue
            removed = await data["pipeline"].async_clear_cache()
            _LOGGER.info("AX BPM clear_cache: removed %d entr(ies)", removed)
        _republish(hass, None)

    async def _clear_overrides(call: ServiceCall) -> None:
        for entry in _entries(hass):
            data = hass.data.get(DOMAIN, {}).get(entry.entry_id)
            if not data:
                continue
            removed = await data["pipeline"].async_clear_overrides()
            _LOGGER.info(
                "AX BPM clear_overrides: removed %d override(s)", removed
            )
        _republish(hass, None)

    hass.services.async_register(
        DOMAIN, SERVICE_HALVE_BPM, _halve, schema=SERVICE_SCHEMA
    )
    hass.services.async_register(
        DOMAIN, SERVICE_DOUBLE_BPM, _double, schema=SERVICE_SCHEMA
    )
    hass.services.async_register(
        DOMAIN, SERVICE_SET_BPM, _set_bpm, schema=SET_BPM_SCHEMA
    )
    hass.services.async_register(
        DOMAIN, SERVICE_SET_MOOD, _set_mood, schema=SET_MOOD_SCHEMA
    )
    hass.services.async_register(
        DOMAIN, SERVICE_CLEAR_OVERRIDE, _clear_override, schema=SERVICE_SCHEMA
    )
    hass.services.async_register(
        DOMAIN, SERVICE_CLEAR_CACHE, _clear_cache, schema=SERVICE_SCHEMA
    )
    hass.services.async_register(
        DOMAIN, SERVICE_CLEAR_OVERRIDES, _clear_overrides, schema=SERVICE_SCHEMA
    )
    _LOGGER.debug("AX BPM services registered")
