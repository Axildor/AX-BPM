"""AX BPM sensor entity.

Push-only (should_poll=False): reacts to media_player state changes.
- state: final BPM (float), unit "BPM", state_class "measurement"
- pause: retain last value; stop/off/unavailable: unknown
- failure: unknown — NEVER publish 0
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import STATE_OFF, STATE_ON, STATE_PLAYING
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_call_later, async_track_state_change_event

from .const import (
    CONF_MEDIA_PLAYER,
    DOMAIN,
    NAME,
    TRACK_DEBOUNCE,
    UNIT_BPM,
)
from .pipeline import BpmPipeline

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the AX BPM sensor from a config entry."""
    media_player_id = entry.data.get(CONF_MEDIA_PLAYER)
    if not media_player_id:
        _LOGGER.error("No media player configured for AX BPM")
        return

    pipeline = hass.data[DOMAIN][entry.entry_id]["pipeline"]
    async_add_entities([AxBpmSensor(entry, media_player_id, pipeline)])


class AxBpmSensor(SensorEntity):
    """Sensor holding the currently playing track's tempo."""

    _attr_should_poll = False
    _attr_icon = "mdi:metronome"
    _attr_native_unit_of_measurement = UNIT_BPM
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_device_class = None  # BPM has no HA device class; keep it custom

    def __init__(self, entry, media_player_id: str, pipeline: BpmPipeline) -> None:
        self._entry = entry
        self._media_player_id = media_player_id
        self._pipeline = pipeline
        self._attr_unique_id = f"{entry.entry_id}_bpm"
        self._attr_name = NAME
        self._attr_native_value = None  # unknown until first track resolves
        self._attr_extra_state_attributes: dict = {}
        self._last_track: tuple | None = None
        self._debounce_unsub = None

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            identifiers={(DOMAIN, self._entry.entry_id)},
            name=NAME,
            entry_type=DeviceEntryType.SERVICE,
        )

    async def async_added_to_hass(self) -> None:
        """Subscribe to the media player."""
        self.async_on_remove(
            async_track_state_change_event(
                self.hass, [self._media_player_id], self._on_media_change
            )
        )

    @callback
    def _on_media_change(self, event) -> None:
        new_state = event.data.get("new_state")
        if new_state is None:
            return

        state = new_state.state

        # Stop / off / unavailable → unknown.
        if state in (STATE_OFF, "unavailable", "unknown"):
            self._cancel_debounce()
            self._attr_native_value = None
            self._last_track = None
            self.async_write_ha_state()
            return

        # Pause → retain last value.
        if state != STATE_PLAYING:
            self._cancel_debounce()
            return

        artist = new_state.attributes.get("media_artist")
        title = new_state.attributes.get("media_title")
        duration = new_state.attributes.get("media_duration")
        track_id = (artist, title, duration)

        if track_id == self._last_track:
            return
        self._last_track = track_id

        # Debounce a few seconds after playback starts / track changes.
        self._cancel_debounce()
        self._debounce_unsub = async_call_later(
            self.hass, TRACK_DEBOUNCE, self._async_resolve_track
        )

    @callback
    def _async_resolve_track(self, _now) -> None:
        self._debounce_unsub = None
        self._last_track = None  # allow re-trigger if metadata settles late
        self.hass.async_create_task(self._async_resolve())

    def _cancel_debounce(self) -> None:
        if self._debounce_unsub:
            self._debounce_unsub()
            self._debounce_unsub = None

    async def _async_resolve(self) -> None:
        state = self.hass.states.get(self._media_player_id)
        if state is None or state.state != STATE_PLAYING:
            return
        artist = state.attributes.get("media_artist")
        title = state.attributes.get("media_title")
        duration = state.attributes.get("media_duration")

        try:
            result = await self._pipeline.async_resolve(
                artist, title, duration
            )
        except Exception:  # noqa: BLE001 — a failure must never crash HA
            _LOGGER.exception("AX BPM resolution failed")
            result = None

        if result is None:
            # Failure → unknown, NEVER 0. Retry on next track change.
            self._attr_native_value = None
        else:
            self._attr_native_value = result.bpm
            self._attr_extra_state_attributes = {
                **result.attrs,
                "last_updated": datetime.now(UTC).isoformat(),
            }
        self.async_write_ha_state()