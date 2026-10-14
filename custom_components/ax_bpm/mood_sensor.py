"""AX BPM mood sensor entity.

Dedicated entity for the currently playing track's mood — split out of
the BPM sensor's extra_state_attributes (3.3.0). The BPM sensor keeps
only BPM-related attributes; mood lives here:

- state: mood_label (aggressive/party/relaxed/electronic/acoustic) or
  unknown — never a mood_scores dict.
- attributes: mood_scores, intensity, calmness, mood_override (when
  user-set), last_updated.

Push-only (should_poll=False), driven by TWO inputs:

1. The `mood_update` hook (registered in hass.data[DOMAIN][entry_id] by
   async_added_to_hass): the BPM sensor calls it after every resolution
   with the track's mood attribute dict (or None on failure — clearing
   the entity, mirroring the defect-4 stale-attribute rule).
2. The same media_player state-change subscription the BPM sensor uses:
   stop/off/unavailable → unknown with attributes CLEARED; pause →
   retain the last mood (a paused track's mood is still valid).

The mood sensor never resolves anything itself — no analyzer calls, no
Deezer lookups. It is a pure display of the pipeline's mood output.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from homeassistant.components.sensor import SensorEntity
from homeassistant.const import STATE_OFF
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event

from .const import DOMAIN, NAME

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the AX BPM mood sensor from a config entry."""
    media_player_id = entry.data.get("media_player")
    if not media_player_id:
        return  # the BPM sensor already logged the error

    async_add_entities([AxBpmMoodSensor(entry, media_player_id)])


class AxBpmMoodSensor(SensorEntity):
    """Sensor holding the currently playing track's mood label."""

    _attr_should_poll = False
    _attr_icon = "mdi:emoticon"
    _attr_has_entity_name = True
    _attr_translation_key = "mood"
    _attr_device_class = None  # custom states; no HA device class

    def __init__(self, entry, media_player_id: str) -> None:
        self._entry = entry
        self._media_player_id = media_player_id
        self._attr_unique_id = f"{entry.entry_id}_mood"
        self._attr_native_value = None  # unknown until the first mood push
        self._attr_extra_state_attributes: dict = {}

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            identifiers={(DOMAIN, self._entry.entry_id)},
            name=NAME,
            entry_type=DeviceEntryType.SERVICE,
        )

    async def async_added_to_hass(self) -> None:
        """Subscribe to the media player + register the push hook."""
        self.async_on_remove(
            async_track_state_change_event(
                self.hass, [self._media_player_id], self._on_media_change
            )
        )
        # The BPM sensor pushes mood attrs here after every resolution.
        # Same hass.data[DOMAIN][entry_id] dict the BPM sensor's hooks
        # live in — indexing hass.data directly with the entry_id raised
        # a KeyError inside this hook in the 3.1.0 regression.
        entry_data = self.hass.data[DOMAIN][self._entry.entry_id]
        entry_data["mood_update"] = self._on_mood_update

    @callback
    def _on_mood_update(self, mood_attrs: dict | None) -> None:
        """Receive mood attributes from the BPM sensor's resolution.

        None → clear to unknown with empty attributes (failed resolution
        or a track with no mood data — never present stale mood).
        """
        if not mood_attrs:
            self._attr_native_value = None
            self._attr_extra_state_attributes = {}
        else:
            self._attr_native_value = mood_attrs.get("mood_label")
            self._attr_extra_state_attributes = {
                **mood_attrs,
                "last_updated": datetime.now(UTC).isoformat(),
            }
        self.async_write_ha_state()

    @callback
    def _on_media_change(self, event) -> None:
        new_state = event.data.get("new_state")
        if new_state is None:
            return

        state = new_state.state

        # Stop / off / unavailable → unknown, attributes cleared.
        if state in (STATE_OFF, "unavailable", "unknown"):
            self._attr_native_value = None
            self._attr_extra_state_attributes = {}
            self.async_write_ha_state()
            return

        # Pause → retain the last mood (mirrors the BPM sensor).
        # Playing → nothing: the mood arrives via the mood_update hook.

    @property
    def extra_state_attributes(self) -> dict:
        """mood_scores / intensity / calmness / mood_override."""
        return self._attr_extra_state_attributes
