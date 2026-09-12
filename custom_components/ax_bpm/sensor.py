"""AX BPM sensor entity.

Push-only (should_poll=False): reacts to media_player state changes.
- state: final BPM (float), unit "BPM", state_class "measurement"
- pause: retain last value; stop/off/unavailable: unknown
- failure: unknown — NEVER publish 0

Phase 1: BPM publishes first (unchanged path). On the Deezer-metadata
path the sensor then triggers the pipeline's post-publish mood
enrichment; when the sidecar responds, mood attributes arrive as a
second `async_write_ha_state`. The local-analysis path already carries
mood attributes in the first publish (sidecar ran concurrently with
aubio).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from homeassistant.components.sensor import (
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import STATE_OFF, STATE_PLAYING
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
        self._resolving_track: tuple | None = None
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
        # NOTE: _last_track is intentionally NOT cleared here. Clearing it
        # made every subsequent attribute update (volume, position, …)
        # re-trigger a full Deezer resolution for a failing track — a
        # request storm. Late-settling metadata still re-triggers naturally
        # because the (artist, title, duration) tuple changes.
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
        track_id = (artist, title, duration)
        self._resolving_track = track_id

        try:
            result = await self._pipeline.async_resolve(
                artist, title, duration
            )
        except Exception:
            _LOGGER.exception("AX BPM resolution failed")
            result = None

        # The track changed while resolving → discard the stale result.
        # The new track's own resolution will publish instead.
        if self._resolving_track != self._current_track_id():
            return
        self._resolving_track = None

        if result is None:
            # Failure → unknown, NEVER 0. Retries on the next real track
            # change (not on every attribute update).
            self._attr_native_value = None
            self.async_write_ha_state()
            return

        # BPM-first publish (unchanged path).
        self._attr_native_value = result.bpm
        self._attr_extra_state_attributes = {
            **result.attrs,
            "last_updated": datetime.now(UTC).isoformat(),
        }
        self.async_write_ha_state()

        # Post-publish mood enrichment (Deezer-metadata path only; the
        # local path already carries mood attrs). A second state write
        # adds the mood attributes when the sidecar responds. Never
        # delays or blocks the BPM publish above.
        try:
            mood_attrs = await self._pipeline.async_enrich(
                artist, title, duration, result
            )
        except Exception:
            _LOGGER.debug("AX BPM mood enrichment failed", exc_info=True)
            mood_attrs = None
        if not mood_attrs:
            return
        # The track changed while enriching → discard the stale mood.
        if self._current_track_id() != track_id:
            return
        self._attr_extra_state_attributes = {
            **self._attr_extra_state_attributes,
            **mood_attrs,
            "last_updated": datetime.now(UTC).isoformat(),
        }
        self.async_write_ha_state()

    def _current_track_id(self) -> tuple | None:
        state = self.hass.states.get(self._media_player_id)
        if state is None:
            return None
        return (
            state.attributes.get("media_artist"),
            state.attributes.get("media_title"),
            state.attributes.get("media_duration"),
        )