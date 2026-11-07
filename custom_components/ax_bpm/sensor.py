"""AX BPM sensor entity.

Push-only (should_poll=False): reacts to media_player state changes.

- state: final BPM (float), unit "BPM", state_class "measurement"
- pause: retain last value; stop/off/unavailable: unknown
- failure: unknown — NEVER publish 0

Track identity (stale-BPM fix)
-----------------------------
Track changes are detected via `media.track_identity()`, which combines
`media_content_id` with the artist/title/duration tuple. A player that
reports the same metadata for consecutive tracks no longer freezes the
sensor — and because the same identity feeds the generation guard below,
a slow resolution from an earlier track can never publish under a later
one (previously a tuple-equality check that compared EQUAL and let it
through).

Two timers back the identity up:

- PUBLISH_GRACE (5 s): after a genuine change the entity HOLDS the
  previous track's value, then clears to `unknown` if nothing published
  for the new track. The consumer (AXiDOS card, `parseBpm`) derives 120
  from `unknown`, so the dance never stalls — and the integration never
  fabricates a numeric state, which would pollute long-term statistics
  and contradict the documented "failure is unknown, never 0" contract.
- WATCHDOG (media_duration + WATCHDOG_GRACE): fires when the expected
  track end has passed while the SAME identity is still reported. This is
  the only signal available when a source never changes ANY attribute —
  the exact "whole album shows track 1's BPM" failure. Arming requires a
  positive `media_duration`; a live stream with no duration keeps its
  BPM, because "should have ended" is unknowable there.

A LATE resolution still publishes: the grace timer only makes the entity
honest in the interim, it does not cancel the in-flight resolve.

BPM publishes first (unchanged path). On the Deezer-metadata path the
sensor then triggers the pipeline's post-publish mood enrichment; when
the analyzer responds, mood attributes arrive as a second
`async_write_ha_state`. The local-analysis path already carries mood
attributes in the first publish (analyzer tempo+mood in one call, or
NumPy floor with a concurrent mood call).

Mood attributes are NOT published on this entity (3.3.0 split): they
are forwarded to the dedicated mood sensor (mood_sensor.py) via the
`mood_update` hook in hass.data[DOMAIN][entry_id]. This entity keeps
only BPM-related attributes (bpm_raw, isrc, source, bpm_engine, …).
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

from .analyzer_status import AxBpmAnalyzerStatusSensor
from .const import (
    CONF_MEDIA_PLAYER,
    DOMAIN,
    NAME,
    PUBLISH_GRACE,
    TRACK_DEBOUNCE,
    UNIT_BPM,
    WATCHDOG_GRACE,
)
from .media import (
    looks_restarted,
    media_duration,
    media_position,
    track_identity,
)
from .mood_sensor import AxBpmMoodSensor
from .pipeline import BpmPipeline, split_mood_attrs

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

    entry_data = hass.data[DOMAIN][entry.entry_id]
    pipeline = entry_data["pipeline"]
    async_add_entities(
        [
            AxBpmSensor(entry, media_player_id, pipeline),
            AxBpmMoodSensor(entry, media_player_id),
            AxBpmAnalyzerStatusSensor(entry, entry_data["status_coordinator"]),
        ]
    )


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
        self._last_position: float | None = None
        # Monotonic generation: bumped on EVERY identity change (and on
        # every forced clear). It replaces tuple-equality for "is this
        # result still for the current track?" — tuple equality answered
        # "yes" for consecutive tracks with identical metadata and let a
        # stale result publish.
        self._generation = 0
        # Generation the currently published value belongs to. When the
        # grace timer fires and this differs from _generation, nothing has
        # published for the current track yet → clear to unknown.
        self._published_generation = -1
        self._resolving_generation: int | None = None
        self._debounce_unsub = None
        self._grace_unsub = None
        self._watchdog_unsub = None

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            identifiers={(DOMAIN, self._entry.entry_id)},
            name=NAME,
            entry_type=DeviceEntryType.SERVICE,
        )

    async def async_added_to_hass(self) -> None:
        """Subscribe to the media player."""
        self.async_on_remove(async_track_state_change_event(self.hass, [self._media_player_id], self._on_media_change))
        # Service hooks (services.py): immediate re-resolve after a
        # halve/double/clear mutation, and a getter for the currently
        # published BPM (the value a correction applies to).
        # Both hooks live in the SAME hass.data[DOMAIN][entry_id] dict
        # that async_setup_entry created — indexing hass.data directly
        # with the entry_id raised a KeyError inside this hook, which
        # aborted the entity add and left the sensor unavailable.
        entry_data = self.hass.data[DOMAIN][self._entry.entry_id]
        entry_data["refresh_sensor"] = self._refresh_now
        entry_data["get_sensor_bpm"] = self._get_published_bpm

    async def async_will_remove_from_hass(self) -> None:
        """Cancel every pending timer (no callbacks after removal)."""
        self._cancel_debounce()
        self._cancel_grace()
        self._cancel_watchdog()

    @callback
    def _refresh_now(self) -> None:
        """Re-resolve the current track immediately (no debounce).

        Called by the services after a store mutation so the correction
        is visible at once. Skips when nothing is playing — the next
        real track change resolves normally.
        """
        state = self.hass.states.get(self._media_player_id)
        if state is None or state.state != STATE_PLAYING:
            return
        # Force the next resolution to run even if the track identity is
        # unchanged (the store changed, not the track). The generation is
        # NOT bumped: this is the same track, so the result is still valid
        # for it and must not be discarded by the generation guard.
        self._last_track = None
        self._published_generation = -1
        self._cancel_debounce()
        self._cancel_grace()
        self.hass.async_create_task(self._async_resolve())

    def _get_published_bpm(self) -> float | None:
        """The currently published BPM (None when unknown)."""
        return self._attr_native_value

    # ------------------------------------------------------------------
    # Timers
    # ------------------------------------------------------------------

    def _cancel_debounce(self) -> None:
        if self._debounce_unsub:
            self._debounce_unsub()
            self._debounce_unsub = None

    def _cancel_grace(self) -> None:
        if self._grace_unsub:
            self._grace_unsub()
            self._grace_unsub = None

    def _cancel_watchdog(self) -> None:
        if self._watchdog_unsub:
            self._watchdog_unsub()
            self._watchdog_unsub = None

    def _arm_grace(self) -> None:
        """Hold the previous value, then clear to unknown if nothing lands.

        PUBLISH_GRACE is deliberately shorter than a worst-case resolution
        (OVERALL_BUDGET + TRACK_DEBOUNCE). A resolution that finishes
        after the grace still publishes normally — the timer only governs
        the INTERIM state, so a slow-but-successful track shows unknown
        briefly instead of another track's BPM.
        """
        self._cancel_grace()
        generation = self._generation
        self._grace_unsub = async_call_later(
            self.hass,
            PUBLISH_GRACE,
            lambda _now: self._on_grace_expired(generation),
        )

    @callback
    def _on_grace_expired(self, generation: int) -> None:
        self._grace_unsub = None
        # A newer track superseded this grace window — its own timer owns
        # the entity now.
        if generation != self._generation:
            return
        # A publish already landed for this generation → nothing to clear.
        if self._published_generation == generation:
            return
        if self._attr_native_value is None:
            return
        _LOGGER.info(
            "AX BPM: no resolution for the current track after %.0fs — "
            "clearing to unknown (the card derives 120 until it resolves)",
            PUBLISH_GRACE,
        )
        self._clear_to_unknown()

    def _arm_watchdog(self) -> None:
        """Clear a value that outlived the track it belongs to.

        Armed per successful publish at `media_duration + WATCHDOG_GRACE`.
        This is the ONLY mechanism that can catch a source which never
        changes any attribute: the identity never moves, so no debounce
        ever fires, and without this the previous track's BPM would sit
        there for the whole album.

        Not armed when the duration is unknown (live streams).
        """
        self._cancel_watchdog()
        state = self.hass.states.get(self._media_player_id)
        duration = media_duration(state)
        if duration is None:
            return  # no expected end → holding the value is correct
        position = media_position(state) or 0.0
        delay = max(0.0, duration - position) + WATCHDOG_GRACE
        generation = self._generation
        self._watchdog_unsub = async_call_later(
            self.hass,
            delay,
            lambda _now: self._on_watchdog_expired(generation),
        )

    @callback
    def _on_watchdog_expired(self, generation: int) -> None:
        self._watchdog_unsub = None
        state = self.hass.states.get(self._media_player_id)
        # Paused / stopped: "should have ended" is meaningless while the
        # position is frozen. The off/unknown handler owns that case.
        if state is None or state.state != STATE_PLAYING:
            return
        # A different track is playing now; its own publish re-armed this.
        if generation != self._generation:
            return
        _LOGGER.info(
            "AX BPM: track outlived its expected duration (%.0f BPM held) — "
            "clearing to unknown until the player reports a new track",
            self._attr_native_value or 0.0,
        )
        self._clear_to_unknown()

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    @callback
    def _clear_to_unknown(self) -> None:
        """Publish unknown with per-track attributes cleared.

        Deliberately does NOT touch `_last_track`: clearing it made every
        subsequent attribute update (volume, position, …) re-trigger a
        full resolution — the request storm documented in
        `_async_resolve_track`. The generation already prevents a stale
        result from republishing.
        """
        self._attr_native_value = None
        self._attr_extra_state_attributes = {}
        self._push_mood(None)
        self.async_write_ha_state()

    @callback
    def _on_media_change(self, event) -> None:
        new_state = event.data.get("new_state")
        if new_state is None:
            return

        state = new_state.state

        # Stop / off / unavailable → unknown (both entities).
        if state in (STATE_OFF, "unavailable", "unknown"):
            self._cancel_debounce()
            self._cancel_grace()
            self._cancel_watchdog()
            self._attr_native_value = None
            self._last_track = None
            self._published_generation = -1
            self._push_mood(None)
            self.async_write_ha_state()
            return

        # Pause → retain last value. The position stops advancing, so a
        # pending watchdog would fire on a frozen clock — cancel it and
        # re-arm on resume (the next playing event re-arms via publish).
        if state != STATE_PLAYING:
            self._cancel_debounce()
            self._cancel_watchdog()
            return

        identity = track_identity(new_state)
        position = media_position(new_state)

        # Same metadata, but playback clearly started over → the source
        # recycles ids/metadata across tracks (the album bug). Treat it as
        # a genuine change so the resolution runs and the new track can
        # publish its own value.
        restarted = looks_restarted(self._last_position, position)
        self._last_position = position

        if identity == self._last_track and not restarted:
            return

        self._last_track = identity
        self._generation += 1
        _LOGGER.debug(
            "AX BPM: track identity changed (generation %d, restarted=%s)",
            self._generation,
            restarted,
        )

        # Hold the previous value for PUBLISH_GRACE so a normal 3–5 s
        # resolution never flashes unknown between songs.
        self._arm_grace()

        # Debounce a few seconds after playback starts / track changes.
        self._cancel_debounce()
        self._debounce_unsub = async_call_later(self.hass, TRACK_DEBOUNCE, self._async_resolve_track)

    @callback
    def _async_resolve_track(self, _now) -> None:
        self._debounce_unsub = None
        # NOTE: _last_track is intentionally NOT cleared here. Clearing it
        # made every subsequent attribute update (volume, position, …)
        # re-trigger a full Deezer resolution for a failing track — a
        # request storm. Late-settling metadata still re-triggers naturally
        # because the track identity changes.
        self.hass.async_create_task(self._async_resolve())

    async def _async_resolve(self) -> None:
        state = self.hass.states.get(self._media_player_id)
        if state is None or state.state != STATE_PLAYING:
            return
        artist = state.attributes.get("media_artist")
        title = state.attributes.get("media_title")
        duration = state.attributes.get("media_duration")
        generation = self._generation
        self._resolving_generation = generation

        try:
            result = await self._pipeline.async_resolve(artist, title, duration)
        except Exception:
            _LOGGER.exception("AX BPM resolution failed")
            result = None

        # The track changed while resolving → discard the stale result.
        # Compared by GENERATION, not by identity: two consecutive tracks
        # with identical metadata compared equal and let the old result
        # publish under the new track.
        if generation != self._generation:
            _LOGGER.debug(
                "AX BPM: discarding result for generation %d (current %d)",
                generation,
                self._generation,
            )
            return
        self._resolving_generation = None

        if result is None:
            # Failure → unknown, NEVER 0. Retries on the next real track
            # change (not on every attribute update). Defect 4 fix: the
            # PREVIOUS track's per-track attributes (mood_scores,
            # mood_label, isrc, bpm_raw, …) must NOT survive on an
            # unknown state — clear them so the entity never presents
            # stale valid-looking metadata alongside unknown.
            self._cancel_grace()  # this resolution has spoken
            self._attr_native_value = None
            self._attr_extra_state_attributes = {}
            self._push_mood(None)
            self.async_write_ha_state()
            return

        # Mood attrs split off at publish time (3.3.0): the BPM sensor
        # keeps only BPM-related attributes; mood goes to the dedicated
        # mood sensor via the mood_update hook.
        bpm_attrs, mood_attrs = split_mood_attrs(result.attrs)

        # BPM-first publish (unchanged path).
        self._attr_native_value = result.bpm
        self._attr_extra_state_attributes = {
            **bpm_attrs,
            "last_updated": datetime.now(UTC).isoformat(),
        }
        self._published_generation = generation
        self._cancel_grace()
        self._arm_watchdog()
        self._push_mood(mood_attrs)
        self.async_write_ha_state()

        # Post-publish mood enrichment (Deezer-metadata path only; the
        # local path already carries mood attrs). A second state write
        # adds the mood attributes when the analyzer responds. Never
        # delays or blocks the BPM publish above.
        try:
            enrich_attrs = await self._pipeline.async_enrich(artist, title, duration, result)
        except Exception:
            _LOGGER.debug("AX BPM mood enrichment failed", exc_info=True)
            enrich_attrs = None
        if not enrich_attrs:
            return
        # The track changed while enriching → discard the stale mood.
        if generation != self._generation:
            return
        self._push_mood(enrich_attrs)
        self.async_write_ha_state()

    def _push_mood(self, mood_attrs: dict | None) -> None:
        """Forward mood attrs to the mood sensor (no-op when absent).

        The mood sensor registers the hook in async_added_to_hass; until
        then (or when it was removed) the push is silently dropped — the
        next resolution re-publishes anyway.
        """
        entry = getattr(self, "_entry", None)
        if entry is None:
            return
        entry_data = self.hass.data.get(DOMAIN, {}).get(entry.entry_id)
        if not entry_data:
            return
        mood_update = entry_data.get("mood_update")
        if mood_update:
            mood_update(mood_attrs)
