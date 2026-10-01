"""Regression tests for the sensor + button entity setup paths.

Covers the 3.1.0 breakage reported by the user:

- sensor.py async_added_to_hass indexed hass.data directly with the
  entry_id (missing the DOMAIN level) → KeyError inside the hook →
  HA aborted the entity add → sensor.ax_bpm unavailable. The fix
  registers BOTH service hooks in the hass.data[DOMAIN][entry_id] dict
  that async_setup_entry created.
- button.py set _attr_translation_key without _attr_has_entity_name —
  HA core only resolves entity.<platform>.<key>.name translations for
  entities that opt into entity-name mode (entity.py _name_internal),
  so the four buttons rendered with no friendly name.
"""

from __future__ import annotations

import pytest

from ax_bpm.button import SERVICE_BY_BUTTON, AxBpmButton
from ax_bpm.const import DOMAIN
from ax_bpm.sensor import AxBpmSensor

ENTRY_ID = "01M3T3Z3VYVXYKYZJP7PP1NRBX"


class _FakeServices:
    """Records async_call invocations."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    async def async_call(self, domain, service, service_data=None, blocking=False):
        self.calls.append((domain, service, service_data or {}))


class _FakeState:
    def __init__(self, state: str, attributes: dict | None = None) -> None:
        self.state = state
        self.attributes = attributes or {}


class _FakeStates:
    def __init__(self, state: _FakeState | None) -> None:
        self._state = state

    def get(self, entity_id: str) -> _FakeState | None:
        return self._state


class _FakeHass:
    """Minimal hass for entity hook registration (no HA runtime)."""

    def __init__(self, playing: bool = True) -> None:
        self.data = {DOMAIN: {ENTRY_ID: {}}}
        self.services = _FakeServices()
        self.states = _FakeStates(
            _FakeState(
                "playing" if playing else "off",
                {
                    "media_artist": "Artist",
                    "media_title": "Title",
                    "media_duration": 180,
                },
            )
        )
        self.created_tasks: list = []

    def async_create_task(self, coro):
        self.created_tasks.append(coro)
        return coro


class _FakeEntry:
    def __init__(self) -> None:
        self.entry_id = ENTRY_ID
        self.data = {"media_player": "media_player.test"}


# ---------------------------------------------------------------------------
# Sensor — hook registration (the KeyError regression)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sensor_added_to_hass_registers_both_hooks():
    """Both service hooks land in hass.data[DOMAIN][entry_id] — no KeyError.

    Regression: the get_sensor_bpm hook was assigned to
    hass.data[entry_id] (missing the DOMAIN level), raising KeyError
    inside async_added_to_hass → entity add aborted → sensor unavailable.
    """
    hass = _FakeHass()
    entry = _FakeEntry()
    sensor = AxBpmSensor(entry, "media_player.test", pipeline=object())
    sensor.hass = hass

    # Must not raise — the old code raised KeyError here.
    await sensor.async_added_to_hass()

    entry_data = hass.data[DOMAIN][ENTRY_ID]
    assert entry_data["refresh_sensor"] == sensor._refresh_now
    assert entry_data["get_sensor_bpm"] == sensor._get_published_bpm


@pytest.mark.asyncio
async def test_sensor_hooks_are_functional():
    """The registered hooks actually work end-to-end.

    get_sensor_bpm returns the published value; refresh_sensor clears the
    track cache and schedules a re-resolve when the player is playing.
    """
    hass = _FakeHass(playing=True)
    entry = _FakeEntry()
    sensor = AxBpmSensor(entry, "media_player.test", pipeline=object())
    sensor.hass = hass
    await sensor.async_added_to_hass()

    entry_data = hass.data[DOMAIN][ENTRY_ID]

    # get_sensor_bpm mirrors the published native value.
    sensor._attr_native_value = 123.5
    assert entry_data["get_sensor_bpm"]() == 123.5

    # refresh_sensor: playing → clears _last_track + schedules a resolve.
    sensor._last_track = ("Artist", "Title", 180)
    entry_data["refresh_sensor"]()
    assert sensor._last_track is None
    assert len(hass.created_tasks) == 1
    # Drain the scheduled resolve coroutine (pipeline is a bare object —
    # async_resolve would fail; close it to avoid an un-awaited warning).
    hass.created_tasks[0].close()


@pytest.mark.asyncio
async def test_sensor_refresh_noop_when_not_playing():
    """refresh_sensor is a no-op when the player is not playing."""
    hass = _FakeHass(playing=False)
    entry = _FakeEntry()
    sensor = AxBpmSensor(entry, "media_player.test", pipeline=object())
    sensor.hass = hass
    await sensor.async_added_to_hass()

    hass.data[DOMAIN][ENTRY_ID]["refresh_sensor"]()
    assert not hass.created_tasks


# ---------------------------------------------------------------------------
# Buttons — translation-key naming (the missing-label regression)
# ---------------------------------------------------------------------------


def test_button_opts_into_entity_name_mode():
    """Buttons must set has_entity_name for translation-key names.

    Regression: _attr_translation_key alone is ignored by HA core's
    _name_internal unless the entity opts into entity-name mode — the
    buttons rendered with no friendly name at all.
    """
    entry = _FakeEntry()
    for action in ("halve", "double", "clear_cache", "clear_overrides"):
        button = AxBpmButton(entry, action)
        assert button._attr_has_entity_name is True, action
        assert button._attr_translation_key == f"btn_{action}"
        assert button._attr_unique_id == f"{ENTRY_ID}_btn_{action}"


def test_button_service_mapping_is_complete():
    """Every button action maps to a registered ax_bpm service."""
    assert SERVICE_BY_BUTTON == {
        "halve": "halve_bpm",
        "double": "double_bpm",
        "clear_cache": "clear_cache",
        "clear_overrides": "clear_overrides",
    }


@pytest.mark.asyncio
async def test_button_press_calls_wrapped_service():
    """A button press routes to the corresponding service (single path)."""
    hass = _FakeHass()
    entry = _FakeEntry()
    button = AxBpmButton(entry, "halve")
    button.hass = hass

    await button.async_press()

    assert hass.services.calls == [(DOMAIN, "halve_bpm", {})]
