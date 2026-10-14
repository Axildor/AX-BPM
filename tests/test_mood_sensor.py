"""Tests for the dedicated mood sensor (3.3.0 split).

Covers:
- split_mood_attrs partitioning (mood keys out, None-valued mood keys
  dropped, BPM keys retained).
- AxBpmMoodSensor: state mirrors mood_label, attributes carry
  mood_scores/intensity/calmness/mood_override, hook registration in
  hass.data[DOMAIN][entry_id], None push → unknown + cleared attrs,
  stop/off → unknown + cleared, pause → retain.
- AxBpmSensor publish path: BPM attrs exclude mood keys, mood attrs are
  forwarded to the mood_update hook, failure → hook(None).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ax_bpm.const import DOMAIN
from ax_bpm.mood_sensor import AxBpmMoodSensor
from ax_bpm.pipeline import ResolutionResult, split_mood_attrs
from ax_bpm.sensor import AxBpmSensor

ENTRY_ID = "01M3T3Z3VYVXYKYZJP7PP1NRBX"


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
    def __init__(self, playing: str = "playing") -> None:
        self.data = {DOMAIN: {ENTRY_ID: {}}}
        self.states = _FakeStates(
            _FakeState(
                playing,
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


def _make_mood_sensor(hass: _FakeHass) -> AxBpmMoodSensor:
    sensor = AxBpmMoodSensor(_FakeEntry(), "media_player.test")
    sensor.hass = hass
    sensor.async_write_ha_state = MagicMock()
    return sensor


# ---------------------------------------------------------------------------
# split_mood_attrs
# ---------------------------------------------------------------------------


def test_split_mood_attrs_partitions():
    """Mood keys move out; BPM keys stay; None mood values are dropped."""
    attrs = {
        "source": "analyzer",
        "bpm_raw": 128.0,
        "isrc": "GBDUW0000059",
        "mood_scores": {"aggressive": 0.9, "party": 0.1},
        "mood_label": "aggressive",
        "intensity": 0.5,
        "calmness": 0.2,
        "mood_override": True,
    }
    bpm, mood = split_mood_attrs(attrs)
    assert bpm == {
        "source": "analyzer",
        "bpm_raw": 128.0,
        "isrc": "GBDUW0000059",
    }
    assert mood == {
        "mood_scores": {"aggressive": 0.9, "party": 0.1},
        "mood_label": "aggressive",
        "intensity": 0.5,
        "calmness": 0.2,
        "mood_override": True,
    }


def test_split_mood_attrs_drops_none_mood_values():
    """None-valued mood placeholders (Deezer path) never reach the entity."""
    bpm, mood = split_mood_attrs(
        {"mood_scores": None, "mood_label": None, "intensity": None,
         "calmness": None, "isrc": "X"}
    )
    assert mood == {}
    assert bpm == {"isrc": "X"}


def test_split_mood_attrs_empty():
    assert split_mood_attrs({}) == ({}, {})


# ---------------------------------------------------------------------------
# AxBpmMoodSensor — hook + state mirror
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mood_sensor_registers_hook():
    """The mood_update hook lands in hass.data[DOMAIN][entry_id]."""
    hass = _FakeHass()
    sensor = _make_mood_sensor(hass)
    await sensor.async_added_to_hass()
    assert hass.data[DOMAIN][ENTRY_ID]["mood_update"] == sensor._on_mood_update


@pytest.mark.asyncio
async def test_mood_sensor_mirrors_label_and_attrs():
    """State = mood_label; attrs carry the mood dict + last_updated."""
    hass = _FakeHass()
    sensor = _make_mood_sensor(hass)
    await sensor.async_added_to_hass()

    sensor._on_mood_update(
        {
            "mood_scores": {"aggressive": 1.0},
            "mood_label": "aggressive",
            "intensity": 1.0,
            "calmness": 0.0,
            "mood_override": True,
        }
    )
    assert sensor._attr_native_value == "aggressive"
    attrs = sensor._attr_extra_state_attributes
    assert attrs["mood_scores"] == {"aggressive": 1.0}
    assert attrs["intensity"] == 1.0
    assert attrs["calmness"] == 0.0
    assert attrs["mood_override"] is True
    assert "last_updated" in attrs


@pytest.mark.asyncio
async def test_mood_sensor_none_push_clears():
    """A None push → unknown with EMPTY attributes (defect-4 mirror)."""
    hass = _FakeHass()
    sensor = _make_mood_sensor(hass)
    await sensor.async_added_to_hass()

    sensor._on_mood_update({"mood_label": "party", "intensity": 0.5})
    assert sensor._attr_native_value == "party"

    sensor._on_mood_update(None)
    assert sensor._attr_native_value is None
    assert sensor._attr_extra_state_attributes == {}


@pytest.mark.asyncio
async def test_mood_sensor_entity_metadata():
    """Entity-name mode + translation key + unique_id + device info."""
    sensor = AxBpmMoodSensor(_FakeEntry(), "media_player.test")
    assert sensor._attr_has_entity_name is True
    assert sensor._attr_translation_key == "mood"
    assert sensor._attr_unique_id == f"{ENTRY_ID}_mood"
    assert sensor._attr_should_poll is False
    assert sensor.device_info.identifiers == {(DOMAIN, ENTRY_ID)}


# ---------------------------------------------------------------------------
# AxBpmMoodSensor — media_player lifecycle
# ---------------------------------------------------------------------------


def _media_event(state: str) -> MagicMock:
    event = MagicMock()
    event.data = {"new_state": _FakeState(state)}
    return event


@pytest.mark.asyncio
async def test_mood_sensor_stop_clears():
    hass = _FakeHass()
    sensor = _make_mood_sensor(hass)
    await sensor.async_added_to_hass()
    sensor._on_mood_update({"mood_label": "relaxed"})

    sensor._on_media_change(_media_event("off"))
    assert sensor._attr_native_value is None
    assert sensor._attr_extra_state_attributes == {}


@pytest.mark.asyncio
async def test_mood_sensor_pause_retains():
    hass = _FakeHass()
    sensor = _make_mood_sensor(hass)
    await sensor.async_added_to_hass()
    sensor._on_mood_update({"mood_label": "relaxed"})

    sensor._on_media_change(_media_event("paused"))
    assert sensor._attr_native_value == "relaxed"
    assert sensor._attr_extra_state_attributes["mood_label"] == "relaxed"


# ---------------------------------------------------------------------------
# AxBpmSensor — publish path forwards mood, keeps BPM attrs clean
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bpm_sensor_publishes_bpm_only_attrs_and_pushes_mood():
    """BPM attrs exclude mood keys; mood attrs go to the mood_update hook."""
    hass = _FakeHass()
    entry = _FakeEntry()
    result = ResolutionResult(
        128.0,
        source="analyzer",
        bpm_raw=128.0,
        isrc="GBDUW0000059",
        mood_scores={"aggressive": 0.9},
        mood_label="aggressive",
        intensity=0.5,
        calmness=0.2,
    )
    pipeline = MagicMock()
    pipeline.async_resolve = AsyncMock(return_value=result)
    pipeline.async_enrich = AsyncMock(return_value=None)

    bpm_sensor = AxBpmSensor(entry, "media_player.test", pipeline=pipeline)
    bpm_sensor.hass = hass
    bpm_sensor.async_write_ha_state = MagicMock()

    mood_sensor = _make_mood_sensor(hass)
    await mood_sensor.async_added_to_hass()

    await bpm_sensor._async_resolve()

    # BPM sensor: BPM-only attributes.
    assert bpm_sensor._attr_native_value == 128.0
    attrs = bpm_sensor._attr_extra_state_attributes
    assert attrs["bpm_raw"] == 128.0
    assert attrs["isrc"] == "GBDUW0000059"
    assert "mood_scores" not in attrs
    assert "mood_label" not in attrs
    assert "intensity" not in attrs
    assert "calmness" not in attrs

    # Mood sensor: received the mood attrs via the hook.
    assert mood_sensor._attr_native_value == "aggressive"
    assert mood_sensor._attr_extra_state_attributes["mood_scores"] == {
        "aggressive": 0.9
    }


@pytest.mark.asyncio
async def test_bpm_sensor_failure_pushes_none_to_mood():
    """Failed resolution → mood hook receives None (mood clears too)."""
    hass = _FakeHass()
    entry = _FakeEntry()
    pipeline = MagicMock()
    pipeline.async_resolve = AsyncMock(return_value=None)
    pipeline.async_enrich = AsyncMock(return_value=None)

    bpm_sensor = AxBpmSensor(entry, "media_player.test", pipeline=pipeline)
    bpm_sensor.hass = hass
    bpm_sensor.async_write_ha_state = MagicMock()

    mood_sensor = _make_mood_sensor(hass)
    await mood_sensor.async_added_to_hass()
    mood_sensor._on_mood_update({"mood_label": "party"})

    await bpm_sensor._async_resolve()

    assert bpm_sensor._attr_native_value is None
    assert bpm_sensor._attr_extra_state_attributes == {}
    assert mood_sensor._attr_native_value is None
    assert mood_sensor._attr_extra_state_attributes == {}


@pytest.mark.asyncio
async def test_bpm_sensor_enrichment_forwards_mood():
    """Enrichment mood attrs are forwarded to the mood sensor, not merged."""
    hass = _FakeHass()
    entry = _FakeEntry()
    result = ResolutionResult(
        120.0,
        source="deezer_metadata",
        isrc="GBDUW0000059",
        mood_scores=None,
        mood_label=None,
        intensity=None,
        calmness=None,
    )
    pipeline = MagicMock()
    pipeline.async_resolve = AsyncMock(return_value=result)
    pipeline.async_enrich = AsyncMock(
        return_value={
            "mood_scores": {"relaxed": 0.8},
            "mood_label": "relaxed",
            "intensity": 0.3,
            "calmness": 0.7,
        }
    )

    bpm_sensor = AxBpmSensor(entry, "media_player.test", pipeline=pipeline)
    bpm_sensor.hass = hass
    bpm_sensor.async_write_ha_state = MagicMock()

    mood_sensor = _make_mood_sensor(hass)
    await mood_sensor.async_added_to_hass()

    await bpm_sensor._async_resolve()

    # BPM sensor attrs stay mood-free after enrichment.
    assert "mood_label" not in bpm_sensor._attr_extra_state_attributes
    # Mood sensor got the enriched mood.
    assert mood_sensor._attr_native_value == "relaxed"
    assert mood_sensor._attr_extra_state_attributes["calmness"] == 0.7


@pytest.mark.asyncio
async def test_bpm_sensor_push_noop_without_mood_sensor():
    """No mood sensor registered → the push is silently dropped."""
    hass = _FakeHass()
    entry = _FakeEntry()
    result = ResolutionResult(
        128.0,
        source="analyzer",
        mood_label="party",
        mood_scores={"party": 1.0},
    )
    pipeline = MagicMock()
    pipeline.async_resolve = AsyncMock(return_value=result)
    pipeline.async_enrich = AsyncMock(return_value=None)

    bpm_sensor = AxBpmSensor(entry, "media_player.test", pipeline=pipeline)
    bpm_sensor.hass = hass
    bpm_sensor.async_write_ha_state = MagicMock()

    # No mood sensor added → no "mood_update" key in hass.data.
    await bpm_sensor._async_resolve()
    assert bpm_sensor._attr_native_value == 128.0
