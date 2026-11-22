"""Regression tests for the configured-media-player resolution (3.4.1).

The bug: `config_flow.py`'s options handler returns
`async_create_entry(data=...)` inside an `OptionsFlow`, so every user edit is
written to `entry.options` ONLY. `entry.data` is never updated after creation.
Three call sites read `entry.data` anyway, so the entities subscribed
`async_track_state_change_event` to the entity_id from CREATION time while the
pipeline resolved against the configured one. No event → `_on_media_change`
never ran → no resolution was ever scheduled → `unknown` forever, silently
(`unknown` is a legal state and every failure path here is quiet).

Covers:
1. options-only entry → sensors use the options value.
2. both present and differing → options wins.
3. data-only legacy entry → still works, no KeyError.
4. nothing anywhere → no entities added + the ERROR is logged.
   plus the "entity_id absent from the registry" WARNING.
5. startup while a track is already playing → a resolution is scheduled
   immediately, without waiting for a state-change event.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.helpers import entity_registry

from ax_bpm.const import DOMAIN
from ax_bpm.media import configured_media_player
from ax_bpm.mood_sensor import async_setup_entry as async_setup_mood
from ax_bpm.sensor import AxBpmSensor
from ax_bpm.sensor import async_setup_entry as async_setup_sensor

ENTRY_ID = "01M3T3Z3VYVXYKYZJP7PP1NRBX"
STALE = "media_player.old_entity"
FRESH = "media_player.kitchen"


class _State:
    def __init__(self, state="playing", **attrs) -> None:
        self.state = state
        self.attributes = attrs


class _States:
    def __init__(self, initial: _State | None) -> None:
        self.current = initial

    def get(self, _entity_id):
        return self.current


class _Hass:
    """Minimal hass plus the entity registry the setup path inspects."""

    def __init__(self, state: _State | None = None) -> None:
        self.data = {DOMAIN: {ENTRY_ID: {}}}
        self.states = _States(state)
        self.created_tasks: list = []

    def async_create_task(self, coro):
        self.created_tasks.append(coro)
        return coro


class _Entry:
    def __init__(self, data: dict, options: dict | None = None) -> None:
        self.entry_id = ENTRY_ID
        self.data = data
        self.options = options or {}


def _register(hass: _Hass, *entity_ids: str) -> None:
    """Put entity_ids into the fake registry (present = real HA entity)."""
    registry = entity_registry.async_get(hass)
    for entity_id in entity_ids:
        registry.entities[entity_id] = object()


def _setup_harness(hass: _Hass, entry: _Entry):
    """hass.data the way __init__.async_setup_entry builds it."""
    pipeline = MagicMock()
    pipeline.async_resolve = AsyncMock(return_value=None)
    pipeline.async_enrich = AsyncMock(return_value=None)
    hass.data[DOMAIN][ENTRY_ID].update(
        {"pipeline": pipeline, "status_coordinator": MagicMock()}
    )
    return pipeline


# ---------------------------------------------------------------------------
# 6. The resolver itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "options", "expected"),
    [
        ({"media_player": STALE}, {"media_player": FRESH}, FRESH),
        ({}, {"media_player": FRESH}, FRESH),
        ({"media_player": STALE}, {}, STALE),
        ({"media_player": STALE}, {"media_player": ""}, STALE),
        ({}, {}, None),
        ({"media_player": STALE}, {"media_player": None}, STALE),
    ],
)
def test_configured_media_player_merges_options_over_data(data, options, expected):
    """Options win; empty options fall through; absence never raises."""
    assert configured_media_player(SimpleNamespace(data=data, options=options)) == expected


# ---------------------------------------------------------------------------
# 1-3. Sensor setup reads the merged value
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_options_only_entry_uses_the_options_value():
    """The reported bug: the player was changed in the options dialog."""
    hass = _Hass()
    entry = _Entry(data={}, options={"media_player": FRESH})
    _setup_harness(hass, entry)
    _register(hass, FRESH)
    added = MagicMock()

    await async_setup_sensor(hass, entry, added)

    entities = added.call_args.args[0]
    assert entities[0]._media_player_id == FRESH, "BPM sensor kept the stale player"
    assert entities[1]._media_player_id == FRESH, "mood sensor kept the stale player"


@pytest.mark.asyncio
async def test_differing_data_and_options_means_options_win():
    """Merge order, not fallback: the options value always wins."""
    hass = _Hass()
    entry = _Entry(data={"media_player": STALE}, options={"media_player": FRESH})
    _setup_harness(hass, entry)
    _register(hass, FRESH, STALE)
    added = MagicMock()

    await async_setup_sensor(hass, entry, added)

    entities = added.call_args.args[0]
    assert entities[0]._media_player_id == FRESH
    assert entities[1]._media_player_id == FRESH


@pytest.mark.asyncio
async def test_legacy_data_only_entry_still_works():
    """Entries created before an options save have only data — no KeyError."""
    hass = _Hass()
    entry = _Entry(data={"media_player": STALE}, options={})
    _setup_harness(hass, entry)
    _register(hass, STALE)
    added = MagicMock()

    await async_setup_sensor(hass, entry, added)

    assert added.call_args.args[0][0]._media_player_id == STALE


@pytest.mark.asyncio
async def test_mood_sensor_setup_reads_the_merged_value():
    """The mood sensor must resolve the SAME player as the BPM sensor."""
    hass = _Hass()
    entry = _Entry(data={"media_player": STALE}, options={"media_player": FRESH})
    added = MagicMock()

    await async_setup_mood(hass, entry, added)

    assert added.call_args.args[0][0]._media_player_id == FRESH


# ---------------------------------------------------------------------------
# 4. Nothing configured
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_media_player_anywhere_adds_no_entities_and_logs(caplog):
    """Unconfigured entry: no entities, and the ERROR is visible."""
    hass = _Hass()
    entry = _Entry(data={}, options={})
    _setup_harness(hass, entry)
    added = MagicMock()

    with caplog.at_level("ERROR", logger="ax_bpm.sensor"):
        await async_setup_sensor(hass, entry, added)

    added.assert_not_called()
    assert "No media player configured for AX BPM" in caplog.text


@pytest.mark.asyncio
async def test_no_media_player_mood_sensor_adds_nothing():
    """The mood sensor stays silent — the BPM sensor owns the error log."""
    hass = _Hass()
    entry = _Entry(data={}, options={})
    added = MagicMock()

    await async_setup_mood(hass, entry, added)

    added.assert_not_called()


@pytest.mark.asyncio
async def test_unregistered_media_player_warns_with_the_entity_id(caplog):
    """The invisible failure mode: an entity_id that can never emit events."""
    hass = _Hass()
    entry = _Entry(data={}, options={"media_player": "media_player.renamed"})
    _setup_harness(hass, entry)
    added = MagicMock()

    with caplog.at_level("WARNING", logger="ax_bpm.sensor"):
        await async_setup_sensor(hass, entry, added)

    added.assert_called_once()  # still set up; this is advisory
    assert "media_player.renamed" in caplog.text
    assert "options dialog" in caplog.text
    # The YAML caveat must be present: unregistered is not always a fault.
    assert "YAML" in caplog.text


@pytest.mark.asyncio
async def test_registered_media_player_does_not_warn(caplog):
    """A normal registered player produces no warning."""
    hass = _Hass()
    entry = _Entry(data={}, options={"media_player": FRESH})
    _setup_harness(hass, entry)
    _register(hass, FRESH)
    added = MagicMock()

    with caplog.at_level("WARNING", logger="ax_bpm.sensor"):
        await async_setup_sensor(hass, entry, added)

    assert caplog.text == ""


@pytest.mark.asyncio
async def test_subscription_targets_the_merged_player():
    """The state-change subscription itself carries the resolved id."""
    hass = _Hass(_State("playing", media_title="T", media_artist="A"))
    entry = _Entry(data={"media_player": STALE}, options={"media_player": FRESH})
    _setup_harness(hass, entry)
    _register(hass, FRESH)

    subscribed: list[list[str]] = []

    def _track(_hass, entities, _action):
        subscribed.append(list(entities))
        return lambda: None

    with patch("ax_bpm.sensor.async_track_state_change_event", _track):
        added = MagicMock()
        await async_setup_sensor(hass, entry, added)
        # Entities 0 and 1 are the BPM + mood sensors; the third is the
        # analyzer status sensor, which does not subscribe to the player.
        for entity in added.call_args.args[0][:2]:
            entity.hass = hass
            entity.async_write_ha_state = MagicMock()
            await entity.async_added_to_hass()

    assert subscribed, "nothing subscribed"
    assert all(ids == [FRESH] for ids in subscribed), subscribed


# ---------------------------------------------------------------------------
# 5. Startup while a track is already playing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_startup_while_playing_schedules_a_resolution():
    """Restart mid-track must not wait for a state change that may not come."""
    state = _State(
        "playing",
        media_content_id="1",
        media_title="Already Playing",
        media_artist="A",
        media_duration=180,
        media_position=42,
    )
    hass = _Hass(state)
    entry = _Entry(data={"media_player": FRESH})
    pipeline = _setup_harness(hass, entry)

    result = MagicMock()
    result.bpm = 175.2
    result.attrs = {"source": "deezer_metadata", "mood_label": "party"}
    pipeline.async_resolve = AsyncMock(return_value=result)
    mood_update = MagicMock()
    hass.data[DOMAIN][ENTRY_ID]["mood_update"] = mood_update

    sensor = AxBpmSensor(entry, FRESH, pipeline)
    sensor.hass = hass
    sensor.async_write_ha_state = MagicMock()

    await sensor.async_added_to_hass()

    assert len(hass.created_tasks) == 1, "no resolution scheduled at startup"
    # Same track: the generation must NOT be bumped, or the guard discards
    # the result that is about to arrive.
    assert sensor._generation == 0
    # Seeded, not cleared — otherwise the first position/volume tick looks
    # like a track change and discards this in-flight result.
    assert sensor._last_track is not None
    assert sensor._last_position == 42.0

    await hass.created_tasks[0]  # drain the scheduled resolution

    assert sensor._attr_native_value == 175.2
    assert sensor._published_generation == sensor._generation
    mood_update.assert_called_with({"mood_label": "party"})


@pytest.mark.asyncio
async def test_startup_while_paused_schedules_nothing():
    """A paused/off player resolves nothing at setup."""
    hass = _Hass(_State("paused", media_title="T", media_artist="A"))
    entry = _Entry(data={"media_player": FRESH})
    _setup_harness(hass, entry)

    sensor = AxBpmSensor(entry, FRESH, MagicMock())
    sensor.hass = hass
    sensor.async_write_ha_state = MagicMock()

    await sensor.async_added_to_hass()

    assert hass.created_tasks == []
    assert sensor._last_track is None


@pytest.mark.asyncio
async def test_startup_with_no_state_schedules_nothing():
    """A player that does not exist yet must not raise at setup."""
    hass = _Hass(None)
    entry = _Entry(data={"media_player": FRESH})
    _setup_harness(hass, entry)

    sensor = AxBpmSensor(entry, FRESH, MagicMock())
    sensor.hass = hass
    sensor.async_write_ha_state = MagicMock()

    await sensor.async_added_to_hass()

    assert hass.created_tasks == []


@pytest.mark.asyncio
async def test_startup_resolution_survives_a_first_tick():
    """The seed + generation-0 pair is what keeps the startup result valid.

    Regression guard for the subtle failure: without seeding `_last_track`,
    the first position tick after a restart bumps the generation and the
    in-flight startup result is discarded by the generation guard.
    """
    state = _State(
        "playing",
        media_content_id="1",
        media_title="T",
        media_artist="A",
        media_duration=180,
        media_position=42,
    )
    hass = _Hass(state)
    entry = _Entry(data={"media_player": FRESH})
    pipeline = _setup_harness(hass, entry)

    sensor = AxBpmSensor(entry, FRESH, pipeline)
    sensor.hass = hass
    sensor.async_write_ha_state = MagicMock()
    hass.data[DOMAIN][ENTRY_ID]["mood_update"] = MagicMock()

    await sensor.async_added_to_hass()
    task = hass.created_tasks.pop()

    # A volume/position-only update arrives BEFORE the resolution lands.
    sensor._on_media_change(MagicMock(data={"new_state": state}))
    assert sensor._generation == 0, "an unchanged tick must not bump the generation"

    result = MagicMock()
    result.bpm = 128.0
    result.attrs = {}
    pipeline.async_resolve = AsyncMock(return_value=result)
    await task

    assert sensor._attr_native_value == 128.0


# ---------------------------------------------------------------------------
# 7. The entry-level read (source-level pin)
# ---------------------------------------------------------------------------


def test_setup_entry_stores_the_merged_media_player():
    """`__init__` cannot be imported under the test stubs, so pin the source.

    hass.data[DOMAIN][entry_id]["media_player"] feeds all four override
    services (services.py) — reading entry.data there handed them the
    player from creation time.
    """
    from pathlib import Path

    source = (
        Path(__file__).parent.parent
        / "custom_components"
        / "ax_bpm"
        / "__init__.py"
    ).read_text(encoding="utf-8")

    assert "configured_media_player(entry)" in source
    assert 'entry.data.get(CONF_MEDIA_PLAYER)' not in source
    # The pipeline must keep receiving the merged view.
    assert "{**entry.data, **entry.options}" in source


def test_no_unmerged_media_player_reads_remain():
    """Every sensor-side read goes through the helper."""
    from pathlib import Path

    base = Path(__file__).parent.parent / "custom_components" / "ax_bpm"
    for name in ("sensor.py", "mood_sensor.py"):
        source = (base / name).read_text(encoding="utf-8")
        assert 'data.get("media_player")' not in source, name
        assert "entry.data.get(CONF_MEDIA_PLAYER)" not in source, name
        assert "configured_media_player(entry)" in source, name
