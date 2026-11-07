"""Regression tests for the stale-BPM / stale-mood fixes.

Covers four independently-failing behaviours that together produced the
"entire album shows track 1's BPM" report:

1. Identity — `media.track_identity()` combines `media_content_id` with the
   artist/title/duration tuple, and `looks_restarted()` catches the case
   where a source recycles BOTH across tracks.
2. Generation guard — a slow resolution from an earlier track can never
   publish under a later one. Tuple equality previously compared EQUAL for
   consecutive tracks with identical metadata and let it through.
3. Publish grace — the entity HOLDS the previous value for
   `PUBLISH_GRACE`, then clears to `unknown` (the card derives 120) when
   nothing published for the new track.
4. Watchdog — a value that outlives `media_duration + WATCHDOG_GRACE`
   while the same identity is still reported is cleared, which is the only
   signal available when a source never changes any attribute at all.

Plus the dual-key cache contract that stopped Deezer-metadata entries from
being write-only.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ax_bpm.const import DOMAIN, PUBLISH_GRACE, WATCHDOG_GRACE
from ax_bpm.media import (
    looks_restarted,
    media_duration,
    track_identity,
)
from ax_bpm.sensor import AxBpmSensor
from ax_bpm.store import BpmCache, cache_key, cache_keys

ENTRY_ID = "01M3T3Z3VYVXYKYZJP7PP1NRBX"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _State:
    def __init__(self, state="playing", **attrs) -> None:
        self.state = state
        self.attributes = attrs


class _States:
    def __init__(self, initial: _State) -> None:
        self.current = initial

    def get(self, _entity_id):
        return self.current


class _Hass:
    """Minimal hass: no debounce/grace/watchdog registration side effects.

    `async_call_later` is patched out by the `sensor` fixture so timer
    callbacks are captured rather than scheduled — the tests then invoke
    them explicitly, which makes the 5 s and duration-based transitions
    deterministic instead of wall-clock dependent.
    """

    def __init__(self, state: _State) -> None:
        self.data = {DOMAIN: {ENTRY_ID: {}}}
        self.states = _States(state)
        self.created_tasks = []

    def async_create_task(self, coro):
        self.created_tasks.append(coro)
        return coro


class _Entry:
    def __init__(self) -> None:
        self.entry_id = ENTRY_ID
        self.data = {"media_player": "media_player.test"}


class _Timers:
    """Captured async_call_later registrations, keyed by delay."""

    def __init__(self) -> None:
        self.pending: list[tuple[float, object]] = []

    def __call__(self, hass, delay, callback):
        self.pending.append((delay, callback))

        def _cancel() -> None:
            self.pending = [p for p in self.pending if p[1] is not callback]

        return _cancel

    def fire(self, delay: float) -> None:
        """Fire the pending callback registered for exactly `delay`."""
        match = [p for p in self.pending if p[0] == delay]
        assert match, f"no timer registered for {delay}s (have {[p[0] for p in self.pending]})"
        _delay, callback = match[-1]
        callback(None)


@pytest.fixture
def timers():
    with patch("ax_bpm.sensor.async_call_later") as patched:
        registry = _Timers()
        patched.side_effect = registry
        yield registry


def _make_sensor(timers, state: _State, pipeline=None) -> tuple:
    hass = _Hass(state)
    entry = _Entry()
    if pipeline is None:
        pipeline = MagicMock()
        pipeline.async_resolve = AsyncMock(return_value=None)
        pipeline.async_enrich = AsyncMock(return_value=None)
    sensor = AxBpmSensor(entry, "media_player.test", pipeline)
    sensor.hass = hass
    sensor.async_write_ha_state = MagicMock()
    hass.data[DOMAIN][ENTRY_ID]["mood_update"] = MagicMock()
    return sensor, hass, pipeline


def _event(state: _State):
    return MagicMock(data={"new_state": state})


# ---------------------------------------------------------------------------
# 1. Identity
# ---------------------------------------------------------------------------


def test_identity_uses_content_id():
    """A content-id change alone counts as a track change."""
    a = _State("playing", media_content_id="1", media_title="Same", media_artist="A")
    b = _State("playing", media_content_id="2", media_title="Same", media_artist="A")
    assert track_identity(a) != track_identity(b)


def test_identity_falls_back_to_metadata():
    """No content id → the metadata tuple still discriminates."""
    a = _State("playing", media_title="One", media_artist="A")
    b = _State("playing", media_title="Two", media_artist="A")
    assert track_identity(a) != track_identity(b)


def test_identity_stable_for_unchanged_track():
    a = _State("playing", media_content_id="1", media_title="X", media_artist="A")
    b = _State("playing", media_content_id="1", media_title="X", media_artist="A")
    assert track_identity(a) == track_identity(b)


def test_identity_never_collides_without_content_id():
    """A metadata-only identity cannot equal a content-id identity."""
    with_id = _State("playing", media_content_id="1", media_title="X", media_artist="A")
    without_id = _State("playing", media_title="X", media_artist="A")
    assert track_identity(with_id) != track_identity(without_id)


def test_looks_restarted_detects_recycled_metadata():
    """Position resetting to ~0 well into a track is a new track."""
    assert looks_restarted(150.0, 0.0) is True
    assert looks_restarted(30.0, 1.5) is True


def test_looks_restarted_ignores_small_seek():
    """A backwards seek near the start must not bump the generation."""
    assert looks_restarted(4.0, 0.5) is False
    assert looks_restarted(3.0, 1.0) is False


def test_looks_restarted_requires_both_positions():
    assert looks_restarted(None, 0.0) is False
    assert looks_restarted(120.0, None) is False


def test_media_duration_ignores_zero_and_none():
    """The watchdog must not arm on a live stream (no expected end)."""
    assert media_duration(_State("playing", media_duration=180)) == 180.0
    assert media_duration(_State("playing", media_duration=0)) is None
    assert media_duration(_State("playing")) is None


# ---------------------------------------------------------------------------
# 2. Generation guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_identical_metadata_discards_stale_result(timers):
    """A result for generation N never publishes under generation N+1.

    Regression: the guard compared identity tuples, which are EQUAL for
    consecutive tracks sharing metadata, so a slow result from the first
    track published as if it belonged to the second.
    """
    state = _State(
        "playing",
        media_content_id="1",
        media_title="Same",
        media_artist="A",
        media_duration=180,
        media_position=10,
    )
    result = MagicMock()
    result.bpm = 174.0
    result.attrs = {"isrc": "GBDUW0000059"}

    pipeline = MagicMock()
    pipeline.async_enrich = AsyncMock(return_value=None)

    sensor, _hass, _ = _make_sensor(timers, state, pipeline)

    # The track changes WHILE the resolution is in flight: the generation
    # bump happens during async_resolve, exactly as _on_media_change does
    # it in production.
    async def _resolve_then_change(*_args, **_kwargs):
        sensor._generation += 1
        return result

    pipeline.async_resolve = AsyncMock(side_effect=_resolve_then_change)

    await sensor._async_resolve()

    assert sensor._attr_native_value is None, "a stale generation's result was published"
    assert sensor._published_generation != sensor._generation


@pytest.mark.asyncio
async def test_current_generation_publishes(timers):
    """Sanity: the matching generation still publishes normally."""
    state = _State(
        "playing",
        media_content_id="1",
        media_title="T",
        media_artist="A",
        media_duration=180,
        media_position=5,
    )
    result = MagicMock()
    result.bpm = 128.0
    result.attrs = {"isrc": "X"}

    pipeline = MagicMock()
    pipeline.async_resolve = AsyncMock(return_value=result)
    pipeline.async_enrich = AsyncMock(return_value=None)

    sensor, _hass, _ = _make_sensor(timers, state, pipeline)
    await sensor._async_resolve()

    assert sensor._attr_native_value == 128.0
    assert sensor._published_generation == sensor._generation


def test_restarted_position_counts_as_track_change(timers):
    """Identical metadata + a position restart still bumps the generation."""
    state = _State(
        "playing",
        media_content_id="1",
        media_title="Same",
        media_artist="A",
        media_duration=180,
        media_position=150,
    )
    sensor, _hass, _ = _make_sensor(timers, state)

    sensor._on_media_change(_event(state))
    first = sensor._generation
    assert first == 1

    # Same metadata/content-id, position reset to 0 → next track.
    restarted = _State(
        "playing",
        media_content_id="1",
        media_title="Same",
        media_artist="A",
        media_duration=180,
        media_position=0,
    )
    sensor._on_media_change(_event(restarted))
    assert sensor._generation == first + 1


def test_unchanged_position_update_does_not_bump(timers):
    """A volume/attribute-only update must not re-trigger resolution."""
    state = _State(
        "playing",
        media_content_id="1",
        media_title="Same",
        media_artist="A",
        media_duration=180,
        media_position=42,
    )
    sensor, _hass, _ = _make_sensor(timers, state)

    sensor._on_media_change(_event(state))
    first = sensor._generation
    sensor._on_media_change(_event(state))
    assert sensor._generation == first


# ---------------------------------------------------------------------------
# 3. Publish grace
# ---------------------------------------------------------------------------


def test_grace_clears_to_unknown_when_nothing_published(timers):
    """5 s after a track change with no publish → unknown, mood cleared."""
    state = _State(
        "playing",
        media_content_id="1",
        media_title="A",
        media_artist="A",
        media_duration=180,
        media_position=0,
    )
    sensor, hass, _ = _make_sensor(timers, state)
    sensor._attr_native_value = 174.0  # previous track's value, held
    sensor._attr_extra_state_attributes = {"isrc": "OLD"}

    sensor._on_media_change(_event(state))
    assert sensor._attr_native_value == 174.0, "value must be HELD, not cleared"

    timers.fire(PUBLISH_GRACE)

    assert sensor._attr_native_value is None
    assert sensor._attr_extra_state_attributes == {}
    hass.data[DOMAIN][ENTRY_ID]["mood_update"].assert_called_with(None)


def test_grace_does_not_clear_after_a_publish(timers):
    """A resolution that landed for this generation blocks the clear.

    Exercised through the guard rather than by cancelling the timer: a
    publish sets _published_generation, and the grace callback must then
    leave the freshly published value alone even if it still fires.
    """
    state = _State(
        "playing",
        media_content_id="1",
        media_title="A",
        media_artist="A",
        media_duration=180,
        media_position=0,
    )
    sensor, _hass, _ = _make_sensor(timers, state)

    sensor._on_media_change(_event(state))
    sensor._attr_native_value = 100.0
    sensor._published_generation = sensor._generation

    sensor._on_grace_expired(sensor._generation)
    assert sensor._attr_native_value == 100.0


def test_grace_from_superseded_generation_is_ignored(timers):
    """Only the current generation's grace may clear the entity."""
    state = _State(
        "playing",
        media_content_id="1",
        media_title="A",
        media_artist="A",
        media_duration=180,
        media_position=0,
    )
    sensor, _hass, _ = _make_sensor(timers, state)

    sensor._on_media_change(_event(state))
    stale_generation = sensor._generation
    sensor._generation += 1  # a newer track took ownership

    sensor._attr_native_value = 120.0
    sensor._on_grace_expired(stale_generation)
    assert sensor._attr_native_value == 120.0


def test_grace_clear_does_not_reset_last_track(timers):
    """Clearing must not re-arm the resolution storm (documented invariant)."""
    state = _State(
        "playing",
        media_content_id="1",
        media_title="A",
        media_artist="A",
        media_duration=180,
        media_position=0,
    )
    sensor, _hass, _ = _make_sensor(timers, state)
    sensor._on_media_change(_event(state))
    tracked = sensor._last_track

    sensor._attr_native_value = 174.0
    timers.fire(PUBLISH_GRACE)

    assert sensor._attr_native_value is None
    assert sensor._last_track == tracked


# ---------------------------------------------------------------------------
# 4. Watchdog
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watchdog_armed_at_duration_plus_grace(timers):
    """A publish arms the watchdog at (duration − position) + grace."""
    state = _State(
        "playing",
        media_content_id="1",
        media_title="A",
        media_artist="A",
        media_duration=180,
        media_position=170,
    )
    result = MagicMock()
    result.bpm = 174.0
    result.attrs = {}

    pipeline = MagicMock()
    pipeline.async_resolve = AsyncMock(return_value=result)
    pipeline.async_enrich = AsyncMock(return_value=None)

    sensor, _hass, _ = _make_sensor(timers, state, pipeline)
    await sensor._async_resolve()

    delays = [delay for delay, _cb in timers.pending]
    assert (180 - 170) + WATCHDOG_GRACE in delays


@pytest.mark.asyncio
async def test_watchdog_not_armed_without_duration(timers):
    """A live stream (no media_duration) must keep its BPM."""
    state = _State(
        "playing",
        media_content_id="1",
        media_title="A",
        media_artist="A",
        media_position=10,
    )
    result = MagicMock()
    result.bpm = 90.0
    result.attrs = {}

    pipeline = MagicMock()
    pipeline.async_resolve = AsyncMock(return_value=result)
    pipeline.async_enrich = AsyncMock(return_value=None)

    sensor, _hass, _ = _make_sensor(timers, state, pipeline)
    await sensor._async_resolve()

    assert sensor._watchdog_unsub is None


def test_watchdog_clears_a_stuck_album_value(timers):
    """The reported bug: one BPM held for the whole album → cleared.

    Nothing changes on the player (identical identity), so only the
    watchdog can act.
    """
    state = _State(
        "playing",
        media_content_id="1",
        media_title="Same",
        media_artist="A",
        media_duration=180,
        media_position=170,
    )
    sensor, _hass, _ = _make_sensor(timers, state)
    sensor._attr_native_value = 174.0
    sensor._generation = 0
    sensor._published_generation = 0

    sensor._on_watchdog_expired(0)

    assert sensor._attr_native_value is None


def test_watchdog_does_not_clear_while_paused(timers):
    """A frozen position makes "should have ended" meaningless."""
    state = _State(
        "paused",
        media_content_id="1",
        media_title="A",
        media_artist="A",
        media_duration=180,
        media_position=170,
    )
    sensor, _hass, _ = _make_sensor(timers, state)
    sensor._attr_native_value = 174.0

    sensor._on_watchdog_expired(0)
    assert sensor._attr_native_value == 174.0


def test_watchdog_ignores_superseded_generation(timers):
    """A newer track owns the entity; the old watchdog must not clear it."""
    state = _State(
        "playing",
        media_content_id="1",
        media_title="A",
        media_artist="A",
        media_duration=180,
        media_position=170,
    )
    sensor, _hass, _ = _make_sensor(timers, state)
    sensor._attr_native_value = 174.0
    sensor._generation = 5

    sensor._on_watchdog_expired(4)
    assert sensor._attr_native_value == 174.0


def test_stop_cancels_every_timer(timers):
    """off/unavailable leaves no pending timers (no post-removal writes)."""
    state = _State("playing", media_content_id="1", media_title="A", media_artist="A")
    sensor, _hass, _ = _make_sensor(timers, state)
    sensor._on_media_change(_event(state))

    sensor._on_media_change(_event(_State("off")))

    assert sensor._debounce_unsub is None
    assert sensor._grace_unsub is None
    assert sensor._watchdog_unsub is None
    assert sensor._attr_native_value is None


# ---------------------------------------------------------------------------
# 5. Dual-key cache
# ---------------------------------------------------------------------------


def test_cache_keys_include_hash_for_isrc_tracks():
    """The pre-match read only has the hash key, so it must always exist."""
    keys = cache_keys("GBDUW0000059", "White Town", "Your Woman", 240.0)
    assert keys[0] == cache_key("GBDUW0000059", "White Town", "Your Woman", 240.0)
    assert keys[0].startswith("isrc:")
    assert keys[-1] == cache_key(None, "White Town", "Your Woman", 240.0)
    assert keys[-1].startswith("hash:")


def test_cache_keys_without_isrc_is_single_hash():
    keys = cache_keys(None, "White Town", "Your Woman", 240.0)
    assert len(keys) == 1
    assert keys[0].startswith("hash:")


@pytest.mark.asyncio
async def test_dual_key_write_is_readable_pre_match():
    """The reported defect: ISRC writes were unreachable before a match."""

    class _FakeStore:
        def __init__(self):
            self.data = {}

        async def async_load(self):
            return dict(self.data)

        async def async_save(self, data):
            self.data = dict(data)

    cache = BpmCache.__new__(BpmCache)
    cache._store = _FakeStore()
    cache._data = {}

    keys = cache_keys("GBDUW0000059", "White Town", "Your Woman", 240.0)
    await cache.async_put_many(keys, {"bpm": 103.0})

    # The pre-match read (isrc unknown) must now hit.
    pre_match = cache.get_first(cache_keys(None, "White Town", "Your Woman", 240.0))
    assert pre_match is not None
    assert pre_match["bpm"] == 103.0

    # The ISRC read is still available and wins when offered first.
    post_match = cache.get_first(keys)
    assert post_match is not None and post_match["bpm"] == 103.0


def test_get_first_prefers_isrc_entry():
    """ISRC key wins over hash (duration-stable identity)."""

    class _FakeStore:
        async def async_load(self):
            return {}

        async def async_save(self, data):
            pass

    cache = BpmCache.__new__(BpmCache)
    cache._store = _FakeStore()
    from ax_bpm.const import CACHE_PAYLOAD_VERSION

    isrc_key = cache_key("GBDUW0000059", "A", "B", 240.0)
    hash_key = cache_key(None, "A", "B", 240.0)
    cache._data = {
        isrc_key: {"bpm": 103.0, "payload_version": CACHE_PAYLOAD_VERSION},
        hash_key: {"bpm": 999.0, "payload_version": CACHE_PAYLOAD_VERSION},
    }

    assert cache.get_first([isrc_key, hash_key])["bpm"] == 103.0
    assert cache.get_first([hash_key, isrc_key])["bpm"] == 999.0
