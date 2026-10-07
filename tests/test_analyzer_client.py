"""Phase 1 tests: analyzer_client, cache v1→v2 migration, config-flow migration.

analyzer_client mock matrix (parent plan Phase 3 item 1):
- healthy / slow (timeout) / HTTP 500 / unreachable — the client returns
  None in every failure mode and never raises.
Cache migration: legacy SVM mood fields dropped, BPM kept.
Config flow: legacy genre/mood toggles → octave_disambiguation dropdown.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from ax_bpm.analyzer_client import AnalyzerClient
from ax_bpm.config_flow import _migrate_legacy_toggles
from ax_bpm.const import (
    OCTAVE_GENRE_MOOD,
    OCTAVE_GENRE_ONLY,
    OCTAVE_OFF,
)
from ax_bpm.store import (
    LEGACY_MOOD_FIELDS,
    STORAGE_VERSION,
    BpmCache,
    MigratingStore,
)

# ---------------------------------------------------------------------------
# AnalyzerClient — failure matrix
# ---------------------------------------------------------------------------


def _make_client(
    manual_url: str | None = None, discovered_url: str | None = None
) -> tuple[AnalyzerClient, MagicMock]:
    session = MagicMock()
    return AnalyzerClient(session, manual_url, discovered_url=discovered_url), session


def _resp(status: int, body=None):
    resp = MagicMock()
    resp.status = status

    async def json(content_type=None):
        return body

    resp.json = json
    return resp


class _Ctx:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *args):
        return False


@pytest.mark.asyncio
async def test_analyze_healthy():
    client, session = _make_client("http://analyzer:8099")
    payload = {"mood_tags": [{"tag": "party", "score": 0.9}], "valence": 0.5}
    session.post = MagicMock(return_value=_Ctx(_resp(200, payload)))
    result = await client.async_analyze(b"mp3")
    assert result == payload


@pytest.mark.asyncio
async def test_analyze_http_500_returns_none():
    client, session = _make_client("http://analyzer:8099")
    session.post = MagicMock(return_value=_Ctx(_resp(500)))
    assert await client.async_analyze(b"mp3") is None


@pytest.mark.asyncio
async def test_analyze_timeout_returns_none():
    client, session = _make_client("http://analyzer:8099")

    class _SlowCtx:
        # Dunder methods are looked up on the type — a subclass is required
        # to simulate a hung request.
        async def __aenter__(self):
            await asyncio.sleep(3600)
            raise AssertionError("should have been cancelled")

        async def __aexit__(self, *args):
            return False

    session.post = MagicMock(return_value=_SlowCtx())
    assert await client.async_analyze(b"mp3") is None


@pytest.mark.asyncio
async def test_analyze_unreachable_returns_none():
    client, session = _make_client("http://analyzer:8099")
    session.post = MagicMock(side_effect=ConnectionError("refused"))
    # aiohttp raises ClientError subclasses; ConnectionError maps onto it.
    assert await client.async_analyze(b"mp3") is None


@pytest.mark.asyncio
async def test_analyze_no_url_feature_off():
    client, _ = _make_client(None)
    assert await client.async_analyze(b"mp3") is None


@pytest.mark.asyncio
async def test_detect_manual_url_wins():
    client, session = _make_client("http://manual:8099")
    session.get = MagicMock(return_value=_Ctx(_resp(200, {"status": "ok"})))
    url = await client.async_detect()
    assert url == "http://manual:8099"
    # Only the manual URL was probed.
    assert session.get.call_count == 1


@pytest.mark.asyncio
async def test_detect_all_down_returns_none():
    client, session = _make_client(None)
    session.get = MagicMock(return_value=_Ctx(_resp(503)))
    assert await client.async_detect() is None
    # Probed every auto-detect candidate exactly once (no retry).
    assert session.get.call_count == len(
        __import__("ax_bpm.const", fromlist=["ANALYZER_URLS"]).ANALYZER_URLS
    )


@pytest.mark.asyncio
async def test_detect_cached_after_first_probe():
    client, session = _make_client("http://manual:8099")
    session.get = MagicMock(return_value=_Ctx(_resp(200, {"status": "ok"})))
    assert await client.async_detect() == "http://manual:8099"
    assert await client.async_detect() == "http://manual:8099"
    assert session.get.call_count == 1


@pytest.mark.asyncio
async def test_detect_discovered_url_used_when_no_manual():
    """Supervisor-discovered URL is probed before the localhost fallback."""
    client, session = _make_client(
        None, discovered_url="http://abc123_ax-bpm-analyzer:8099"
    )
    session.get = MagicMock(return_value=_Ctx(_resp(200, {"status": "ok"})))
    url = await client.async_detect()
    assert url == "http://abc123_ax-bpm-analyzer:8099"
    assert session.get.call_count == 1


@pytest.mark.asyncio
async def test_detect_manual_beats_discovered():
    """A manual override always wins over the discovered URL."""
    client, session = _make_client(
        "http://manual:8099", discovered_url="http://discovered:8099"
    )
    session.get = MagicMock(return_value=_Ctx(_resp(200, {"status": "ok"})))
    assert await client.async_detect() == "http://manual:8099"
    assert session.get.call_count == 1


@pytest.mark.asyncio
async def test_detect_discovered_falls_back_to_localhost():
    """Discovered URL down → the homeassistant.local fallback is probed."""
    client, session = _make_client(None, discovered_url="http://discovered:8099")
    probed: list[str] = []

    def _get(url, **kwargs):
        probed.append(url)
        # Only the localhost fallback answers.
        if "homeassistant.local" in url:
            return _Ctx(_resp(200, {"status": "ok"}))
        return _Ctx(_resp(503))

    session.get = MagicMock(side_effect=_get)
    url = await client.async_detect()
    assert url is not None and "homeassistant.local" in url
    assert any("discovered" in u for u in probed)


# ---------------------------------------------------------------------------
# AnalyzerClient — cooldown-bounded re-probe (startup-ordering self-heal)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_detect_cooldown_blocks_immediate_reprobe():
    """A failed detection is NOT re-probed inside the cooldown window."""
    client, session = _make_client(None)
    session.get = MagicMock(return_value=_Ctx(_resp(503)))
    assert await client.async_detect() is None
    first_count = session.get.call_count
    # Immediate second call: cooldown active → no new probes.
    assert await client.async_detect() is None
    assert session.get.call_count == first_count


@pytest.mark.asyncio
async def test_failed_detect_reprobes_after_cooldown(monkeypatch):
    """After REPROBE_COOLDOWN the client retries detection and can recover."""
    from ax_bpm import analyzer_client as mod

    client, session = _make_client(None)
    session.get = MagicMock(return_value=_Ctx(_resp(503)))
    assert await client.async_detect() is None

    # Simulate the cooldown having elapsed (offset from the REAL monotonic
    # clock — a fixed constant can be below the current uptime).
    now = mod.time.monotonic()
    monkeypatch.setattr(
        mod.time, "monotonic", lambda: now + mod.REPROBE_COOLDOWN + 1.0
    )
    session.get = MagicMock(return_value=_Ctx(_resp(200, {"status": "ok"})))
    assert await client.async_detect() == mod.ANALYZER_URLS[0]
    assert client.base_url == mod.ANALYZER_URLS[0]


@pytest.mark.asyncio
async def test_async_analyze_reprobes_when_never_detected(monkeypatch):
    """async_analyze with no resolved URL triggers a cooldown-bounded re-detect."""
    from ax_bpm import analyzer_client as mod

    client, session = _make_client(None)
    # Health probe fails on the first detect, succeeds on the re-probe.
    responses = [_Ctx(_resp(503)), _Ctx(_resp(200, {"status": "ok"}))]
    session.get = MagicMock(side_effect=lambda *a, **k: responses.pop(0))
    # /analyze answers once the URL resolves.
    payload = {"bpm": 120.0}
    session.post = MagicMock(return_value=_Ctx(_resp(200, payload)))

    # First analyze: detect fails → None (no POST).
    assert await client.async_analyze(b"mp3") is None
    assert session.post.call_count == 0

    # Cooldown elapsed → the next analyze re-detects and POSTs (offset from
    # the REAL monotonic clock — a fixed constant can be below the uptime).
    now = mod.time.monotonic()
    monkeypatch.setattr(
        mod.time, "monotonic", lambda: now + mod.REPROBE_COOLDOWN + 1.0
    )
    assert await client.async_analyze(b"mp3") == payload
    assert session.post.call_count == 1


@pytest.mark.asyncio
async def test_async_analyze_force_reprobe_when_resolved_url_dies():
    """A resolved URL that stops answering triggers a forced re-detect."""
    client, session = _make_client(None)
    session.get = MagicMock(return_value=_Ctx(_resp(200, {"status": "ok"})))
    assert await client.async_detect() is not None

    # Analyzer goes down: /analyze POST fails with a ClientError.
    session.post = MagicMock(side_effect=ConnectionError("refused"))
    assert await client.async_analyze(b"mp3") is None
    # The forced re-detect probed health again (detect ran after the failure).
    assert session.get.call_count >= 2


# ---------------------------------------------------------------------------
# Cache v1 → v2 migration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_migration_drops_legacy_mood_fields():
    """Legacy SVM mood fields are stripped on load AND the entries are
    invalidated by payload versioning (mood-degeneracy fix).

    Pre-versioning entries carry no payload_version — their provenance
    cannot be verified, so they read as absent and the tracks are
    re-analyzed (first-burst latency, by design). The strip still runs
    so the on-disk data is clean for the eventual overwrite.
    """
    cache = BpmCache.__new__(BpmCache)
    cache._store = MagicMock()
    cache._store.async_load = AsyncMock(
        return_value={
            "isrc:GBDUW0000059": {
                "bpm": 174.0,
                "mood_scores": {"aggressive": 0.9},
                "mood_label": "aggressive",
                "track": "Daft Punk - Test Track",
            },
            "hash:abc": {"bpm": 120.0},
        }
    )
    await cache.async_load()
    # In-memory data was stripped of the legacy fields…
    stored = cache._data["isrc:GBDUW0000059"]
    for field in LEGACY_MOOD_FIELDS:
        assert field not in stored
    # …but BOTH entries are invalidated (no payload_version) — the
    # pre-versioning era cannot be trusted post scale-fix.
    assert cache.get("isrc:GBDUW0000059") is None
    assert cache.get("hash:abc") is None


@pytest.mark.asyncio
async def test_store_migrate_func_strips_legacy_fields_from_v1_data():
    """The Store-level migration drops legacy mood fields from v1 data.

    This is the path HA takes when the on-disk .storage/ax_bpm_cache
    file was written as version 1 — without the override, HA's base
    _async_migrate_func raises NotImplementedError and setup fails.
    """
    store = MigratingStore.__new__(MigratingStore)
    v1_data = {
        "isrc:GBDUW0000059": {
            "bpm": 174.0,
            "mood_scores": {"aggressive": 0.9},
            "mood_label": "aggressive",
            "track": "Daft Punk - Test Track",
        },
        "hash:abc": {"bpm": 120.0},
        "not-a-dict": "junk",
    }
    migrated = await store._async_migrate_func(1, 1, v1_data)
    assert migrated is v1_data  # mutated in place, passed through
    entry = migrated["isrc:GBDUW0000059"]
    assert entry["bpm"] == 174.0  # BPM kept
    assert entry["track"] == "Daft Punk - Test Track"
    for field in LEGACY_MOOD_FIELDS:
        assert field not in entry
    # Non-mood entry and non-dict value untouched.
    assert migrated["hash:abc"] == {"bpm": 120.0}
    assert migrated["not-a-dict"] == "junk"


@pytest.mark.asyncio
async def test_store_migrate_func_passes_through_current_version():
    """Already-v2 data is returned unchanged (no legacy fields to strip)."""
    store = MigratingStore.__new__(MigratingStore)
    v2_data = {"isrc:GBDUW0000059": {"bpm": 174.0, "track": "Test"}}
    migrated = await store._async_migrate_func(STORAGE_VERSION, 1, v2_data)
    assert migrated == v2_data


@pytest.mark.asyncio
async def test_store_migrate_func_future_version_passthrough():
    """A future on-disk version is passed through without mutation.

    HA raises UnsupportedStorageVersionError before calling the migrate
    func in that case, but the override must stay defensive.
    """
    store = MigratingStore.__new__(MigratingStore)
    future_data = {
        "isrc:GBDUW0000059": {
            "bpm": 174.0,
            "mood_scores": {"aggressive": 0.9},
        }
    }
    migrated = await store._async_migrate_func(99, 1, future_data)
    assert migrated is future_data
    assert migrated["isrc:GBDUW0000059"]["mood_scores"] == {"aggressive": 0.9}


@pytest.mark.asyncio
async def test_cache_load_handles_non_dict():
    cache = BpmCache.__new__(BpmCache)
    cache._store = MagicMock()
    cache._store.async_load = AsyncMock(return_value=None)
    await cache.async_load()
    assert cache.get("anything") is None


# ---------------------------------------------------------------------------
# Config flow — legacy toggle migration
# ---------------------------------------------------------------------------


def test_migrate_legacy_toggles():
    # genre+mood → Genre + mood
    assert (
        _migrate_legacy_toggles({"genre_correction": True, "mood_correction": True})
        == OCTAVE_GENRE_MOOD
    )
    # genre-only → Genre only
    assert (
        _migrate_legacy_toggles({"genre_correction": True, "mood_correction": False})
        == OCTAVE_GENRE_ONLY
    )
    # both off → Off
    assert (
        _migrate_legacy_toggles({"genre_correction": False, "mood_correction": False})
        == OCTAVE_OFF
    )
    # stored dropdown value wins over legacy toggles
    assert (
        _migrate_legacy_toggles(
            {
                "genre_correction": True,
                "mood_correction": True,
                "octave_disambiguation": OCTAVE_OFF,
            }
        )
        == OCTAVE_OFF
    )
    # defaults when nothing stored: genre on, mood off → Genre only
    assert _migrate_legacy_toggles({}) == OCTAVE_GENRE_ONLY


# ---------------------------------------------------------------------------
# Config flow — async_step_hassio discovery handling
# ---------------------------------------------------------------------------


class _FakeEntry:
    """Minimal ConfigEntry stand-in for the hassio-step tests."""

    def __init__(self, data):
        self.entry_id = "test-entry"
        self.data = dict(data)
        self.title = "AX BPM"


class _FakeHass:
    def __init__(self, entries):
        self._entries = entries
        self.config_entries = MagicMock()
        # Mirror the real API: async_update_entry(entry, data=...) mutates
        # the entry in place.
        self.config_entries.async_update_entry = MagicMock(
            side_effect=self._update_entry
        )

    @staticmethod
    def _update_entry(entry, data=None, **kwargs):
        if data is not None:
            entry.data = dict(data)

    def entries(self):
        return self._entries


class _HassioInfo:
    def __init__(self, config):
        self.config = config


def _make_flow(entries):
    from ax_bpm.config_flow import AxBpmConfigFlow

    flow = AxBpmConfigFlow.__new__(AxBpmConfigFlow)
    flow.hass = _FakeHass(entries)
    flow._discovered_url = None
    # Base-class methods (stubbed ConfigFlow in conftest) — bind fakes that
    # mirror the real return shapes.
    flow._async_current_entries = lambda: flow.hass.entries()
    flow.async_abort = lambda *, reason: {"type": "abort", "reason": reason}
    flow.async_show_form = lambda *, step_id, **kwargs: {
        "type": "form",
        "step_id": step_id,
    }
    # The user step probes the analyzer — stub it out (network-free).
    flow._probe_analyzer = AsyncMock(return_value=False)
    return flow


@pytest.fixture
def _stub_schema(monkeypatch):
    """Replace _build_schema — the conftest selector stubs return tuples
    that real voluptuous cannot compile (schema building is not under test
    here; the hassio routing logic is)."""
    from ax_bpm import config_flow as cf

    monkeypatch.setattr(
        cf, "_build_schema", lambda defaults, detected: MagicMock()
    )


@pytest.mark.asyncio
async def test_hassio_existing_entry_patched_and_aborted():
    """Discovery on an ALREADY-CONFIGURED entry patches the URL + aborts."""
    entry = _FakeEntry({"media_player": "media_player.x"})
    flow = _make_flow([entry])

    result = await flow.async_step_hassio(
        _HassioInfo({"host": "abc123-ax-bpm-analyzer", "port": 8099})
    )

    assert result["type"] == "abort"
    assert result["reason"] == "already_configured"
    assert entry.data["discovered_analyzer_url"] == (
        "http://abc123-ax-bpm-analyzer:8099"
    )
    flow.hass.config_entries.async_update_entry.assert_called_once()


@pytest.mark.asyncio
async def test_hassio_existing_entry_with_url_not_repainted(_stub_schema):
    """An entry that already carries a discovered URL is left untouched."""
    entry = _FakeEntry(
        {"media_player": "media_player.x", "discovered_analyzer_url": "http://old:8099"}
    )
    flow = _make_flow([entry])

    result = await flow.async_step_hassio(
        _HassioInfo({"host": "new-host", "port": 8099})
    )

    assert result["type"] == "abort"
    assert result["reason"] == "already_configured"
    assert entry.data["discovered_analyzer_url"] == "http://old:8099"
    flow.hass.config_entries.async_update_entry.assert_not_called()


@pytest.mark.asyncio
async def test_hassio_no_existing_entry_shows_user_form(_stub_schema):
    """No existing entry → the normal user setup form with the URL recorded."""
    flow = _make_flow([])

    result = await flow.async_step_hassio(
        _HassioInfo({"host": "abc123-ax-bpm-analyzer", "port": 8099})
    )

    assert result["type"] == "form"
    assert result["step_id"] == "user"
    assert flow._discovered_url == "http://abc123-ax-bpm-analyzer:8099"


@pytest.mark.asyncio
async def test_hassio_missing_host_port_falls_back_to_user_step(_stub_schema):
    """A discovery announcement without host/port → manual setup form."""
    flow = _make_flow([])

    result = await flow.async_step_hassio(_HassioInfo({}))

    assert result["type"] == "form"
    assert result["step_id"] == "user"
    assert flow._discovered_url is None
