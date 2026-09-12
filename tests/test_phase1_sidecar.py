"""Phase 1 tests: mood_client, cache v1→v2 migration, config-flow migration.

mood_client mock matrix (parent plan Phase 3 item 1):
- healthy / slow (timeout) / HTTP 500 / unreachable — the client returns
  None in every failure mode and never raises.
Cache migration: legacy SVM mood fields dropped, BPM kept.
Config flow: legacy genre/mood toggles → octave_disambiguation dropdown.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from ax_bpm.config_flow import _migrate_legacy_toggles
from ax_bpm.const import (
    OCTAVE_GENRE_MOOD,
    OCTAVE_GENRE_ONLY,
    OCTAVE_OFF,
)
from ax_bpm.mood_client import MoodClient
from ax_bpm.store import LEGACY_MOOD_FIELDS, BpmCache

# ---------------------------------------------------------------------------
# MoodClient — failure matrix
# ---------------------------------------------------------------------------


def _make_client(manual_url: str | None = None) -> tuple[MoodClient, MagicMock]:
    session = MagicMock()
    return MoodClient(session, manual_url), session


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
    client, session = _make_client("http://sidecar:8099")
    payload = {"mood_tags": [{"tag": "party", "score": 0.9}], "valence": 0.5}
    session.post = MagicMock(return_value=_Ctx(_resp(200, payload)))
    result = await client.async_analyze(b"mp3")
    assert result == payload


@pytest.mark.asyncio
async def test_analyze_http_500_returns_none():
    client, session = _make_client("http://sidecar:8099")
    session.post = MagicMock(return_value=_Ctx(_resp(500)))
    assert await client.async_analyze(b"mp3") is None


@pytest.mark.asyncio
async def test_analyze_timeout_returns_none():
    client, session = _make_client("http://sidecar:8099")

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
    client, session = _make_client("http://sidecar:8099")
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
        __import__("ax_bpm.const", fromlist=["SIDECAR_URLS"]).SIDECAR_URLS
    )


@pytest.mark.asyncio
async def test_detect_cached_after_first_probe():
    client, session = _make_client("http://manual:8099")
    session.get = MagicMock(return_value=_Ctx(_resp(200, {"status": "ok"})))
    assert await client.async_detect() == "http://manual:8099"
    assert await client.async_detect() == "http://manual:8099"
    assert session.get.call_count == 1


# ---------------------------------------------------------------------------
# Cache v1 → v2 migration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_migration_drops_legacy_mood_fields():
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
    entry = cache.get("isrc:GBDUW0000059")
    assert entry is not None
    assert entry["bpm"] == 174.0  # BPM kept
    assert entry["track"] == "Daft Punk - Test Track"
    for field in LEGACY_MOOD_FIELDS:
        assert field not in entry
    # Non-mood entry untouched.
    assert cache.get("hash:abc")["bpm"] == 120.0


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