"""Tests for the manual BPM override suite.

Covers:
- BpmOverrideStore CRUD (put/get/remove/clear) + persistence via mocked Store
- override_key stability (normalization, no duration component)
- Pipeline override-first ordering: override beats cache AND Deezer
- async_apply_correction math (halve/double) + invalid factor rejection
- clear_cache / clear_overrides independence
- override survives a "restart" (fresh store instance, same mocked file)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ax_bpm.const import (
    RULE_MANUAL_DOUBLE,
    RULE_MANUAL_HALF,
    SOURCE_OVERRIDE,
)
from ax_bpm.overrides import BpmOverrideStore, override_key
from ax_bpm.pipeline import BpmPipeline
from ax_bpm.store import BpmCache, cache_key

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_hass() -> MagicMock:
    return MagicMock()


def _mock_storage(backing: dict) -> MagicMock:
    """An in-memory HA Store mock over `backing` (simulates the file)."""
    storage = MagicMock()
    storage.async_load = AsyncMock(return_value=dict(backing))

    async def _save(data: dict) -> None:
        backing.clear()
        backing.update(data)

    storage.async_save = AsyncMock(side_effect=_save)
    return storage


def _make_store_with_file(initial: dict | None = None) -> tuple:
    """A BpmOverrideStore whose HA Store is an in-memory mock.

    Returns (store, backing_dict) — backing_dict simulates the on-disk
    file so a "restart" (new instance over the same backing) is testable.
    """
    backing: dict = dict(initial or {})
    store = BpmOverrideStore(_fake_hass())
    store._store = _mock_storage(backing)
    return store, backing


def _make_cache_with_file(initial: dict | None = None) -> tuple:
    backing: dict = dict(initial or {})
    cache = BpmCache(_fake_hass())
    cache._store = _mock_storage(backing)
    return cache, backing


# ---------------------------------------------------------------------------
# override_key
# ---------------------------------------------------------------------------

def test_override_key_normalizes_and_ignores_duration():
    k1 = override_key("Alicia Keys", "A Woman's Worth")
    k2 = override_key("alicia  keys", "a woman's worth")
    assert k1 == k2
    # No duration parameter exists at all — signature enforces it.
    assert k1.startswith("ovr:")


# ---------------------------------------------------------------------------
# BpmOverrideStore CRUD
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_get_roundtrip():
    store, _ = _make_store_with_file()
    await store.async_put("Alicia Keys", "A Woman's Worth", 75.0, RULE_MANUAL_HALF, 150.0)
    entry = store.get("Alicia Keys", "A Woman's Worth")
    assert entry == {
        "bpm": 75.0,
        "rule": RULE_MANUAL_HALF,
        "original_bpm": 150.0,
        "created": entry["created"],  # timestamp present
    }
    assert entry["created"]  # non-empty ISO timestamp


@pytest.mark.asyncio
async def test_get_miss_returns_none():
    store, _ = _make_store_with_file()
    assert store.get("Nobody", "Nothing") is None


@pytest.mark.asyncio
async def test_remove():
    store, _ = _make_store_with_file()
    await store.async_put("A", "B", 80.0, RULE_MANUAL_DOUBLE, 40.0)
    assert await store.async_remove("A", "B") is True
    assert store.get("A", "B") is None
    assert await store.async_remove("A", "B") is False


@pytest.mark.asyncio
async def test_clear_returns_count_and_empties():
    store, _ = _make_store_with_file()
    await store.async_put("A", "B", 80.0, RULE_MANUAL_DOUBLE, 40.0)
    await store.async_put("C", "D", 60.0, RULE_MANUAL_HALF, 120.0)
    assert await store.async_clear() == 2
    assert store.count() == 0
    assert await store.async_clear() == 0


@pytest.mark.asyncio
async def test_put_rejects_bad_rule_and_bpm():
    store, _ = _make_store_with_file()
    with pytest.raises(ValueError):
        await store.async_put("A", "B", 80.0, "genre_half", 160.0)
    with pytest.raises(ValueError):
        await store.async_put("A", "B", 0.0, RULE_MANUAL_HALF, 0.0)


@pytest.mark.asyncio
async def test_override_survives_restart():
    store, backing = _make_store_with_file()
    await store.async_put("Alicia Keys", "A Woman's Worth", 75.0, RULE_MANUAL_HALF, 150.0)

    # "Restart": a fresh instance loading the same backing file.
    store2, _ = _make_store_with_file(initial=backing)
    await store2.async_load()
    entry = store2.get("Alicia Keys", "A Woman's Worth")
    assert entry and entry["bpm"] == 75.0


# ---------------------------------------------------------------------------
# Pipeline: override-first ordering
# ---------------------------------------------------------------------------

def _make_pipeline(cache, overrides=None) -> BpmPipeline:
    """Pipeline with all network clients mocked out (never called here)."""
    hass = _fake_hass()
    session = MagicMock()
    pipeline = BpmPipeline(hass, session, cache, {}, overrides)
    # Stub the network clients so any accidental call fails loudly.
    pipeline._deezer = MagicMock()
    pipeline._deezer.find_match = AsyncMock(side_effect=AssertionError("Deezer called"))
    pipeline._analyzer = MagicMock()
    return pipeline


@pytest.mark.asyncio
async def test_override_beats_cache():
    cache, _ = _make_cache_with_file()
    overrides, _ = _make_store_with_file()
    await cache.async_put(
        cache_key(None, "Alicia Keys", "A Woman's Worth", 302),
        {"bpm": 149.8, "source": "deezer_metadata"},
    )
    await overrides.async_put(
        "Alicia Keys", "A Woman's Worth", 75.0, RULE_MANUAL_HALF, 149.8
    )
    pipeline = _make_pipeline(cache, overrides)
    result = await pipeline.async_resolve("Alicia Keys", "A Woman's Worth", 302)
    assert result is not None
    assert result.bpm == 75.0
    assert result.source == SOURCE_OVERRIDE
    assert result.attrs["octave_rule"] == RULE_MANUAL_HALF
    assert result.attrs["original_bpm"] == 149.8


@pytest.mark.asyncio
async def test_no_override_falls_through_to_cache():
    cache, _ = _make_cache_with_file()
    overrides, _ = _make_store_with_file()
    await cache.async_put(
        cache_key(None, "Alicia Keys", "A Woman's Worth", 302),
        {"bpm": 149.8, "source": "deezer_metadata"},
    )
    pipeline = _make_pipeline(cache, overrides)
    result = await pipeline.async_resolve("Alicia Keys", "A Woman's Worth", 302)
    assert result is not None
    assert result.bpm == 149.8
    assert result.source == "cache"


@pytest.mark.asyncio
async def test_pipeline_without_overrides_still_works():
    cache, _ = _make_cache_with_file()
    pipeline = _make_pipeline(cache, None)
    assert pipeline.overrides is None
    assert await pipeline.async_clear_overrides() == 0
    assert await pipeline.async_clear_override("A", "B") is False


# ---------------------------------------------------------------------------
# async_apply_correction math
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_apply_correction_halve_and_double():
    cache, _ = _make_cache_with_file()
    overrides, _ = _make_store_with_file()
    pipeline = _make_pipeline(cache, overrides)

    corrected = await pipeline.async_apply_correction(
        "Alicia Keys", "A Woman's Worth", 149.8, 0.5
    )
    assert corrected == 74.9
    entry = overrides.get("Alicia Keys", "A Woman's Worth")
    assert entry["rule"] == RULE_MANUAL_HALF
    assert entry["original_bpm"] == 149.8

    corrected = await pipeline.async_apply_correction(
        "A", "B", 75.0, 2.0
    )
    assert corrected == 150.0
    entry = overrides.get("A", "B")
    assert entry["rule"] == RULE_MANUAL_DOUBLE


@pytest.mark.asyncio
async def test_apply_correction_rejects_other_factors():
    cache, _ = _make_cache_with_file()
    overrides, _ = _make_store_with_file()
    pipeline = _make_pipeline(cache, overrides)
    with pytest.raises(ValueError):
        await pipeline.async_apply_correction("A", "B", 100.0, 3.0)


# ---------------------------------------------------------------------------
# clear_cache / clear_overrides independence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_clear_cache_leaves_overrides():
    cache, _ = _make_cache_with_file()
    overrides, _ = _make_store_with_file()
    await cache.async_put(cache_key(None, "A", "B", None), {"bpm": 100.0})
    await overrides.async_put("A", "B", 50.0, RULE_MANUAL_HALF, 100.0)

    pipeline = _make_pipeline(cache, overrides)
    assert await pipeline.async_clear_cache() == 1
    assert cache.get("hash:fake") is None
    assert overrides.get("A", "B") is not None  # untouched


@pytest.mark.asyncio
async def test_clear_overrides_leaves_cache():
    cache, _ = _make_cache_with_file()
    overrides, _ = _make_store_with_file()
    await cache.async_put(cache_key(None, "A", "B", None), {"bpm": 100.0})
    await overrides.async_put("A", "B", 50.0, RULE_MANUAL_HALF, 100.0)

    pipeline = _make_pipeline(cache, overrides)
    assert await pipeline.async_clear_overrides() == 1
    assert overrides.get("A", "B") is None
    assert cache.get(cache_key(None, "A", "B", None)) is not None  # untouched


@pytest.mark.asyncio
async def test_clear_override_restores_cache_value():
    """After clearing the override, the pipeline falls back to the cache."""
    cache, _ = _make_cache_with_file()
    overrides, _ = _make_store_with_file()
    await cache.async_put(
        cache_key(None, "Alicia Keys", "A Woman's Worth", 302),
        {"bpm": 149.8},
    )
    pipeline = _make_pipeline(cache, overrides)

    await pipeline.async_apply_correction(
        "Alicia Keys", "A Woman's Worth", 149.8, 0.5
    )
    result = await pipeline.async_resolve("Alicia Keys", "A Woman's Worth", 302)
    assert result.bpm == 74.9

    await pipeline.async_clear_override("Alicia Keys", "A Woman's Worth")
    result = await pipeline.async_resolve("Alicia Keys", "A Woman's Worth", 302)
    assert result.bpm == 149.8
    assert result.source == "cache"
