"""Tests for the manual BPM + mood override suite.

Covers:
- BpmOverrideStore CRUD (put/get/remove/clear) + persistence via mocked Store
- override_key stability (normalization, no duration component)
- Pipeline override-first ordering: override beats cache AND Deezer
- async_apply_correction math (halve/double) + invalid factor rejection
- clear_cache / clear_overrides independence
- override survives a "restart" (fresh store instance, same mocked file)
- 3.3.0: manual_set rule, mood overrides (set/merge/clear), synthetic
  one-hot attrs with formula parity, enrichment skip, service handlers
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ax_bpm.const import (
    RULE_MANUAL_DOUBLE,
    RULE_MANUAL_HALF,
    RULE_MANUAL_SET,
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


# ---------------------------------------------------------------------------
# 3.3.0: manual_set (exact BPM) + mood overrides — store level
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_set_bpm_stores_manual_set_rule():
    store, _ = _make_store_with_file()
    await store.async_put("A", "B", 174.0, RULE_MANUAL_SET, 174.0)
    entry = store.get("A", "B")
    assert entry["bpm"] == 174.0
    assert entry["rule"] == RULE_MANUAL_SET


@pytest.mark.asyncio
async def test_set_mood_creates_mood_only_entry():
    store, _ = _make_store_with_file()
    await store.async_set_mood("A", "B", "relaxed")
    entry = store.get("A", "B")
    assert entry["mood_label"] == "relaxed"
    assert "bpm" not in entry  # mood-only: no BPM aspect


@pytest.mark.asyncio
async def test_set_mood_merges_into_existing_bpm_entry():
    store, _ = _make_store_with_file()
    await store.async_put("A", "B", 174.0, RULE_MANUAL_SET, 174.0)
    created = store.get("A", "B")["created"]
    await store.async_set_mood("A", "B", "party")
    entry = store.get("A", "B")
    assert entry["bpm"] == 174.0
    assert entry["rule"] == RULE_MANUAL_SET
    assert entry["mood_label"] == "party"
    assert entry["created"] == created  # first-write timestamp preserved


@pytest.mark.asyncio
async def test_bpm_put_merges_into_existing_mood_entry():
    store, _ = _make_store_with_file()
    await store.async_set_mood("A", "B", "relaxed")
    await store.async_put("A", "B", 90.0, RULE_MANUAL_HALF, 180.0)
    entry = store.get("A", "B")
    assert entry["mood_label"] == "relaxed"  # mood survives the BPM write
    assert entry["bpm"] == 90.0


@pytest.mark.asyncio
async def test_set_mood_rejects_invalid_label():
    store, _ = _make_store_with_file()
    with pytest.raises(ValueError):
        await store.async_set_mood("A", "B", "happy")


@pytest.mark.asyncio
async def test_clear_removes_both_aspects():
    store, _ = _make_store_with_file()
    await store.async_put("A", "B", 174.0, RULE_MANUAL_SET, 174.0)
    await store.async_set_mood("A", "B", "acoustic")
    assert await store.async_remove("A", "B") is True
    assert store.get("A", "B") is None


# ---------------------------------------------------------------------------
# 3.3.0: pipeline — synthetic mood attrs + replacement on all paths
# ---------------------------------------------------------------------------

def _mood_payload() -> dict:
    """An analyzer payload with a complete five-head set (automatic mood)."""
    return {
        "mood_scores": {
            "aggressive": 0.9, "party": 0.8, "relaxed": 0.1,
            "electronic": 0.7, "acoustic": 0.0,
        },
        "valence": 0.6, "arousal": 0.8,
    }


@pytest.mark.asyncio
async def test_mood_only_override_replaces_cache_mood():
    cache, _ = _make_cache_with_file()
    overrides, _ = _make_store_with_file()
    await cache.async_put(
        cache_key(None, "A", "B", 302),
        {"bpm": 120.0, "source": "cache", **_mood_attrs_from_payload(_mood_payload())},
    )
    await overrides.async_set_mood("A", "B", "relaxed")
    pipeline = _make_pipeline(cache, overrides)
    result = await pipeline.async_resolve("A", "B", 302)
    assert result.bpm == 120.0  # BPM untouched (mood-only override)
    assert result.source == "cache"
    assert result.attrs["mood_label"] == "relaxed"
    assert result.attrs["mood_scores"] == {
        "aggressive": 0.0, "party": 0.0, "relaxed": 1.0,
        "electronic": 0.0, "acoustic": 0.0,
    }
    assert result.attrs["intensity"] == 0.0
    assert result.attrs["calmness"] == 0.5
    assert result.attrs["mood_override"] is True


@pytest.mark.asyncio
async def test_bpm_and_mood_override_hit_publishes_both():
    cache, _ = _make_cache_with_file()
    overrides, _ = _make_store_with_file()
    await overrides.async_put("A", "B", 174.0, RULE_MANUAL_SET, 140.0)
    await overrides.async_set_mood("A", "B", "electronic")
    pipeline = _make_pipeline(cache, overrides)
    result = await pipeline.async_resolve("A", "B", 302)
    assert result.bpm == 174.0
    assert result.source == SOURCE_OVERRIDE
    assert result.attrs["mood_label"] == "electronic"
    assert result.attrs["mood_override"] is True
    assert result.attrs["intensity"] == 0.333  # (1+0+0)/3, rounded to 3 dp
    assert result.attrs["calmness"] == 0.0


@pytest.mark.asyncio
async def test_no_mood_override_keeps_automatic_mood():
    cache, _ = _make_cache_with_file()
    overrides, _ = _make_store_with_file()
    automatic = _mood_attrs_from_payload(_mood_payload())
    await cache.async_put(
        cache_key(None, "A", "B", 302),
        {"bpm": 120.0, "source": "cache", **automatic},
    )
    pipeline = _make_pipeline(cache, overrides)
    result = await pipeline.async_resolve("A", "B", 302)
    assert result.attrs["mood_label"] == automatic["mood_label"]
    assert "mood_override" not in result.attrs


@pytest.mark.asyncio
async def test_enrichment_skipped_when_mood_override_exists():
    cache, _ = _make_cache_with_file()
    overrides, _ = _make_store_with_file()
    await overrides.async_set_mood("A", "B", "relaxed")
    pipeline = _make_pipeline(cache, overrides)
    pipeline._mood_ready = True
    pipeline._octave_mode = "genre_mood"
    from ax_bpm.pipeline import ResolutionResult

    result = ResolutionResult(120.0, source="deezer_metadata", isrc="ISRC123")
    assert await pipeline.async_enrich("A", "B", 302, result) is None


def _mood_attrs_from_payload(payload: dict) -> dict:
    """Mirror of BpmPipeline._mood_attrs_from_payload for test fixtures."""
    scores = payload["mood_scores"]
    intensity = (scores["aggressive"] + scores["party"] + scores["electronic"]) / 3
    calmness = (scores["relaxed"] + scores["acoustic"]) / 2
    return {
        "mood_scores": scores,
        "mood_label": max(scores, key=scores.get),
        "intensity": round(intensity, 3),
        "calmness": round(calmness, 3),
    }


# ---------------------------------------------------------------------------
# 3.3.0: formula parity — override path vs analyzer-payload path
# ---------------------------------------------------------------------------

def test_formula_parity_override_vs_payload():
    """One-hot override scores through the payload derivation must equal
    the override derivation (structural parity, not copy-paste)."""
    from ax_bpm.pipeline import BpmPipeline as BP

    override_attrs = BP._mood_attrs_from_override("relaxed")
    payload_attrs = BP._mood_attrs_from_payload({
        "mood_scores": override_attrs["mood_scores"],
        "arousal": None,
    })
    assert override_attrs["intensity"] == payload_attrs["intensity"]
    assert override_attrs["calmness"] == payload_attrs["calmness"]
    assert override_attrs["mood_label"] == payload_attrs["mood_label"]


# ---------------------------------------------------------------------------
# 3.3.0: pipeline async_set_bpm / async_set_mood
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_async_set_bpm():
    cache, _ = _make_cache_with_file()
    overrides, _ = _make_store_with_file()
    pipeline = _make_pipeline(cache, overrides)
    stored = await pipeline.async_set_bpm("A", "B", 174.0)
    assert stored == 174.0
    entry = overrides.get("A", "B")
    assert entry["rule"] == RULE_MANUAL_SET
    with pytest.raises(ValueError):
        await pipeline.async_set_bpm("A", "B", 0)


@pytest.mark.asyncio
async def test_pipeline_async_set_mood():
    cache, _ = _make_cache_with_file()
    overrides, _ = _make_store_with_file()
    pipeline = _make_pipeline(cache, overrides)
    await pipeline.async_set_mood("A", "B", "party")
    assert overrides.get("A", "B")["mood_label"] == "party"
    with pytest.raises(ValueError):
        await pipeline.async_set_mood("A", "B", "happy")
