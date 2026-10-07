"""Regression tests for the mood-degeneracy fix (Phase 3).

Pins the four parallel defects fixed alongside the decode scale root fix:

1. Attribute-assembly parity: the local path (_resolve_local) and the
   enrichment path (async_enrich) must derive mood attributes from the
   SAME _mood_attrs_from_payload — fresh-path attrs == enrich-path attrs
   for the same analyzer payload.
2. Formula parity: published intensity/calmness use the pinned /3, /2
   formula (test_formula_parity_regression in test_phase2_atomicity.py
   pins the values; here we pin that the local path emits the SAME
   values as _mood_attrs_from_payload).
3. mood_label wiring: argmax ONLY over the COMPLETE five-head set
   (atomic rule) — an incomplete set emits NO mood_label.
4. Sensor unknown state: a failed resolution clears stale per-track
   attributes (never presents old valid-looking metadata with unknown).
5. Cache payload versioning: an entry written with a different (or
   missing) payload_version is treated as absent — a previously-cached
   pre-fix ISRC must NOT reproduce its pre-fix values.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from test_pipeline import (
    SEARCH_RESPONSE,
    TRACK_RESPONSE_BPM0,
    make_pipeline,
)

from ax_bpm.const import CACHE_PAYLOAD_VERSION, EXPECTED_MOOD_SCORES
from ax_bpm.pipeline import BpmPipeline
from ax_bpm.store import BpmCache, cache_key


@pytest.fixture
def mock_session():
    """Local fixture (importing test_pipeline's would trip ruff F811)."""
    return MagicMock()


@pytest.fixture
def mock_cache():
    cache = MagicMock(spec=BpmCache)
    cache.get = MagicMock(return_value=None)
    cache.async_put = AsyncMock()
    return cache


def _complete_scores(**overrides) -> dict:
    """A complete five-head mood_scores dict (atomic set)."""
    scores = {
        "aggressive": 0.1,
        "party": 0.2,
        "relaxed": 0.7,
        "electronic": 0.3,
        "acoustic": 0.6,
    }
    scores.update(overrides)
    return scores


def _analyzer_payload(bpm: float, scores: dict | None) -> dict:
    payload = {
        "bpm": bpm,
        "bpm_confidence": 0.9,
        "mood_tags": [
            {"tag": k, "score": v} for k, v in (scores or {}).items()
        ],
        "valence": 0.4,
        "arousal": 0.8,
        "valence_std": None,
        "arousal_std": None,
        "analyzed_seconds": 30.0,
        "model_versions": {},
    }
    if scores is not None:
        payload["mood_scores"] = scores
    return payload


# ---------------------------------------------------------------------------
# Defects 1+2+3: single derivation, parity between paths
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_local_path_attrs_match_enrich_path_attrs(
    mock_session, mock_cache
):
    """Parity: _resolve_local's published mood attrs == _mood_attrs_from_payload.

    The local path must merge the SAME derivation the enrichment path
    uses — no hand-rolled mood_scores/mood_label/intensity/calmness.
    """
    pipeline = make_pipeline(mock_session, mock_cache, mood_ready=True)
    pipeline._deezer.find_match = AsyncMock(return_value=SEARCH_RESPONSE["data"][0])
    pipeline._deezer.get_track = AsyncMock(return_value=TRACK_RESPONSE_BPM0)
    pipeline._deezer.get_album_genres = AsyncMock(return_value=[])
    pipeline._deezer.download_preview = AsyncMock(return_value=True)
    scores = _complete_scores()
    payload = _analyzer_payload(103.0, scores)
    pipeline._analyzer_client.async_analyze_file = AsyncMock(return_value=payload)

    with __import__("unittest").mock.patch(
        "ax_bpm.pipeline.tempfile.mkstemp", return_value=(99, "/tmp/fake.mp3")
    ), __import__("unittest").mock.patch("ax_bpm.pipeline.os.close"), \
            __import__("unittest").mock.patch("ax_bpm.pipeline.os.unlink"):
        result = await pipeline.async_resolve("Daft Punk", "Test Track", 224.0)

    assert result is not None
    expected = BpmPipeline._mood_attrs_from_payload(payload)
    expected.pop("source", None)  # result-level source is set separately

    for key, value in expected.items():
        assert result.attrs[key] == value, (
            f"local-path attr {key!r} diverges from the single derivation: "
            f"{result.attrs.get(key)!r} != {value!r}"
        )
    # The published intensity/calmness are the /3, /2 formula values
    # (defect 2): intensity = (0.1+0.2+0.3)/3, calmness = (0.7+0.6)/2.
    assert result.attrs["intensity"] == pytest.approx(
        (0.1 + 0.2 + 0.3) / 3.0, abs=1e-3
    )
    assert result.attrs["calmness"] == pytest.approx((0.7 + 0.6) / 2.0)


@pytest.mark.asyncio
async def test_mood_label_argmax_only_over_complete_set(mock_session, mock_cache):
    """mood_label fires ONLY over the COMPLETE five-head set (atomic rule).

    An incomplete set (e.g. tags-derived scores missing a head) must emit
    NO mood_label — and no mood_scores either (genre-only fallback).
    """
    pipeline = make_pipeline(mock_session, mock_cache, mood_ready=True)
    pipeline._deezer.find_match = AsyncMock(return_value=SEARCH_RESPONSE["data"][0])
    pipeline._deezer.get_track = AsyncMock(return_value=TRACK_RESPONSE_BPM0)
    pipeline._deezer.get_album_genres = AsyncMock(return_value=[])
    pipeline._deezer.download_preview = AsyncMock(return_value=True)
    # Payload with mood_tags only (no mood_scores key) → tags-derived
    # scores are INCOMPLETE over the five-head set.
    payload = _analyzer_payload(103.0, None)
    payload["mood_tags"] = [
        {"tag": "aggressive", "score": 0.9},
        {"tag": "party", "score": 0.8},
    ]
    pipeline._analyzer_client.async_analyze_file = AsyncMock(return_value=payload)

    with __import__("unittest").mock.patch(
        "ax_bpm.pipeline.tempfile.mkstemp", return_value=(99, "/tmp/fake.mp3")
    ), __import__("unittest").mock.patch("ax_bpm.pipeline.os.close"), \
            __import__("unittest").mock.patch("ax_bpm.pipeline.os.unlink"):
        result = await pipeline.async_resolve("Daft Punk", "Test Track", 224.0)

    assert result is not None
    assert "mood_label" not in result.attrs
    assert "mood_scores" not in result.attrs

    # Complete set → mood_label present and equals the argmax.
    scores = _complete_scores(aggressive=0.95)
    payload2 = _analyzer_payload(103.0, scores)
    pipeline._analyzer_client.async_analyze_file = AsyncMock(return_value=payload2)
    with __import__("unittest").mock.patch(
        "ax_bpm.pipeline.tempfile.mkstemp", return_value=(99, "/tmp/fake.mp3")
    ), __import__("unittest").mock.patch("ax_bpm.pipeline.os.close"), \
            __import__("unittest").mock.patch("ax_bpm.pipeline.os.unlink"):
        result2 = await pipeline.async_resolve("Daft Punk", "Test Track", 224.0)
    assert result2 is not None
    assert result2.attrs["mood_label"] == "aggressive"


def test_mood_label_deterministic_tiebreak():
    """Ties resolve deterministically via sorted-key order."""
    scores = _complete_scores(aggressive=0.5, relaxed=0.5)
    attrs = BpmPipeline._mood_attrs_from_payload({"mood_scores": scores})
    five = {k: float(scores[k]) for k in sorted(EXPECTED_MOOD_SCORES)}
    assert attrs["mood_label"] == max(five, key=five.get)


# ---------------------------------------------------------------------------
# Defect 4: sensor clears stale attributes on unknown
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_sensor_unknown_clears_stale_attributes():
    """A failed resolution → unknown with EMPTY per-track attributes."""
    from ax_bpm.sensor import AxBpmSensor

    hass = MagicMock()
    state = MagicMock()
    state.state = "playing"
    state.attributes = {
        "media_artist": "Artist",
        "media_title": "Title",
        "media_duration": 180,
    }
    hass.states.get = MagicMock(return_value=state)

    pipeline = MagicMock()
    pipeline.async_resolve = AsyncMock(return_value=None)
    pipeline.async_enrich = AsyncMock(return_value=None)

    sensor = AxBpmSensor.__new__(AxBpmSensor)
    sensor.hass = hass
    sensor._media_player_id = "media_player.test"
    sensor._pipeline = pipeline
    sensor._debounce_unsub = None
    sensor._resolving_track = None
    sensor._last_track_id = None
    sensor.async_write_ha_state = MagicMock()  # HA entity method (stubbed)
    # Simulate the stale attributes a PREVIOUS successful track left.
    sensor._attr_native_value = 120.0
    sensor._attr_extra_state_attributes = {
        "mood_scores": {"aggressive": 0.9},
        "mood_label": "aggressive",
        "isrc": "GBDUW0000059",
    }

    await sensor._async_resolve()

    assert sensor._attr_native_value is None
    assert sensor._attr_extra_state_attributes == {}, (
        "stale per-track attributes survived the unknown state (defect 4)"
    )


# ---------------------------------------------------------------------------
# Cache payload versioning
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cache_version_mismatch_invalidates_entry():
    """A pre-fix cached entry (wrong/missing payload_version) reads as absent.

    Post-fix analysis of a previously-cached ISRC must NOT reproduce the
    pre-fix degenerate values — the stale entry is ignored and the
    re-analysis overwrites it.
    """
    class _FakeStore:
        def __init__(self):
            self.data = {}

        async def async_load(self):
            return dict(self.data)

        async def async_save(self, data):
            self.data = dict(data)

    fake = _FakeStore()
    cache = BpmCache.__new__(BpmCache)
    cache._store = fake
    cache._data = {}

    key = cache_key("GBDUW0000059", "Artist", "Title", 224.0)

    # Pre-fix entry: degenerate mood values, WRONG payload_version.
    await cache.async_put(key, {"bpm": 103.0, "mood_scores": {
        "aggressive": 0.93, "party": 0.98, "relaxed": 0.006,
        "electronic": 0.98, "acoustic": 0.008,
    }})
    # Simulate the on-disk pre-fix state: version 1 stamped.
    fake.data[key]["payload_version"] = CACHE_PAYLOAD_VERSION - 1

    assert cache.get(key) is None, (
        "pre-fix cache entry with a stale payload_version was served"
    )

    # Missing payload_version entirely (pre-versioning era) → also absent.
    fake.data[key].pop("payload_version")
    assert cache.get(key) is None

    # Current-version entry → served.
    await cache.async_put(key, {"bpm": 103.0, "mood_scores": {
        "aggressive": 0.12, "party": 0.31, "relaxed": 0.55,
        "electronic": 0.28, "acoustic": 0.61,
    }})
    served = cache.get(key)
    assert served is not None
    assert served["mood_scores"]["aggressive"] == 0.12
    assert served["source"] == "cache"


@pytest.mark.asyncio
async def test_cache_hit_requires_current_version_in_pipeline(
    mock_session, mock_cache
):
    """Pipeline cache-hit path: a version-mismatched entry is NOT served.

    Uses a REAL BpmCache (fake Store) so the store-level version check
    runs — a MagicMock(spec=BpmCache) would bypass it.
    """
    pipeline = make_pipeline(mock_session, mock_cache, mood_ready=True)

    class _FakeStore:
        def __init__(self):
            self.data = {}

        async def async_load(self):
            return dict(self.data)

        async def async_save(self, data):
            self.data = dict(data)

    real_cache = BpmCache.__new__(BpmCache)
    real_cache._store = _FakeStore()
    real_cache._data = {}
    key = cache_key(None, "Daft Punk", "Test Track", 224.0)
    # Pre-fix entry: degenerate mood values, stale payload_version.
    real_cache._data[key] = {
        "bpm": 103.0,
        "mood_scores": {
            "aggressive": 0.93, "party": 0.98, "relaxed": 0.006,
            "electronic": 0.98, "acoustic": 0.008,
        },
        "payload_version": CACHE_PAYLOAD_VERSION - 1,
    }
    pipeline._cache = real_cache
    pipeline._deezer.find_match = AsyncMock(return_value=None)

    result = await pipeline.async_resolve("Daft Punk", "Test Track", 224.0)

    # The stale entry must not be published; the pipeline proceeds to the
    # Deezer match (mocked empty here → None).
    assert result is None
    pipeline._deezer.find_match.assert_awaited_once()
    # Sanity: a current-version entry IS served from cache.
    real_cache._data[key]["payload_version"] = CACHE_PAYLOAD_VERSION
    served = real_cache.get(key)
    assert served is not None and served["bpm"] == 103.0
