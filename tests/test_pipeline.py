"""Integration tests: mocked Deezer responses + synthetic audio fixture.

Covers:
- mocked Deezer search/track/album (incl. the bpm:0 case)
- Deezer metadata bpm published as-is, never corrected
- bpm:0 → local analysis fallback path (aubio + mood mocked)
- cache hit → no network calls
- failure → None (never 0)
- synthetic audio fixture exercising the aubio analyzer end-to-end
  (skipped when the aubio package is not installed)
"""

from __future__ import annotations

import asyncio
import math
import struct
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ax_bpm.deezer import DeezerClient, clean_title, parse_artist_title  # noqa: E402
from ax_bpm.math import RULE_NONE  # noqa: E402
from ax_bpm.pipeline import BpmPipeline  # noqa: E402
from ax_bpm.store import BpmCache, cache_key  # noqa: E402


def _aubio_importable() -> bool:
    try:
        import aubio  # noqa: F401, PLC0415
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SEARCH_RESPONSE = {
    "data": [
        {
            "id": 3135556,
            "title": "Harder, Better, Faster, Stronger",
            "duration": 224,
            "rank": 951234,
            "artist": {"name": "Daft Punk"},
            "album": {"id": 302127},
        }
    ]
}

TRACK_RESPONSE_BPM0 = {
    "id": 3135556,
    "title": "Harder, Better, Faster, Stronger",
    "duration": 224,
    "rank": 951234,
    "bpm": 0.0,  # verified live: top-tier catalog frequently returns 0
    "isrc": "GBDUW0000059",
    "preview": "https://cdns-preview.dzcdn.net/stream/fake.mp3",
    "album": {"id": 302127},
}

TRACK_RESPONSE_BPM123 = {
    **TRACK_RESPONSE_BPM0,
    "bpm": 123.0,
}

ALBUM_RESPONSE = {
    "id": 302127,
    "genres": {"data": [{"name": "Electro"}]},
}


@pytest.fixture
def mock_session():
    return MagicMock()


@pytest.fixture
def mock_cache():
    cache = MagicMock(spec=BpmCache)
    cache.get = MagicMock(return_value=None)
    cache.async_put = AsyncMock()
    return cache


def make_pipeline(mock_session, mock_cache, mood_ready=False):
    pipeline = BpmPipeline.__new__(BpmPipeline)
    pipeline._hass = MagicMock()
    pipeline._hass.config.path = MagicMock(return_value="/tmp/ax_bpm_models")
    pipeline._deezer = MagicMock(spec=DeezerClient)
    pipeline._cache = mock_cache
    pipeline._aubio = MagicMock()
    pipeline._aubio.get_bpm = AsyncMock(return_value=None)
    pipeline._mood = MagicMock()
    pipeline._mood.get_scores = AsyncMock(return_value=None)
    pipeline._mood_ready = mood_ready
    pipeline._genre_correction = True
    pipeline._mood_correction = True
    pipeline._lock = asyncio.Lock()
    pipeline._current_token = None
    pipeline._last_match = None
    return pipeline


# ---------------------------------------------------------------------------
# Deezer client unit behavior
# ---------------------------------------------------------------------------


def test_clean_title_strips_noise():
    assert clean_title("One More Time (Radio Edit)") == "One More Time"
    assert clean_title("Harder, Better, Faster, Stronger (Remix)") == (
        "Harder, Better, Faster, Stronger"
    )
    assert clean_title("Around the World (feat. Nobody)") == "Around the World"
    assert clean_title("Digital Love") == "Digital Love"


def test_parse_artist_title_fallback():
    assert parse_artist_title("Daft Punk", "One More Time") == (
        "Daft Punk",
        "One More Time",
    )
    assert parse_artist_title(None, "Daft Punk - One More Time") == (
        "Daft Punk",
        "One More Time",
    )
    assert parse_artist_title(None, "No Separator Here") is None
    assert parse_artist_title(None, None) is None


@pytest.mark.asyncio
async def test_find_match_duration_filter_and_rank():
    client = DeezerClient(MagicMock())
    client.search_tracks = AsyncMock(
        return_value=[
            {"id": 1, "duration": 224, "rank": 100, "artist": {"name": "Daft Punk"}},
            {"id": 2, "duration": 300, "rank": 999999, "artist": {"name": "Daft Punk"}},  # out of ±3s
            {"id": 3, "duration": 225, "rank": 500, "artist": {"name": "Daft Punk"}},
        ]
    )
    # Candidates within ±3s tolerance: id 1 (rank 100) and id 3 (rank 500);
    # id 2 (duration 300) is filtered out. Highest rank wins → id 3.
    match = await client.find_match("Daft Punk", "Test", 224.0)
    assert match["id"] == 3


@pytest.mark.asyncio
async def test_search_tracks_falls_back_to_plain_text():
    """Field-quoted query returns nothing → plain-text retry is used."""
    client = DeezerClient(MagicMock())
    calls: list[str] = []

    async def fake_get_json(path, params=None):
        calls.append(params["q"])
        if "artist:" in params["q"]:
            return {"data": []}  # field-quoted → 0 results (Deezer 2026-09)
        return {"data": [{"id": 2511224, "title": "Kryptonite"}]}

    client._get_json = fake_get_json
    results = await client.search_tracks("3 Doors Down", "Kryptonite")
    assert results and results[0]["id"] == 2511224
    assert calls == [
        'artist:"3 Doors Down" track:"Kryptonite"',
        "3 Doors Down Kryptonite",
    ]


@pytest.mark.asyncio
async def test_find_match_rejects_wrong_artist():
    """Plain-text results from a different artist are never matched."""
    client = DeezerClient(MagicMock())
    client.search_tracks = AsyncMock(
        return_value=[
            # Right title/duration, WRONG artist — must be rejected.
            {"id": 9, "duration": 224, "rank": 9999999, "artist": {"name": "Some Cover Band"}},
            {"id": 3, "duration": 225, "rank": 500, "artist": {"name": "Daft Punk"}},
        ]
    )
    match = await client.find_match("Daft Punk", "Test", 224.0)
    assert match["id"] == 3


@pytest.mark.asyncio
async def test_find_match_artist_variant_accepted():
    """Artist variants (feat. suffixes, case) still verify."""
    client = DeezerClient(MagicMock())
    client.search_tracks = AsyncMock(
        return_value=[
            {"id": 5, "duration": 224, "rank": 100, "artist": {"name": "daft punk"}},
            {"id": 6, "duration": 224, "rank": 200, "artist": {"name": "Daft Punk feat. Someone"}},
        ]
    )
    match = await client.find_match("Daft Punk", "Test", 224.0)
    assert match["id"] == 6  # higher rank wins among verified candidates


# ---------------------------------------------------------------------------
# Pipeline: Deezer metadata path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deezer_bpm_published_as_is(mock_session, mock_cache):
    """Deezer metadata bpm present → published as-is, never corrected."""
    pipeline = make_pipeline(mock_session, mock_cache)
    pipeline._deezer.find_match = AsyncMock(return_value=SEARCH_RESPONSE["data"][0])
    pipeline._deezer.get_track = AsyncMock(return_value=TRACK_RESPONSE_BPM123)
    pipeline._deezer.get_album_genres = AsyncMock(return_value=["Electro"])

    result = await pipeline.async_resolve("Daft Punk", "Test Track", 224.0)
    assert result is not None
    assert result.bpm == 123.0
    assert result.source == "deezer_metadata"
    assert result.attrs["octave_corrected"] is False
    assert result.attrs["octave_rule"] == RULE_NONE
    assert result.attrs["preview_analyzed"] is False


@pytest.mark.asyncio
async def test_deezer_bpm_zero_falls_through_to_local(mock_session, mock_cache):
    """bpm:0 → local analysis fallback (aubio + mood mocked)."""
    pipeline = make_pipeline(mock_session, mock_cache, mood_ready=True)
    pipeline._deezer.find_match = AsyncMock(return_value=SEARCH_RESPONSE["data"][0])
    pipeline._deezer.get_track = AsyncMock(return_value=TRACK_RESPONSE_BPM0)
    pipeline._deezer.get_album_genres = AsyncMock(return_value=["Electro"])
    pipeline._deezer.download_preview = AsyncMock(return_value=True)
    pipeline._aubio.get_bpm = AsyncMock(return_value=87.0)
    pipeline._mood.get_scores = AsyncMock(
        return_value={"aggressive": 0.9, "party": 0.8, "electronic": 0.7}
    )

    with patch("ax_bpm.pipeline.tempfile.mkstemp", return_value=(99, "/tmp/fake.mp3")), patch(
        "ax_bpm.pipeline.os.close"
    ), patch("ax_bpm.pipeline.os.unlink"):
        result = await pipeline.async_resolve("Daft Punk", "Test Track", 224.0)

    assert result is not None
    assert result.source == "aubio"
    assert result.attrs["bpm_raw"] == 87.0
    # I = (.9+.8+.7+0)/4 = 0.6 ≥ T_APPLY → doubled (mood path).
    assert result.bpm == 174.0
    assert result.attrs["octave_rule"] == "mood_double"
    assert result.attrs["preview_analyzed"] is True


@pytest.mark.asyncio
async def test_no_match_returns_none_never_zero(mock_session, mock_cache):
    """A failed match can never produce sensor state 0 → None (unknown)."""
    pipeline = make_pipeline(mock_session, mock_cache)
    pipeline._deezer.find_match = AsyncMock(return_value=None)

    result = await pipeline.async_resolve("Unknown Artist", "Obscure Track", 200.0)
    assert result is None


@pytest.mark.asyncio
async def test_missing_artist_title_returns_none(mock_session, mock_cache):
    pipeline = make_pipeline(mock_session, mock_cache)
    result = await pipeline.async_resolve(None, None, None)
    assert result is None


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_hit_no_network(mock_session, mock_cache):
    """Cache hit → publish immediately, no network calls, no analysis."""
    pipeline = make_pipeline(mock_session, mock_cache)
    cached_entry = {
        "bpm": 174.0,
        "track": "Daft Punk - Test Track",
        "octave_rule": "mood_double",
        "isrc": "GBDUW0000059",
    }
    mock_cache.get = MagicMock(return_value=cached_entry)

    result = await pipeline.async_resolve("Daft Punk", "Test Track", 224.0)
    assert result is not None
    assert result.bpm == 174.0
    assert result.source == "cache"
    pipeline._deezer.find_match.assert_not_called()


def test_cache_key_isrc_preferred():
    assert cache_key("GBDUW0000059", "a", "b", 100).startswith("isrc:")
    assert cache_key(None, "Daft Punk", "One More Time", 300.0).startswith("hash:")
    # Same bucket (±3s granularity → 6s buckets) → same key.
    k1 = cache_key(None, "a", "b", 300.0)
    k2 = cache_key(None, "a", "b", 303.0)
    k3 = cache_key(None, "a", "b", 310.0)
    assert k1 == k2
    assert k1 != k3


# ---------------------------------------------------------------------------
# Synthetic audio fixture through the real aubio analyzer (end-to-end)
# ---------------------------------------------------------------------------


def _write_click_track(path: Path, bpm: float, seconds: float = 20.0) -> None:
    """Write a WAV of periodic clicks at the given BPM (44.1 kHz mono)."""
    rate = 44100
    beat_samples = int(rate * 60.0 / bpm)
    total = int(rate * seconds)
    frames = bytearray(total * 2)
    for i in range(0, total, beat_samples):
        for j in range(min(2000, total - i)):
            # Decaying click.
            sample = int(20000 * math.exp(-j / 300.0))
            struct.pack_into("<h", frames, (i + j) * 2, sample)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(bytes(frames))


@pytest.fixture
def click_track_120(tmp_path):
    path = tmp_path / "click_120.wav"
    _write_click_track(path, 120.0)
    return path


@pytest.mark.skipif(
    not _aubio_importable(), reason="aubio package not installed"
)
def test_aubio_end_to_end_synthetic(click_track_120):
    """Synthetic 120 BPM click track → aubio analyzer returns ~120 BPM."""
    from ax_bpm.analyzer import AubioAnalyzer

    analyzer = AubioAnalyzer()
    bpm = asyncio.get_event_loop().run_until_complete(
        analyzer.get_bpm(None, str(click_track_120))
    )
    assert bpm is not None
    assert abs(bpm - 120.0) < 5.0  # a few BPM of accuracy is sufficient