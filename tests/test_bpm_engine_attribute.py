"""bpm_engine attribute tests (mood-degeneracy closeout).

`bpm_engine: aubio|numpy` appears on locally-analyzed results only:
- analyzer tempo present → "aubio" (source=analyzer)
- analyzer tempo failed → NumPy floor → "numpy" (source=numpy)
- Deezer-metadata BPM → attribute OMITTED (no engine)
- cache round-trip: make_result_from_cache preserves the attribute.

Semantics (owner tightening 4b): the attribute describes the BPM SOURCE;
cached mood attributes can outlive the current run — "mood from cache +
BPM from floor" is a legitimate mixed state, not a bug.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import _stub_homeassistant

_stub_homeassistant()

from unittest.mock import AsyncMock, MagicMock

from ax_bpm.pipeline import ResolutionResult, make_result_from_cache
from ax_bpm.store import BpmCache


@pytest.fixture
def mock_session():
    return MagicMock()


@pytest.fixture
def mock_cache():
    cache = MagicMock(spec=BpmCache)
    cache.get = MagicMock(return_value=None)
    cache.async_put = AsyncMock()
    return cache


def test_local_analyzer_result_has_bpm_engine_aubio():
    """An analyzer-sourced local result carries bpm_engine=aubio."""
    result = ResolutionResult(
        103.8,
        source="analyzer",
        bpm_engine="aubio",
        track="White Town - Your Woman",
    )
    assert result.attrs["bpm_engine"] == "aubio"


def test_local_numpy_result_has_bpm_engine_numpy():
    """A NumPy-floor local result carries bpm_engine=numpy."""
    result = ResolutionResult(
        129.2,
        source="numpy",
        bpm_engine="numpy",
        track="Smashing Pumpkins - 1979",
    )
    assert result.attrs["bpm_engine"] == "numpy"


def test_deezer_metadata_result_omits_bpm_engine():
    """Deezer-metadata BPM has no engine → attribute omitted."""
    result = ResolutionResult(
        176.7,
        source="deezer_metadata",
        track="Norah Jones - Don't Know Why",
    )
    assert "bpm_engine" not in result.attrs


def test_cache_round_trip_preserves_bpm_engine():
    """make_result_from_cache preserves bpm_engine (attrs pass through)."""
    cached = {
        "bpm": 103.8,
        "bpm_engine": "aubio",
        "track": "White Town - Your Woman",
        "source": "analyzer",
    }
    result = make_result_from_cache(cached)
    assert result.source == "cache"
    assert result.attrs["bpm_engine"] == "aubio"


@pytest.mark.asyncio
async def test_pipeline_local_path_sets_bpm_engine_aubio(
    mock_session, mock_cache
):
    """End-to-end: analyzer tempo present → bpm_engine=aubio, source=analyzer."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from ax_bpm.deezer import DeezerClient
    from ax_bpm.itunes import ItunesClient
    from ax_bpm.pipeline import BpmPipeline

    pipeline = BpmPipeline.__new__(BpmPipeline)
    pipeline._hass = MagicMock()
    pipeline._deezer = MagicMock(spec=DeezerClient)
    pipeline._itunes = MagicMock(spec=ItunesClient)
    pipeline._cache = mock_cache
    pipeline._overrides = None
    pipeline._analyzer = MagicMock()
    pipeline._analyzer.get_bpm = AsyncMock(return_value=None)
    pipeline._analyzer_client = MagicMock()
    pipeline._analyzer_client.async_analyze_file = AsyncMock(
        return_value={
            "bpm": 103.8,
            "bpm_confidence": 0.9,
            "mood_scores": {
                "aggressive": 0.1,
                "party": 0.6,
                "relaxed": 0.2,
                "electronic": 0.7,
                "acoustic": 0.1,
            },
            "mood_tags": [],
            "danceability": 0.8,
        }
    )
    pipeline._mood_ready = True
    pipeline._octave_mode = "genre_mood"
    pipeline._lock = asyncio.Lock()
    pipeline._current_token = None

    pipeline._deezer.find_match = AsyncMock(return_value={})
    pipeline._download_preview = AsyncMock(return_value=True)

    match = {
        "id": "t1",
        "isrc": "GBDUW0000059",
        "provider": "deezer",
        "preview_url": "https://example.com/p.mp3",
        "genres": [],
    }
    result = await pipeline._resolve_local(
        "White Town", "Your Woman", 240.0, "isrc:GBDUW0000059", 1e9, match
    )
    assert result is not None
    assert result.attrs["bpm_engine"] == "aubio"
    assert result.source == "analyzer"


@pytest.mark.asyncio
async def test_pipeline_local_path_sets_bpm_engine_numpy(
    mock_session, mock_cache
):
    """End-to-end: analyzer tempo failed → floor → bpm_engine=numpy."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from ax_bpm.deezer import DeezerClient
    from ax_bpm.itunes import ItunesClient
    from ax_bpm.pipeline import BpmPipeline

    pipeline = BpmPipeline.__new__(BpmPipeline)
    pipeline._hass = MagicMock()
    pipeline._deezer = MagicMock(spec=DeezerClient)
    pipeline._itunes = MagicMock(spec=ItunesClient)
    pipeline._cache = mock_cache
    pipeline._overrides = None
    pipeline._analyzer = MagicMock()
    pipeline._analyzer.get_bpm = AsyncMock(return_value=103.4)
    pipeline._analyzer_client = MagicMock()
    pipeline._analyzer_client.async_analyze_file = AsyncMock(return_value=None)
    pipeline._mood_ready = True
    pipeline._octave_mode = "genre_mood"
    pipeline._lock = asyncio.Lock()
    pipeline._current_token = None

    pipeline._deezer.find_match = AsyncMock(return_value={})
    pipeline._download_preview = AsyncMock(return_value=True)

    match = {
        "id": "t1",
        "isrc": "GBDUW0000059",
        "provider": "deezer",
        "preview_url": "https://example.com/p.mp3",
        "genres": [],
    }
    result = await pipeline._resolve_local(
        "White Town", "Your Woman", 240.0, "isrc:GBDUW0000059", 1e9, match
    )
    assert result is not None
    assert result.attrs["bpm_engine"] == "numpy"
    assert result.source == "numpy"
