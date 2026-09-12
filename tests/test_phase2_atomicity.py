"""Phase 2 integration tests: mood_scores atomicity + formula parity.

Catch 1: mood_scores is consumed only when present AND complete over
EXPECTED_MOOD_SCORES — a missing key must NEVER be read as 0.0 (that
would mean "maximally non-X" and bias octave gating toward intensity
during partial failure). Incomplete → genre-only fallback.

Nit: intensity/calmness formula parity — back-compat attribute names
must mean back-compat values (fixed payload → literal expected values).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ax_bpm.const import EXPECTED_MOOD_SCORES
from ax_bpm.pipeline import BpmPipeline


def _payload(mood_scores=None, arousal=None):
    return {
        "mood_tags": [{"tag": "party", "score": 0.6}],
        "mood_scores": mood_scores,
        "valence": None,
        "arousal": arousal,
        "valence_std": None,
        "arousal_std": None,
        "danceability": 0.7,
        "analyzed_seconds": 30.0,
        "model_versions": {},
    }


def test_complete_scores_consumed():
    """Complete five-signal dict → intensity/calmness computed."""
    scores = {
        "aggressive": 0.6, "party": 0.4, "electronic": 0.2,
        "relaxed": 0.1, "acoustic": 0.3,
    }
    attrs = BpmPipeline._mood_attrs_from_payload(_payload(scores))
    # intensity = (0.6+0.4+0.2)/3 = 0.4; calmness = (0.1+0.3)/2 = 0.2
    assert attrs["intensity"] == 0.4
    assert attrs["calmness"] == 0.2
    assert attrs["mood_scores"] == scores


def test_incomplete_scores_fall_back_to_genre_only():
    """Missing key → mood_scores dropped entirely, no 0.0 substitution."""
    scores = {
        "aggressive": 0.6, "party": 0.4, "electronic": 0.2,
        # relaxed MISSING — must not be read as 0.0
        "acoustic": 0.3,
    }
    attrs = BpmPipeline._mood_attrs_from_payload(_payload(scores))
    assert "intensity" not in attrs
    assert "calmness" not in attrs
    assert "mood_scores" not in attrs
    # Attribute layer still present.
    assert attrs["mood_tags"] == [{"tag": "party", "score": 0.6}]
    assert attrs["danceability"] == 0.7


def test_missing_scores_object():
    """mood_scores absent entirely → genre-only fallback."""
    attrs = BpmPipeline._mood_attrs_from_payload(_payload(None))
    assert "intensity" not in attrs
    assert "calmness" not in attrs


def test_scores_not_dict_ignored():
    """Malformed mood_scores (non-dict) → ignored, no crash."""
    attrs = BpmPipeline._mood_attrs_from_payload(_payload(["bad"]))
    assert "intensity" not in attrs


def test_formula_parity_regression():
    """Back-compat intensity/calmness names mean back-compat values.

    Mirrors math.compute_signals' mood means: intensity =
    (aggressive+party+electronic)/3, calmness = (relaxed+acoustic)/2.
    """
    scores = {
        "aggressive": 0.9, "party": 0.9, "electronic": 0.9,
        "relaxed": 0.0, "acoustic": 0.0,
    }
    attrs = BpmPipeline._mood_attrs_from_payload(_payload(scores))
    assert attrs["intensity"] == 0.9
    assert attrs["calmness"] == 0.0

    scores_calm = {
        "aggressive": 0.0, "party": 0.0, "electronic": 0.0,
        "relaxed": 1.0, "acoustic": 1.0,
    }
    attrs = BpmPipeline._mood_attrs_from_payload(_payload(scores_calm))
    assert attrs["intensity"] == 0.0
    assert attrs["calmness"] == 1.0


def test_arousal_tiebreaker():
    """Arousal tie-breaker fires only when |intensity - calmness| < 0.05."""
    scores = {
        "aggressive": 0.5, "party": 0.5, "electronic": 0.5,
        "relaxed": 0.5, "acoustic": 0.5,
    }
    attrs = BpmPipeline._mood_attrs_from_payload(_payload(scores, arousal=0.8))
    # intensity = calmness = 0.5 → tie-breaker: intensity→0.8, calmness→0.2
    assert attrs["intensity"] == 0.8
    assert attrs["calmness"] == 0.2


def test_expected_set_contents():
    """The atomicity set is exactly the five gating signals."""
    assert EXPECTED_MOOD_SCORES == {
        "aggressive", "party", "relaxed", "electronic", "acoustic",
    }