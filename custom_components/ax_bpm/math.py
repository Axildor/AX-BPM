"""Octave disambiguation math — pure, side-effect-free.

The ONLY permitted correction is the whitelisted, mood/genre-gated math
below. Readings outside both windows are NEVER corrected. At most ONE
correction per track. Default is always "no correction" on ambiguity.

Signals:
- genre_fast / genre_slow: 1 if any Deezer album genre is in the
  FAST_GENRES / SLOW_GENRES whitelist, else 0.
- intensity  I = (S_aggressive + S_party + S_electronic + genre_fast) / 4
- calmness   C = (S_relaxed + S_acoustic + genre_slow) / 3

Mood scores S_x are continuous values in [0, 1] from the sidecar mood
analyzer's tag scores (Phase 1: previously Essentia SVM classifiers).

Graceful degradation: when mood scores are unavailable (None), the mood
terms are dropped and genre alone decides (thresholds unchanged). When
both mood and genre are unavailable, the raw estimate is published.
"""

from __future__ import annotations

from collections.abc import Mapping

from .const import (
    FAST_GENRES,
    HIGH_WINDOW,
    LOW_WINDOW,
    SLOW_GENRES,
    T_APPLY,
    T_CALM,
)

RULE_NONE = "none"
RULE_GENRE_DOUBLE = "genre_double"
RULE_MOOD_DOUBLE = "mood_double"
RULE_GENRE_HALF = "genre_half"
RULE_MOOD_HALF = "mood_half"


def _in_window(value: float, window: tuple[float, float]) -> bool:
    """Half-open window check: low [a, b), high (a, b]."""
    low, high = window
    if window is LOW_WINDOW:
        return low <= value < high
    return low < value <= high


def genre_flags(genres: list[str] | None) -> tuple[int, int]:
    """Return (genre_fast, genre_slow) from a list of genre names."""
    if not genres:
        return 0, 0
    normalized = {g.strip().lower() for g in genres if g}
    genre_fast = int(bool(normalized & FAST_GENRES))
    genre_slow = int(bool(normalized & SLOW_GENRES))
    return genre_fast, genre_slow


def compute_signals(
    mood_scores: Mapping[str, float] | None,
    genres: list[str] | None,
) -> tuple[float, float, int, int]:
    """Compute (intensity, calmness, genre_fast, genre_slow).

    Missing mood scores reduce to pure genre means; missing genres reduce
    to pure mood means; both missing gives 0.0 for both signals.
    """
    genre_fast, genre_slow = genre_flags(genres)

    if mood_scores:
        s_aggr = float(mood_scores.get("aggressive", 0.0))
        s_party = float(mood_scores.get("party", 0.0))
        s_elec = float(mood_scores.get("electronic", 0.0))
        s_relax = float(mood_scores.get("relaxed", 0.0))
        s_acou = float(mood_scores.get("acoustic", 0.0))
        intensity = (s_aggr + s_party + s_elec + genre_fast) / 4.0
        calmness = (s_relax + s_acou + genre_slow) / 3.0
    else:
        # Mood unavailable: genre-only path. genre_fast/genre_slow decide
        # directly, so map the flags onto the signal values.
        intensity = float(genre_fast)
        calmness = float(genre_slow)

    return intensity, calmness, genre_fast, genre_slow


def apply_octave_disambiguation(
    bpm_raw: float,
    mood_scores: Mapping[str, float] | None = None,
    genres: list[str] | None = None,
) -> dict:
    """Apply the octave disambiguation decision tree.

    Evaluate in this exact order; at most ONE correction per track;
    readings outside both windows are NEVER corrected.

    Returns a dict with:
      bpm:        final BPM (float, never 0 when bpm_raw > 0)
      rule:       one of none/genre_double/mood_double/genre_half/mood_half
      intensity:  I in [0, 1]
      calmness:   C in [0, 1]
      corrected:  bool
    """
    bpm_raw = float(bpm_raw)
    intensity, calmness, genre_fast, genre_slow = compute_signals(
        mood_scores, genres
    )

    result = {
        "bpm": bpm_raw,
        "rule": RULE_NONE,
        "intensity": intensity,
        "calmness": calmness,
        "corrected": False,
    }

    # 1. Low window: real tempo might be 2x raw.
    if _in_window(bpm_raw, LOW_WINDOW):
        if genre_fast == 1:
            result["bpm"] = 2.0 * bpm_raw
            result["rule"] = RULE_GENRE_DOUBLE
        elif intensity >= T_APPLY:
            result["bpm"] = 2.0 * bpm_raw
            result["rule"] = RULE_MOOD_DOUBLE
    # 2. High window: real tempo might be raw / 2.
    elif _in_window(bpm_raw, HIGH_WINDOW):
        if genre_slow == 1:
            result["bpm"] = bpm_raw / 2.0
            result["rule"] = RULE_GENRE_HALF
        elif calmness >= T_CALM:
            result["bpm"] = bpm_raw / 2.0
            result["rule"] = RULE_MOOD_HALF
    # 3. Outside both windows: never corrected.

    result["corrected"] = result["rule"] != RULE_NONE
    return result