"""Class-order map test: stubbed head outputs → positive-class probability.

The class order is INCONSISTENT across heads (mood_party/mood_relaxed are
classes[1]; the rest classes[0]) — this test pins the mapping with stubbed
[0.9, 0.1]-style outputs, no real ONNX needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ax_bpm_sidecar import config as cfg
from ax_bpm_sidecar.inference import InferenceEngine
from ax_bpm_sidecar.models import ModelManager
from conftest import require_onnx  # noqa: F401 — guard semantics shared


class _StubSession:
    """Returns a canned output list per model name."""

    def __init__(self, outputs):
        self._outputs = outputs

    def get_inputs(self):
        class _I:
            name = "model/Placeholder"

        return [_I()]

    def run(self, _names, _feed):
        return self._outputs


class _StubEngine(InferenceEngine):
    """Engine with stubbed sessions (no onnxruntime)."""

    def __init__(self, head_probs: dict[str, list[float] | None]):
        self._models = ModelManager()
        self._sessions = {"effnet": _StubSession([])}  # analyze() guard
        for name, probs in head_probs.items():
            if probs is None:
                continue  # e.g. effnet: _embeddings is overridden
            self._sessions[name] = _StubSession([np.array([probs], dtype=np.float32)])

    def _embeddings(self, patches):
        """Bypass the real effnet path (stub has no 1280-d output)."""
        return np.zeros(1280, dtype=np.float32)


def test_positive_class_map():
    """Each mood_scores key receives the positive-class probability."""
    # aggressive/electronic/acoustic: positive = classes[0] → 0.9
    # party/relaxed: positive = classes[1] → 0.9 (stub [0.1, 0.9])
    engine = _StubEngine(
        {
            "effnet": None,  # _embeddings is overridden; session unused
            "mood_aggressive": [0.9, 0.1],
            "mood_party": [0.1, 0.9],
            "mood_relaxed": [0.1, 0.9],
            "mood_electronic": [0.9, 0.1],
            "mood_acoustic": [0.9, 0.1],
            "danceability": [0.7, 0.3],
        }
    )
    audio = np.zeros(cfg.SAMPLE_RATE, dtype=np.float32)  # 1 s of silence
    payload = engine.analyze(audio)
    assert payload is not None
    scores = payload["mood_scores"]
    assert scores == {
        "aggressive": 0.9,
        "party": 0.9,
        "relaxed": 0.9,
        "electronic": 0.9,
        "acoustic": 0.9,
    }
    assert payload["danceability"] == 0.7


def test_mood_scores_atomic_when_head_missing():
    """One head down → mood_scores omitted ENTIRELY (never 0.0-substituted)."""
    engine = _StubEngine(
        {
            "effnet": None,
            "mood_aggressive": [0.9, 0.1],
            "mood_party": [0.1, 0.9],
            # mood_relaxed MISSING
            "mood_electronic": [0.9, 0.1],
            "mood_acoustic": [0.9, 0.1],
        }
    )
    audio = np.zeros(cfg.SAMPLE_RATE, dtype=np.float32)
    payload = engine.analyze(audio)
    assert payload is not None
    assert "mood_scores" not in payload


def test_moodtheme_pinned_output_index():
    """The pinned index selects the Sigmoid output among dual 56-d outputs."""
    sigmoid = np.array([[0.5] * 56], dtype=np.float32)
    logits = np.array([[5.0] * 56], dtype=np.float32)  # out of [0,1]
    engine = _StubEngine({})
    engine._sessions["moodtheme"] = _StubSession([sigmoid, logits])
    engine._sessions["effnet"] = _StubSession(
        [np.zeros((1, 400), dtype=np.float32), np.zeros((1, 1280), dtype=np.float32)]
    )
    out = engine._head("moodtheme", np.zeros(1280, dtype=np.float32))
    assert out is not None
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_moodtheme_logits_rejected():
    """If the pin pointed at logits, the [0,1] guard rejects it."""
    logits = np.array([[5.0] * 56], dtype=np.float32)
    engine = _StubEngine({})
    engine._sessions["moodtheme"] = _StubSession([logits, logits])
    out = engine._head("moodtheme", np.zeros(1280, dtype=np.float32))
    assert out is None