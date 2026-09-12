"""GATE 2 — ONNX inference vs committed Phase 0 goldens.

Runs the REAL ONNX sessions (effnet bsdynamic + heads) over the golden
clips and compares against the frozen goldens:
- pooled embeddings: elementwise abs-diff ≤ 1e-4 (Phase 0 criterion)
- head probs: abs-diff ≤ 1e-3 (Phase 0 criterion)
- cosine similarity logged as DIAGNOSTIC only (never a gate)
- explicit assert: the pinned moodtheme output is in [0,1]

TIER GATING (require_onnx): local musl workspace = module-level skip;
CI (AXBPM_REQUIRE_ONNX=1) = hard fail if onnxruntime is missing. ZERO
skips allowed in this file on CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import (
    CLIP_SAMPLE_RATE,
    GOLDENS_DIR,
    generate_clips,
    require_onnx,
    resample_libsamplerate,
)

require_onnx()

from ax_bpm_sidecar import config as cfg
from ax_bpm_sidecar.inference import InferenceEngine
from ax_bpm_sidecar.models import ModelManager

EMB_TOL = 1e-4   # elementwise, Phase 0 criterion
PROB_TOL = 1e-3  # Phase 0 criterion


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """Real ModelManager + InferenceEngine pointed at the model cache dir."""
    data_dir = Path(
        __import__("os").environ.get("AXBPM_CI_MODEL_DIR", "/data")
    )
    models = ModelManager(data_dir)
    models.ensure_all()
    eng = InferenceEngine(models)
    eng.load_sessions()
    if "effnet" not in eng._sessions:
        pytest.skip("effnet session could not be loaded", allow_module_level=True)
    return eng


@pytest.fixture(scope="module")
def clips_16k(tmp_path_factory):
    import wave

    tmp = tmp_path_factory.mktemp("clips")
    generate_clips(tmp)
    out = {}
    for wav in sorted(tmp.glob("*.wav")):
        with wave.open(str(wav), "rb") as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        out[wav.stem] = resample_libsamplerate(
            pcm.astype(np.float32) / 32768.0, CLIP_SAMPLE_RATE, cfg.SAMPLE_RATE
        )
    return out


@pytest.mark.parametrize(
    "name", ["click_90bpm", "click_160bpm", "pink_noise", "chirp_sweep"]
)
def test_pooled_embedding_golden(engine, clips_16k, name):
    """Pooled embedding vs golden: elementwise ≤1e-4 (cosine = diagnostic)."""
    goldens = np.load(GOLDENS_DIR / "goldens.npz")
    audio = clips_16k[name]
    patches = __import__("ax_bpm_sidecar.frontend", fromlist=["front_end"]).front_end(audio)
    pooled = engine._embeddings(patches)

    ref = goldens[f"{name}__pooled_emb"]
    abs_diff = np.abs(pooled - ref)
    max_abs = float(abs_diff.max())
    cosine = float(
        np.dot(pooled, ref) / (np.linalg.norm(pooled) * np.linalg.norm(ref))
    )
    print(f"{name}: emb max_abs={max_abs:.3e} cosine={cosine:.9f}")
    assert max_abs <= EMB_TOL, (
        f"{name}: pooled embedding max_abs {max_abs:.3e} > {EMB_TOL:.0e}"
    )


@pytest.mark.parametrize(
    "name", ["click_90bpm", "click_160bpm", "pink_noise", "chirp_sweep"]
)
def test_head_probs_golden(engine, clips_16k, name):
    """Head probs vs golden: abs-diff ≤1e-3; pinned moodtheme output in [0,1]."""
    goldens = np.load(GOLDENS_DIR / "goldens.npz")
    audio = clips_16k[name]
    patches = __import__("ax_bpm_sidecar.frontend", fromlist=["front_end"]).front_end(audio)
    pooled = engine._embeddings(patches)

    theme = engine._head("moodtheme", pooled)
    assert theme is not None, "pinned moodtheme output rejected by [0,1] guard"
    assert theme.min() >= 0.0 and theme.max() <= 1.0
    ref_theme = goldens[f"{name}__mood_probs"]
    assert float(np.abs(theme - ref_theme).max()) <= PROB_TOL

    dance = engine._head("danceability", pooled)
    assert dance is not None
    ref_dance = goldens[f"{name}__dance_probs"]
    assert float(np.abs(dance - ref_dance).max()) <= PROB_TOL


def test_analyze_payload_shape(engine, clips_16k):
    """End-to-end payload: atomic mood_scores, tags, danceability."""
    audio = clips_16k["click_90bpm"]
    payload = engine.analyze(audio)
    assert payload is not None
    assert "mood_scores" in payload  # all heads healthy in CI
    assert set(payload["mood_scores"]) == set(cfg.MOOD_HEADS)
    assert all(0.0 <= v <= 1.0 for v in payload["mood_scores"].values())
    assert isinstance(payload["mood_tags"], list)
    assert payload["danceability"] is not None
    assert payload["valence"] is None  # v1.1 DEAM deferred
    assert payload["analyzed_seconds"] > 0