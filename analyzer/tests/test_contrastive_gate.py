"""GATE 3 — Tier 3 contrastive gate: REAL previews through the REAL pipeline.

The mood-degeneracy post-fix verification deferred three pre-registered
checks (findings §3) to a glibc container with onnxruntime + aubio:

- P4 BPM unchanged (aubio is amplitude-scale invariant): white_town
  ≈ 103.8, smashing_pumpkins ≈ 126–127, norah_jones ≈ 176.7.
- P5 pairwise cosine < 0.9 for every pair across the three tracks
  (pre-fix reference: cos ≈ 1.0 — the degenerate embedding).
- P6 head directions: white_town electronic/party HIGH with aggressive
  DROPPED from the 0.93–0.95 saturated constant; norah_jones acoustic
  the argmax head (risen from 0.0062–0.0095) with relaxed HIGH;
  pumpkins aggressive ABOVE norah_jones and electronic BELOW white_town.

The three previews are fetched AT RUNTIME from the anonymous Deezer
public API by track ID (public identifiers — committed here by owner
decision). The AUDIO is never committed: CI caches it via
actions/cache keyed on the track-ID set; locally it lands in
AXBPM_CONTRASTIVE_CACHE_DIR (default: system temp).

ANTI-VACUUM: a preview-fetch failure is a HARD FAIL, never a skip — a
skip would let the permanent gate die silently the day the Deezer URL
scheme changes while the workflow stays green on nothing.

TIER GATING: local musl workspace (no onnxruntime) = module-level skip;
CI (AXBPM_REQUIRE_ONNX=1) = hard fail if missing. ZERO skips allowed in
this file on CI (the workflow's zero-skip guard covers this file).

The measured numbers (BPMs, pairwise cosines, five head scores per
track) are written to AXBPM_CONTRASTIVE_OUT (default /tmp) as
contrastive_gate_numbers.json for the findings report.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import require_onnx

require_onnx()

from ax_bpm_analyzer import tempo
from ax_bpm_analyzer.decode import decode_traced, downsample
from ax_bpm_analyzer.inference import InferenceEngine
from ax_bpm_analyzer.models import ModelManager

# CI hard-fails when aubio is missing (zero-skip guard, same as tempo tests).
if os.environ.get("AXBPM_REQUIRE_TEMPO") and not tempo.aubio_available():
    pytest.fail("AXBPM_REQUIRE_TEMPO=1 but aubio is not importable")

# The three §2 contrastive tracks — Deezer track IDs are public
# identifiers; the preview AUDIO is runtime-fetched and never committed.
CONTRASTIVE_TRACKS: dict[str, dict[str, object]] = {
    "white_town": {
        "track_id": 3802592782,  # White Town — Your Woman
        "bpm_ref": 103.8,
    },
    "smashing_pumpkins": {
        "track_id": 68976286,  # The Smashing Pumpkins — 1979
        "bpm_ref": 126.5,  # pre-registered band 126–127
    },
    "norah_jones": {
        "track_id": 3155839,  # Norah Jones — Don't Know Why
        "bpm_ref": 176.7,
    },
}

BPM_TOL = 1.0     # owner-pinned band (covers Deezer re-encode drift only)
COSINE_MAX = 0.9  # P5 gate (pre-registered §3)

DEEZER_API = "https://api.deezer.com"
FETCH_RETRIES = 2


def _cache_dir() -> Path:
    return Path(os.environ.get("AXBPM_CONTRASTIVE_CACHE_DIR") or tempfile.gettempdir()) / "axbpm-contrastive"


def _fetch_url(url: str, timeout: float = 30.0) -> bytes:
    last_err: Exception | None = None
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except Exception as err:  # noqa: BLE001 — retry, then hard-fail
            last_err = err
            if attempt < FETCH_RETRIES:
                time.sleep(2.0 * attempt)
    raise RuntimeError(f"fetch failed after {FETCH_RETRIES} attempts: {url}: {last_err}")


def _fetch_preview(name: str, spec: dict[str, object]) -> bytes:
    """Fetch one preview by track ID, with a local disk cache.

    ANY failure here is a HARD FAIL (anti-vacuum): the gate must never
    silently degrade to a skip.
    """
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    mp3 = cache / f"{name}.mp3"
    if mp3.is_file() and mp3.stat().st_size > 0:
        return mp3.read_bytes()

    track_id = spec["track_id"]
    meta = json.loads(_fetch_url(f"{DEEZER_API}/track/{track_id}"))
    preview_url = meta.get("preview")
    if not preview_url:
        raise RuntimeError(f"Deezer track {track_id} ({name}) has no preview URL")
    data = _fetch_url(preview_url)
    if len(data) < 100_000:  # ~30 s @ 128 kbps ≈ 480 KB; tiny = truncated
        raise RuntimeError(
            f"preview for {name} suspiciously small ({len(data)} bytes)"
        )
    mp3.write_bytes(data)
    return data


@pytest.fixture(scope="module")
def engine():
    """Real ModelManager + InferenceEngine (same recipe as the goldens)."""
    data_dir = Path(os.environ.get("AXBPM_CI_MODEL_DIR", "/data"))
    models = ModelManager(data_dir)
    models.ensure_all()
    eng = InferenceEngine(models)
    eng.load_sessions()
    if "effnet" not in eng._sessions:
        pytest.fail("effnet session could not be loaded (contrastive gate)")
    return eng


@pytest.fixture(scope="module")
def results(engine):
    """Run all three previews through the REAL pipeline; keep everything."""
    out: dict[str, dict[str, object]] = {}
    for name, spec in CONTRASTIVE_TRACKS.items():
        data = _fetch_preview(name, spec)
        traced = decode_traced(data, sample_rate=44100)
        if traced is None:
            pytest.fail(f"{name}: decode failed in the contrastive gate")
        samples_44k, sr_44k, decoder = traced
        samples_16k = downsample(samples_44k, sr_44k)
        payload = engine.analyze(samples_16k, audio_44k=samples_44k, sr_44k=sr_44k)
        if payload is None:
            pytest.fail(f"{name}: engine.analyze returned None (effnet down?)")
        patches = __import__(
            "ax_bpm_analyzer.frontend", fromlist=["front_end"]
        ).front_end(samples_16k)
        pooled = engine._embeddings(patches)
        out[name] = {
            "decoder": decoder,
            "bpm": payload.get("bpm"),
            "bpm_confidence": payload.get("bpm_confidence"),
            "mood_scores": payload.get("mood_scores"),
            "pooled": pooled,
            "patch_count": int(patches.shape[0]),
        }
    _emit_numbers(out)
    return out


def _emit_numbers(results: dict[str, dict[str, object]]) -> None:
    """Write the measured numbers for the findings report (evidence, not repo)."""
    names = list(results)
    cosines = {
        f"{a}__{b}": round(
            float(
                np.dot(results[a]["pooled"], results[b]["pooled"])
                / (
                    np.linalg.norm(results[a]["pooled"])  # type: ignore[arg-type]
                    * np.linalg.norm(results[b]["pooled"])  # type: ignore[arg-type]
                )
            ),
            6,
        )
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }
    summary = {
        name: {
            "decoder": r["decoder"],
            "bpm": r["bpm"],
            "bpm_confidence": r["bpm_confidence"],
            "patch_count": r["patch_count"],
            "mood_scores": r["mood_scores"],
        }
        for name, r in results.items()
    }
    summary["pairwise_cosine"] = cosines
    out_dir = Path(os.environ.get("AXBPM_CONTRASTIVE_OUT") or tempfile.gettempdir())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "contrastive_gate_numbers.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"\ncontrastive gate numbers → {out_dir / 'contrastive_gate_numbers.json'}")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# P4 — BPM unchanged (aubio is amplitude-scale invariant)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(CONTRASTIVE_TRACKS))
def test_p4_bpm_unchanged(results, name):
    spec = CONTRASTIVE_TRACKS[name]
    bpm = results[name]["bpm"]
    assert bpm is not None, f"{name}: no bpm in payload (tempo tier down?)"
    ref = float(spec["bpm_ref"])  # type: ignore[arg-type]
    assert abs(float(bpm) - ref) <= BPM_TOL, (
        f"{name}: bpm {bpm} outside ±{BPM_TOL} of pre-registered {ref}"
    )


# ---------------------------------------------------------------------------
# P5 — pairwise cosine < 0.9 (pre-fix reference: cos ≈ 1.0)
# ---------------------------------------------------------------------------


def test_p5_pairwise_cosine_below_gate(results):
    names = list(results)
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    assert len(pairs) == 3
    for a, b in pairs:
        va = results[a]["pooled"]  # type: ignore[index]
        vb = results[b]["pooled"]  # type: ignore[index]
        cos = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
        print(f"cosine({a}, {b}) = {cos:.6f}")
        assert cos < COSINE_MAX, (
            f"P5 FAIL: cosine({a}, {b}) = {cos:.6f} >= {COSINE_MAX} "
            "(degenerate-embedding signature)"
        )


# ---------------------------------------------------------------------------
# P6 — head directions (relative, per pre-registered §3.4)
# ---------------------------------------------------------------------------


def test_p6_white_town_directions(results):
    scores = results["white_town"]["mood_scores"]
    assert scores, "white_town: mood_scores missing (atomicity violation?)"
    assert float(scores["electronic"]) > 0.5, f"electronic not HIGH: {scores}"
    assert float(scores["party"]) > 0.5, f"party not HIGH: {scores}"
    assert float(scores["aggressive"]) < 0.9, (
        f"aggressive still at the saturated constant: {scores}"
    )


def test_p6_norah_jones_directions(results):
    scores = results["norah_jones"]["mood_scores"]
    assert scores, "norah_jones: mood_scores missing (atomicity violation?)"
    argmax = max(scores, key=lambda k: float(scores[k]))
    assert argmax == "acoustic", (
        f"acoustic should dominate (risen from 0.0062–0.0095): {scores}"
    )
    assert float(scores["relaxed"]) > 0.5, f"relaxed not HIGH: {scores}"


def test_p6_pumpkins_relative_directions(results):
    pumpkins = results["smashing_pumpkins"]["mood_scores"]
    norah = results["norah_jones"]["mood_scores"]
    white = results["white_town"]["mood_scores"]
    assert pumpkins and norah and white
    assert float(pumpkins["aggressive"]) > float(norah["aggressive"]), (
        f"pumpkins aggressive {pumpkins['aggressive']} should exceed "
        f"norah_jones {norah['aggressive']}"
    )
    assert float(pumpkins["electronic"]) < float(white["electronic"]), (
        f"pumpkins electronic {pumpkins['electronic']} should be below "
        f"white_town {white['electronic']}"
    )


# ---------------------------------------------------------------------------
# Structural sanity (patch count stays 29 — pre-registered §3.6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(CONTRASTIVE_TRACKS))
def test_patch_count_unchanged(results, name):
    assert results[name]["patch_count"] == 29
