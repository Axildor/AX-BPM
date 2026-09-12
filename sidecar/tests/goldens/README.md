# Phase 0 — Golden Validation (AX-BPM sidecar mood analyzer)

This directory gates ALL sidecar work. It contains the frozen reference
tensors ("goldens") for the log-mel front end and the discogs-effnet
embedding model, plus the scripts that produce and verify them.

## Why this exists

The AX-BPM integration is removing in-process Essentia and moving mood
analysis to a sidecar add-on using ONNX models. Before any sidecar code
is written, we must prove:

1. **Model equivalence** — `discogs-effnet-bsdynamic-1.onnx` produces
   embeddings equivalent to the TF reference `discogs-effnet-bs64-1.pb`
   (the classification heads may have been trained on bs64 embeddings).
2. **Front-end reproducibility** — the log-mel front end can be composed
   from plain essentia standard DSP / numpy (NO `TensorflowInput*`
   components, which require essentia-tensorflow) and matches the TF
   reference within tolerance.

## Files

| File | Role |
|---|---|
| `generate_clips.py` | Deterministic synthetic clip generator (committed; clips are NOT) |
| `freeze_goldens.py` | TF reference path: mel patches, bs64 embeddings, head probs → `goldens.npz` |
| `check_onnx_equivalence.py` | **THE GATE**: ONNX bsdynamic vs bs64 TF goldens |
| `composed_frontend.py` | Pure-numpy/essentia-DSP front end vs reference (CI harness) |
| `goldens.npz` | Frozen tensors (small; committed) |
| `goldens_meta.json` | Model URLs/sizes/sha256, front-end specs, versions |
| `onnx_equivalence_results.json` | Gate run output |
| `frontend_composition_results.json` | Front-end test run output |

## Execution environment

The devcontainer is **Alpine/musl** — TensorFlow and essentia-tensorflow
ship glibc (manylinux) wheels only and can never install there. The
golden workflow therefore runs as a one-shot **GitHub Actions
`workflow_dispatch` on ubuntu-latest** (glibc):

    .github/workflows/phase0-golden-validation.yml

Dispatch it from the Actions tab. It regenerates the clips, freezes the
goldens, runs the gate, and commits refreshed goldens back.

## Models (downloaded at run time — NEVER committed, CC BY-NC-SA)

| Model | URL | Size |
|---|---|---|
| discogs-effnet-bs64-1.pb | https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb | ~21 MB |
| discogs-effnet-bsdynamic-1.onnx | https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bsdynamic-1.onnx | ~18 MB |
| mtg_jamendo_moodtheme-discogs-effnet-1.pb | https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme-discogs-effnet-1.pb | ~2.7 MB |
| danceability-discogs-effnet-1.pb | https://essentia.upf.edu/models/classification-heads/danceability-discogs-effnet-1.pb | ~0.5 MB |

Exact sizes and SHA-256 digests are recorded in `goldens_meta.json` at
freeze time; the gate script warns if the ONNX model differs.

## Gate criteria

- ONNX vs bs64 TF embeddings: cosine ≈ 1.0 (min ≥ 0.999999), max abs-diff ≤ 1e-4
- Head probabilities (TF heads on ONNX embeddings vs frozen): abs-diff ≤ 1e-3
- Composed front end vs reference mel patches: abs-diff ≤ 1e-3 (documented
  achieved tolerance in `frontend_composition_results.json`)

## Verdict

**PENDING** — dispatch the workflow to fill in the verdict. A FAIL here
stops all sidecar work (the heads may have been trained on bs64
embeddings and the plan must be reassessed).