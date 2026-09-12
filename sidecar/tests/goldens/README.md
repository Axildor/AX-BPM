# Phase 0 — Golden Validation (AX-BPM sidecar gate)

This directory contains the Phase 0 golden-validation harness for the
AX-BPM sidecar mood analyzer. It gates ALL sidecar work: the ONNX
inference path must reproduce the TensorFlow reference path bit-closely
before any integration proceeds.

## Verdict

**PASS** (workflow run 34717614049, 2026-09-12, commit 4651aa5)

| Check | Result | Tolerance |
|---|---|---|
| Pooled embedding cosine (ONNX vs bs64 TF) | 0.99999994 – 1.00000012 | >= 0.999999 |
| Pooled embedding max abs-diff | 3.4e-07 – 6.9e-06 | <= 1e-4 |
| Moodtheme head prob max abs-diff | 3.2e-07 – 9.2e-07 | <= 1e-3 |
| Danceability head prob max abs-diff | 2.4e-07 – 1.0e-06 | <= 1e-3 |
| Mel patch 0 sha256 (composed vs reference) | exact match, all 4 clips | exact |
| Log-mel subsample abs-diff | 0.0 (bit-exact) | <= 1e-3 |

Per-clip details in `onnx_equivalence_results.json` and
`frontend_composition_results.json`.

## How it works

1. `generate_clips.py` — deterministic clip generator (SEED=20260912):
   `click_90bpm`, `click_160bpm`, `pink_noise`, `chirp_sweep`.
   44.1 kHz mono, 30 s each. Clips are NEVER committed; they are
   regenerated identically on every run.
2. `freeze_goldens.py` — the TF reference path (runs on glibc only):
   - log-mel via `TensorflowInputMusiCNN` (the ONLY front-end algorithm;
     `TensorflowPredictEffnetDiscogs` reuses it internally — there is no
     `TensorflowInputEffnetDiscogs`)
   - patching: `patchSize=128`, `patchHopSize=62` (overlapping patches,
     ~1.008 Hz prediction rate), `lastPatchMode="repeat"`,
     `batchSize=64`, `lastBatchMode="discard"`
   - `discogs-effnet-bs64-1.pb` -> per-patch embeddings `[n, 1280]`
     (selected by SHAPE: trailing dim 1280, not by TF output name)
   - **pooling**: mean over per-patch embeddings -> single 1280-d vector
   - heads on the POOLED vector: moodtheme `[56]`, danceability `[2]`
   - stores: pooled embedding, head probs, sha256 of raw mel patch 0
     float32 bytes, 16-frame strided subsample of patch 0
3. `check_onnx_equivalence.py` — the GATE: runs
   `discogs-effnet-bsdynamic-1.onnx` over the same clips with the same
   front end and patching, pools identically, and compares
   pooled-vs-pooled.
4. `composed_frontend.py` — front-end composition test: rebuilds the
   log-mel front end from plain essentia DSP (Windowing / Spectrum /
   MelBands / UnaryOperator) WITHOUT any `TensorflowInput*` component,
   replicates the patching, and compares patch 0 against the frozen
   reference (sha256 + tolerance).

## Pinned front-end constants (confirmed from Essentia source)

Source: `src/algorithms/spectral/tensorflowinputmusicnn.cpp` (master),
`src/algorithms/machinelearning/tensorflowpredicteffnetdiscogs.cpp`,
`src/algorithms/standard/vectorrealtotensor.h`.

| Constant | Value |
|---|---|
| sampleRate | 16000 |
| frameSize | 512 |
| hopSize | 256 |
| numberBands | 96 |
| lowFrequencyBound | 0 (default) |
| highFrequencyBound | 8000 (sampleRate/2) |
| warpingFormula | slaneyMel |
| weighting | linear |
| normalize | unit_tri |
| bands type | power (default) |
| Windowing | type hann (default), normalized=false, zeroPhase=true |
| log transform | `log10(10000 * mel + 1)` (UnaryOperator shift=1 scale=10000, then log10) |
| FrameCutter | startFromZero=false (zero-centered) |
| patchSize (effnet) | 128 |
| patchHopSize (effnet) | 62 (overlapping) |
| lastPatchMode (effnet) | repeat |
| batchSize (effnet) | 64 |
| lastBatchMode (effnet) | discard (hardcoded in streaming configure) |
| patchSize (musicnn) | 187 |
| patchHopSize (musicnn) | 187 (contiguous) |

## Model I/O (from model schema JSONs)

| Model | Input | Outputs |
|---|---|---|
| discogs-effnet-bs64-1.pb | `[64, 1, 128, 96]` | `[64, 400]` Sigmoid predictions; `[64, 1280]` Flatten embeddings |
| discogs-effnet-bsdynamic-1.onnx | `[n, 128, 96]` (dynamic batch) | `[n, 400]`; `[n, 1280]` |
| mtg_jamendo_moodtheme-discogs-effnet-1.pb | `[1280]` | `[56]` Sigmoid |
| danceability-discogs-effnet-1.pb | `[1280]` | `[2]` Softmax |

Tensor selection in all ONNX code is by SHAPE (trailing dim 1280 =
embeddings, 400 = predictions; head input = the 1280-d input), NOT by TF
output name — `PartitionedCall:1` may not survive ONNX export.

## Gate criteria

| Check | Tolerance |
|---|---|
| Pooled embedding cosine (ONNX vs bs64 TF) | >= 0.999999 |
| Pooled embedding abs-diff | <= 1e-4 |
| Head probability abs-diff | <= 1e-3 |
| Mel patch 0 sha256 (composed vs reference) | exact match |
| Log-mel subsample abs-diff | <= 1e-3 |

A FAIL on any criterion stops all sidecar work.

## Model URLs

| Model | URL |
|---|---|
| effnet bs64 (TF) | https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb |
| effnet bsdynamic (ONNX) | https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bsdynamic-1.onnx |
| moodtheme head | https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb |
| danceability head | https://essentia.upf.edu/models/classification-heads/danceability/danceability-discogs-effnet-1.pb |

Model weights are downloaded at run time to a temp dir and NEVER
committed (CC BY-NC-SA licensing).

## Running

The devcontainer is Alpine/musl and cannot run TensorFlow. Run the
one-shot glibc job instead:

```
gh workflow run phase0-golden-validation.yml -R Axildor/AX-BPM \
  -f commit_goldens=true -f debug=false
```

Debug mode (`-f debug=true`) runs the full sequence but commits nothing
and uploads logs for iteration.

## Note for v1.1 (musiCNN family)

The musiCNN-family models (e.g. DEAM / msd-musicnn mood/arousal/valence)
use the SAME front end (`TensorflowInputMusiCNN`) but DIFFERENT patching:
**200-dim embeddings at patchSize=187** (contiguous), vs effnet's
**1280-dim embeddings at patchSize=128, hop 62**. Do NOT reuse the effnet
constants for the musiCNN stage — only the front-end DSP constants are
shared.