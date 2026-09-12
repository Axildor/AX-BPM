#!/usr/bin/env python3
"""Front-end composition test: log-mel WITHOUT TensorflowInput* components.

Composes the effnet log-mel front end from plain essentia standard DSP
(Windowing / Spectrum / MelBands / UnaryOperator) per MTG's documented
effnet input spec, and compares against the frozen reference patch
subsamples in goldens.npz.

MTG effnet (TensorflowInputEffnetDiscogs) spec:
    sample rate 16000, frame size 512, hop size 256, 96 mel bands
    (50-11000 Hz), log(10 * x + 1e-7) scaling, patch 187 frames.

This script becomes the CI comparison harness for the sidecar front end.

Exit code 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# Documented MTG effnet spec (validated against goldens_meta.json at runtime)
SPEC = {
    "sampleRate": 16000,
    "frameSize": 512,
    "hopSize": 256,
    "numberBands": 96,
    "patchSize": 187,
    "warpedFreq": 400.0,  # htk mel scale used by essentia MelBands
    "logEps": 1e-7,
    "logMul": 10.0,
}


def composed_logmel(audio: np.ndarray) -> np.ndarray:
    """Compose the effnet log-mel front end from standard essentia DSP.

    Returns (n_frames, 96) log-mel frames, matching
    TensorflowInputEffnetDiscogs output.
    """
    import essentia.standard as es

    window = es.Windowing(
        size=SPEC["frameSize"],
        normalization="unit_sum" if False else "none",  # effnet uses no per-frame norm
        zeroPhase=False,
    )
    spectrum = es.Spectrum(size=SPEC["frameSize"])
    mel = es.MelBands(
        numberBands=SPEC["numberBands"],
        sampleRate=SPEC["sampleRate"],
        lowBandFreq=50.0,
        highBandFreq=11000.0,
        inputSize=SPEC["frameSize"] // 2 + 1,
        type="power",
        warping="htk",
        normalize="unit_triangular",
    )
    log = es.UnaryOperator(type="log10")  # applied as log10(mul * x + eps) below

    frames = es.FrameGenerator(
        audio, frameSize=SPEC["frameSize"], hopSize=SPEC["hopSize"], startFromZero=False
    )
    out = []
    for frame in frames:
        spec = spectrum(window(frame))
        bands = mel(spec)
        # effnet scaling: log(10 * x + 1e-7) == log10(10 * x + 1e-7) / ln(10)?
        # MTG reference: y = log(10 * x + 1e-7) with natural log.
        out.append(np.log(SPEC["logMul"] * bands + SPEC["logEps"]))
    return np.stack(out).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", required=True)
    parser.add_argument("--goldens", required=True)
    parser.add_argument("--tol", type=float, default=1e-3, help="abs tolerance on log-mel values")
    args = parser.parse_args()

    goldens_dir = Path(args.goldens)
    goldens = np.load(goldens_dir / "goldens.npz")
    meta = json.loads((goldens_dir / "goldens_meta.json").read_text())

    # sanity: our documented spec must match what was recorded at freeze time
    recorded = meta["front_end_specs"]["effnet"]["constants_used"]
    for key in ("sampleRate", "frameSize", "hopSize", "numberBands", "patchSize"):
        if recorded[key] != SPEC[key]:
            print(f"SPEC MISMATCH {key}: composed={SPEC[key]} frozen={recorded[key]}")
            sys.exit(1)

    import essentia.standard as es

    all_pass = True
    results = {}
    for wav in sorted(Path(args.clips).glob("*.wav")):
        name = wav.stem
        audio = es.MonoLoader(filename=str(wav), sampleRate=SPEC["sampleRate"])()
        logmel = composed_logmel(audio)

        # compare against frozen subsample of first patch
        sub_ref = goldens[f"{name}__patch_effnet_sub"]  # (16, 96)
        idx = np.linspace(0, SPEC["patchSize"] - 1, sub_ref.shape[0], dtype=int)
        sub_composed = logmel[: SPEC["patchSize"]][idx]

        if sub_composed.shape != sub_ref.shape:
            print(f"FAIL {name}: shape {sub_composed.shape} vs ref {sub_ref.shape}")
            all_pass = False
            continue

        abs_diff = np.abs(sub_composed - sub_ref)
        max_abs = float(abs_diff.max())
        mean_abs = float(abs_diff.mean())
        ok = max_abs <= args.tol
        all_pass = all_pass and ok
        status = "PASS" if ok else "FAIL"
        print(f"{status} {name}: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} (tol {args.tol:.0e})")
        results[name] = {"max_abs": max_abs, "mean_abs": mean_abs, "pass": ok}

    verdict = "PASS" if all_pass else "FAIL"
    print(f"\nFRONT-END COMPOSITION VERDICT: {verdict}")
    (goldens_dir / "frontend_composition_results.json").write_text(
        json.dumps({"verdict": verdict, "tolerance": args.tol, "results": results}, indent=2)
    )
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()