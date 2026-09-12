#!/usr/bin/env python3
"""Front-end composition test: log-mel WITHOUT TensorflowInput* components.

Composes the effnet log-mel front end from plain essentia standard DSP
(Windowing / Spectrum / MelBands / UnaryOperator) with the EXACT constants
from TensorflowInputMusiCNN::configure() (the front end effnet predict
reuses internally), replicates the TensorflowPredictEffnetDiscogs patching
(patchSize=128, patchHopSize=62, lastPatchMode="repeat"), and compares
against the frozen reference patch subsamples in goldens.npz.

Pinned constant set (all confirmed from Essentia source, master branch):
    sample rate 16000, frameSize 512, hopSize 256, 96 mel bands,
    0-8000 Hz, slaneyMel / linear / unit_tri / power,
    Windowing type hann (default), normalized=false, zeroPhase=true,
    log10(10000 * mel + 1)   [UnaryOperator shift=1 scale=10000, then log10]

This script becomes the CI comparison harness for the sidecar front end.

Exit code 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from phase0_common import (
    BANDS_TYPE,
    COMPRESSION,
    EFFNET_PATCH_SPEC,
    FRAME_SIZE,
    HIGH_FREQUENCY_BOUND,
    HOP_SIZE,
    LOW_FREQUENCY_BOUND,
    NORMALIZE,
    NUMBER_BANDS,
    PATCH_SIZE_EFFNET,
    SAMPLE_RATE,
    SCALE,
    SHIFT,
    WARPING_FORMULA,
    WEIGHTING,
    WINDOW_NORMALIZED,
    WINDOW_TYPE,
    dump_diagnostics,
    sha256_bytes,
)

# The full pinned constant set (mirrors phase0_common; asserted against
# goldens_meta.json at runtime so freeze-time and compare-time agree).
SPEC = {
    "sampleRate": SAMPLE_RATE,
    "frameSize": FRAME_SIZE,
    "hopSize": HOP_SIZE,
    "numberBands": NUMBER_BANDS,
    "lowFrequencyBound": LOW_FREQUENCY_BOUND,
    "highFrequencyBound": HIGH_FREQUENCY_BOUND,
    "warpingFormula": WARPING_FORMULA,
    "weighting": WEIGHTING,
    "normalize": NORMALIZE,
    "bandsType": BANDS_TYPE,
    "windowType": WINDOW_TYPE,
    "windowNormalized": WINDOW_NORMALIZED,
    "shift": SHIFT,
    "scale": SCALE,
    "compression": COMPRESSION,
    "patchSize": PATCH_SIZE_EFFNET,
    "patchHopSize": EFFNET_PATCH_SPEC.patch_hop_size,
    "lastPatchMode": EFFNET_PATCH_SPEC.last_patch_mode,
}


def composed_logmel(audio: np.ndarray) -> np.ndarray:
    """Compose the effnet log-mel front end from standard essentia DSP.

    Returns (n_frames, 96) log-mel frames, matching TensorflowInputMusiCNN
    output (which is what effnet predict feeds its model).
    """
    import essentia.standard as es

    # Windowing: only `normalized=false` is set in the reference; type
    # defaults to "hann", zeroPhase to true, zeroPadding to 0.
    window = es.Windowing(normalized=False)
    spectrum = es.Spectrum(size=SPEC["frameSize"])
    mel = es.MelBands(
        inputSize=SPEC["frameSize"] // 2 + 1,
        numberBands=SPEC["numberBands"],
        sampleRate=SPEC["sampleRate"],
        lowFrequencyBound=SPEC["lowFrequencyBound"],
        highFrequencyBound=SPEC["highFrequencyBound"],
        warpingFormula=SPEC["warpingFormula"],
        weighting=SPEC["weighting"],
        normalize=SPEC["normalize"],
        type=SPEC["bandsType"],
    )
    # Two UnaryOperator steps, exactly as in TensorflowInputMusiCNN:
    #   shift step:  y = x * scale + shift   (type defaults to identity)
    #   compression: y = log10(x)
    shift_op = es.UnaryOperator(shift=SPEC["shift"], scale=SPEC["scale"])
    compression_op = es.UnaryOperator(type=SPEC["compression"])

    # FrameGenerator is the standard-mode iterator over FrameCutter
    # semantics: startFromZero=false (zero-centered first frame), matching
    # the reference FrameCutter configuration.
    frames = es.FrameGenerator(
        audio, frameSize=SPEC["frameSize"], hopSize=SPEC["hopSize"], startFromZero=False
    )

    out = []
    for frame in frames:
        # standard-mode algorithms expect VECTOR_REAL (Python list), not
        # the numpy arrays FrameGenerator yields.
        frame = frame.tolist()
        spec = spectrum(window(frame))
        bands = mel(spec)
        out.append(compression_op(shift_op(bands)))
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

    # sanity: our pinned spec must match what was recorded at freeze time
    recorded = meta["front_end_specs"]["effnet"]["constants_used"]
    mismatches = [k for k, v in SPEC.items() if recorded.get(k) != v]
    if mismatches:
        print(f"SPEC MISMATCH keys: {mismatches}")
        for k in mismatches:
            print(f"  {k}: composed={SPEC[k]} frozen={recorded.get(k)}")
        sys.exit(1)

    import essentia.standard as es

    all_pass = True
    results = {}
    for wav in sorted(Path(args.clips).glob("*.wav")):
        name = wav.stem
        audio = es.MonoLoader(filename=str(wav), sampleRate=SPEC["sampleRate"])()
        logmel = composed_logmel(audio)

        # replicate the reference patching, then compare patch 0
        patches = EFFNET_PATCH_SPEC.make_patches(logmel)
        if patches.shape[0] < 1:
            print(f"FAIL {name}: no patches from {logmel.shape[0]} frames")
            all_pass = False
            continue
        patch0 = patches[0]  # (128, 96)

        # bit-exactness guard: sha256 of raw mel patch 0 float32 bytes
        cksum = sha256_bytes(patch0)
        ref_cksum = meta["clips"][name]["patch_effnet_sha256"]
        cksum_ok = cksum == ref_cksum

        # tolerance comparison against frozen subsample of patch 0
        sub_ref = goldens[f"{name}__patch_effnet_sub"]  # (16, 96)
        idx = np.linspace(0, SPEC["patchSize"] - 1, sub_ref.shape[0], dtype=int)
        sub_composed = patch0[idx]

        if sub_composed.shape != sub_ref.shape:
            dump_diagnostics(name, composed=sub_composed, ref=sub_ref, logmel=logmel)
            print(f"FAIL {name}: shape {sub_composed.shape} vs ref {sub_ref.shape}")
            all_pass = False
            continue

        abs_diff = np.abs(sub_composed - sub_ref)
        max_abs = float(abs_diff.max())
        mean_abs = float(abs_diff.mean())
        ok = max_abs <= args.tol and cksum_ok
        all_pass = all_pass and ok
        status = "PASS" if ok else "FAIL"
        print(
            f"{status} {name}: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} "
            f"(tol {args.tol:.0e}) sha256_match={cksum_ok}"
        )
        if not ok:
            dump_diagnostics(name, composed_patch0=patch0, sub_composed=sub_composed, sub_ref=sub_ref)
        results[name] = {
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "sha256_match": cksum_ok,
            "n_patches": int(patches.shape[0]),
            "pass": ok,
        }

    verdict = "PASS" if all_pass else "FAIL"
    print(f"\nFRONT-END COMPOSITION VERDICT: {verdict}")
    (goldens_dir / "frontend_composition_results.json").write_text(
        json.dumps({"verdict": verdict, "tolerance": args.tol, "results": results}, indent=2)
    )
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()