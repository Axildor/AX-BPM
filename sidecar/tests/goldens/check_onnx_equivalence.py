#!/usr/bin/env python3
"""Phase 0 GATE: ONNX bsdynamic embeddings vs frozen bs64 TF goldens.

Runs discogs-effnet-bsdynamic-1.onnx (onnxruntime) on the same clips and
compares against the frozen bs64 TF embeddings in goldens.npz.

Gate criteria (from the plan):
  - cosine similarity between ONNX and bs64 TF embeddings must be ~1.0
  - embedding abs-diff tolerance: <= 1e-4
  - head probability tolerance:   <= 1e-3 (heads run on ONNX embeddings
    vs frozen TF-head probabilities)

Exit code 0 = PASS, 1 = FAIL. A FAIL stops all sidecar work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

BASE = "https://essentia.upf.edu/models"
ONNX_URL = f"{BASE}/feature-extractors/discogs-effnet/discogs-effnet-bsdynamic-1.onnx"

# Gate tolerances (plan: embeddings <= 1e-4, probs <= 1e-3)
EMB_ABS_TOL = 1e-4
PROB_ABS_TOL = 1e-3
COSINE_MIN = 0.999999  # "approximately 1.0"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", required=True)
    parser.add_argument("--goldens", required=True, help="dir with goldens.npz + goldens_meta.json")
    parser.add_argument("--models", required=True, help="model cache dir")
    parser.add_argument("--heads", action="store_true", help="also run TF heads on ONNX embeddings (needs TF)")
    args = parser.parse_args()

    import essentia.standard as es
    import onnxruntime as ort

    goldens_dir = Path(args.goldens)
    goldens = np.load(goldens_dir / "goldens.npz")
    meta = json.loads((goldens_dir / "goldens_meta.json").read_text())

    onnx_path = Path(args.models) / "discogs-effnet-bsdynamic-1.onnx"
    if not onnx_path.exists():
        print(f"downloading {ONNX_URL}")
        urllib.request.urlretrieve(ONNX_URL, onnx_path)  # noqa: S310
    digest = sha256_file(onnx_path)
    size = onnx_path.stat().st_size
    print(f"onnx model: {size} bytes sha256={digest[:16]}...")

    # cross-check against the URL/size recorded at freeze time
    frozen = meta["models"]["effnet_onnx"]
    if frozen["size"] != size or frozen["sha256"] != digest:
        print("WARNING: ONNX model differs from the one used at freeze time")

    so = ort.SessionOptions()
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    onnx_in_name = session.get_inputs()[0].name
    print(f"onnx input: {onnx_in_name} shape={session.get_inputs()[0].shape}")

    # front end for ONNX path: composed (no TensorflowInput*) - see
    # composed_frontend.py; here we reuse the reference component ONLY to
    # feed the same patches the goldens were frozen with, so this script
    # isolates the model-equivalence question from the front-end question.
    fe = es.TensorflowInputEffnetDiscogs()
    patch_size = meta["front_end_specs"]["effnet"]["constants_used"]["patchSize"]
    bands = meta["front_end_specs"]["effnet"]["constants_used"]["numberBands"]

    results = {}
    all_pass = True

    for wav in sorted(Path(args.clips).glob("*.wav")):
        name = wav.stem
        audio = es.MonoLoader(filename=str(wav), sampleRate=16000)()
        patch = fe(audio)
        n_frames = (patch.shape[0] // patch_size) * patch_size
        patch = patch[:n_frames].reshape(-1, patch_size, bands, 1).astype(np.float32)

        # ONNX inference over patches
        onnx_embs = []
        for i in range(patch.shape[0]):
            out = session.run(None, {onnx_in_name: patch[i : i + 1]})[0]
            onnx_embs.append(np.squeeze(out, axis=0) if out.ndim == 3 else out[0])
        onnx_emb = np.concatenate(onnx_embs, axis=0).astype(np.float32)

        ref_emb = goldens[f"{name}__emb"]  # (n_patches, 200)
        if onnx_emb.shape != ref_emb.shape:
            print(f"FAIL {name}: shape mismatch onnx {onnx_emb.shape} vs ref {ref_emb.shape}")
            all_pass = False
            continue

        # cosine similarity per patch, then aggregate
        norms_on = np.linalg.norm(onnx_emb, axis=1)
        norms_ref = np.linalg.norm(ref_emb, axis=1)
        cos = np.sum(onnx_emb * ref_emb, axis=1) / (norms_on * norms_ref + 1e-12)
        abs_diff = np.abs(onnx_emb - ref_emb)
        max_abs = float(abs_diff.max())
        mean_cos = float(cos.mean())
        min_cos = float(cos.min())

        ok = min_cos >= COSINE_MIN and max_abs <= EMB_ABS_TOL
        all_pass = all_pass and ok
        results[name] = {
            "mean_cosine": mean_cos,
            "min_cosine": min_cos,
            "max_abs_diff": max_abs,
            "pass": ok,
        }
        status = "PASS" if ok else "FAIL"
        print(f"{status} {name}: mean_cos={mean_cos:.9f} min_cos={min_cos:.9f} max_abs_diff={max_abs:.3e}")

    # Head probabilities: run TF heads on ONNX embeddings if requested
    if args.heads:
        import tensorflow as tf

        for head_key, prob_key in (
            ("moodtheme_head", "mood_probs"),
            ("danceability_head", "dance_probs"),
        ):
            head_fname = meta["models"][head_key]["filename"]
            graph = tf.Graph()
            with graph.as_default():
                gd = tf.compat.v1.GraphDef()
                with tf.io.gfile.GFile(str(Path(args.models) / head_fname), "rb") as f:
                    gd.ParseFromString(f.read())
                tf.import_graph_def(gd, name="")
            ops = [op.name for op in graph.get_operations()]
            in_name, out_name = ops[0] + ":0", ops[-1] + ":0"
            sess = tf.compat.v1.Session(graph=graph)

            for wav in sorted(Path(args.clips).glob("*.wav")):
                name = wav.stem
                audio = es.MonoLoader(filename=str(wav), sampleRate=16000)()
                patch = fe(audio)
                n_frames = (patch.shape[0] // patch_size) * patch_size
                patch = patch[:n_frames].reshape(-1, patch_size, bands, 1).astype(np.float32)
                onnx_embs = []
                for i in range(patch.shape[0]):
                    out = session.run(None, {onnx_in_name: patch[i : i + 1]})[0]
                    onnx_embs.append(np.squeeze(out, axis=0) if out.ndim == 3 else out[0])
                mean_onnx = np.mean(np.concatenate(onnx_embs, axis=0), axis=0, keepdims=True)
                probs = sess.run(out_name, {in_name: mean_onnx})[0].astype(np.float32)
                ref = goldens[f"{name}__{prob_key}"]
                max_abs = float(np.abs(probs - ref).max())
                ok = max_abs <= PROB_ABS_TOL
                all_pass = all_pass and ok
                status = "PASS" if ok else "FAIL"
                print(f"{status} {name} {prob_key}: max_abs_diff={max_abs:.3e}")

    verdict = "PASS" if all_pass else "FAIL"
    print(f"\nGATE VERDICT: {verdict}")
    (goldens_dir / "onnx_equivalence_results.json").write_text(
        json.dumps({"verdict": verdict, "tolerances": {"emb_abs": EMB_ABS_TOL, "prob_abs": PROB_ABS_TOL}, "results": results}, indent=2)
    )
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()