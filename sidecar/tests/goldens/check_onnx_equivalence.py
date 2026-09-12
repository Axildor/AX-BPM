#!/usr/bin/env python3
"""Phase 0 GATE: ONNX bsdynamic embeddings vs frozen bs64 TF goldens.

Runs discogs-effnet-bsdynamic-1.onnx (onnxruntime) on the same clips and
compares against the frozen bs64 TF goldens in goldens.npz.

Pipeline (identical to the freeze path except the model):
  log-mel (TensorflowInputMusiCNN) -> patches (128 / hop 62 / repeat)
  -> ONNX bsdynamic -> per-patch embeddings [n, 1280] (selected by SHAPE,
  trailing dim 1280, NOT by TF output name) -> mean-pool -> single 1280-d
  vector -> heads (optional) -> probabilities.

Gate criteria (from the plan):
  - pooled-vs-pooled embedding cosine similarity must be ~1.0 (>= 0.999999)
  - pooled embedding abs-diff tolerance: <= 1e-4
  - head probability tolerance:          <= 1e-3 (heads run on ONNX pooled
    embeddings vs frozen TF-head probabilities)

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
from phase0_common import (
    DANCEABILITY_DIM,
    EFFNET_PATCH_SPEC,
    EMB_DIM,
    FRAME_SIZE,
    HOP_SIZE,
    MOODTHEME_DIM,
    NUMBER_BANDS,
    SAMPLE_RATE,
    dump_diagnostics,
    select_by_shape,
)

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


def onnx_pooled_embedding(session, in_name: str, patches: np.ndarray) -> tuple[np.ndarray, int]:
    """Run ONNX over patches; select embeddings by SHAPE; mean-pool.

    Returns (pooled_vector, n_patches).
    """
    embs = []
    for i in range(patches.shape[0]):
        outs = session.run(None, {in_name: patches[i : i + 1]})
        emb = select_by_shape(outs, EMB_DIM)  # trailing dim 1280
        embs.append(emb.reshape(-1, EMB_DIM))
    per_patch = np.concatenate(embs, axis=0).astype(np.float32)  # (n, 1280)
    return per_patch.mean(axis=0).astype(np.float32), per_patch.shape[0]


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", required=True)
    parser.add_argument("--goldens", required=True, help="dir with goldens.npz + goldens_meta.json")
    parser.add_argument("--models", required=True, help="model cache dir")
    parser.add_argument("--heads", action="store_true", help="also run TF heads on ONNX pooled embeddings (needs TF)")
    args = parser.parse_args()

    import essentia.standard as es
    import onnxruntime as ort

    goldens_dir = Path(args.goldens)
    goldens = np.load(goldens_dir / "goldens.npz")
    meta = json.loads((goldens_dir / "goldens_meta.json").read_text())

    onnx_path = Path(args.models) / "discogs-effnet-bsdynamic-1.onnx"
    if not onnx_path.exists():
        print(f"downloading {ONNX_URL}")
        urllib.request.urlretrieve(ONNX_URL, onnx_path)
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

    # SHAPE-based input selection: the input with trailing dim 96.
    in_meta = next(
        (i for i in session.get_inputs() if i.shape and i.shape[-1] == NUMBER_BANDS),
        None,
    )
    if in_meta is None:
        print(
            f"FATAL: no ONNX input with trailing dim {NUMBER_BANDS}; "
            f"inputs: {[(i.name, i.shape) for i in session.get_inputs()]}",
            file=sys.stderr,
        )
        sys.exit(1)
    onnx_in_name = in_meta.name
    print(f"onnx input: {onnx_in_name} shape={in_meta.shape}")

    # front end for ONNX path: the reference component (TensorflowInputMusiCNN),
    # so this script isolates the MODEL-equivalence question from the
    # front-end question (that is composed_frontend.py's job).
    # TensorflowInputMusiCNN accepts 512-sample FRAMES, not raw audio: the
    # predict algorithms wire FrameCutter(512/256, startFromZero=false)
    # -> TensorflowInputMusiCNN internally. We do the same via FrameGenerator.
    fe = es.TensorflowInputMusiCNN()

    def logmel_frames(audio) -> np.ndarray:
        gen = es.FrameGenerator(audio, frameSize=FRAME_SIZE, hopSize=HOP_SIZE, startFromZero=False)
        # standard-mode algorithms expect VECTOR_REAL (Python list), not
        # the numpy arrays FrameGenerator yields.
        return np.stack([fe(frame.tolist()) for frame in gen]).astype(np.float32)

    results = {}
    pooled_by_clip: dict[str, np.ndarray] = {}
    all_pass = True

    for wav in sorted(Path(args.clips).glob("*.wav")):
        name = wav.stem
        audio = es.MonoLoader(filename=str(wav), sampleRate=SAMPLE_RATE)()
        logmel = logmel_frames(audio)
        patches = EFFNET_PATCH_SPEC.make_patches(logmel)  # (n, 128, 96)
        if patches.shape[0] < 1:
            print(f"FAIL {name}: no patches from {logmel.shape[0]} frames")
            all_pass = False
            continue

        pooled_onnx, n_patches = onnx_pooled_embedding(session, onnx_in_name, patches)
        pooled_by_clip[name] = pooled_onnx

        ref_pooled = goldens[f"{name}__pooled_emb"]  # (1280,)
        if pooled_onnx.shape != ref_pooled.shape:
            dump_diagnostics(name, pooled_onnx=pooled_onnx, ref_pooled=ref_pooled, logmel=logmel)
            print(f"FAIL {name}: shape mismatch onnx {pooled_onnx.shape} vs ref {ref_pooled.shape}")
            all_pass = False
            continue

        # pooled-vs-pooled comparison
        cos = float(
            np.dot(pooled_onnx, ref_pooled)
            / (np.linalg.norm(pooled_onnx) * np.linalg.norm(ref_pooled) + 1e-12)
        )
        max_abs = float(np.abs(pooled_onnx - ref_pooled).max())

        ok = cos >= COSINE_MIN and max_abs <= EMB_ABS_TOL
        all_pass = all_pass and ok
        results[name] = {
            "n_patches": int(n_patches),
            "ref_n_patches": int(meta["clips"][name]["n_patches_effnet"]),
            "cosine": cos,
            "max_abs_diff": max_abs,
            "pass": ok,
        }
        status = "PASS" if ok else "FAIL"
        print(
            f"{status} {name}: cos={cos:.9f} max_abs_diff={max_abs:.3e} "
            f"patches={n_patches} (ref {meta['clips'][name]['n_patches_effnet']})"
        )
        if not ok:
            dump_diagnostics(name, pooled_onnx=pooled_onnx, ref_pooled=ref_pooled)

    # Head probabilities: run TF heads on ONNX pooled embeddings if requested
    if args.heads:
        import tensorflow as tf

        def load_head(path: Path) -> tuple:
            graph = tf.Graph()
            with graph.as_default():
                gd = tf.compat.v1.GraphDef()
                with tf.io.gfile.GFile(str(path), "rb") as f:
                    gd.ParseFromString(f.read())
                tf.import_graph_def(gd, name="")
            in_name = None
            out_name = None
            for op in graph.get_operations():
                if op.type == "Placeholder" and in_name is None:
                    in_name = op.name + ":0"
                if op.type in ("Sigmoid", "Softmax"):
                    out_name = op.name + ":0"
            if out_name is None:
                out_name = graph.get_operations()[-1].name + ":0"
            sess = tf.compat.v1.Session(graph=graph)
            # Head placeholders may be 1D [1280] or 2D [?, 1280].
            in_rank = graph.get_tensor_by_name(in_name).shape.rank
            return sess, in_name, out_name, in_rank

        heads = {}
        for head_key, out_dim in (
            ("moodtheme_head", MOODTHEME_DIM),
            ("danceability_head", DANCEABILITY_DIM),
        ):
            head_fname = meta["models"][head_key]["filename"]
            sess, in_name, out_name, in_rank = load_head(Path(args.models) / head_fname)
            heads[head_key] = (sess, in_name, out_name, in_rank)

        for head_key, prob_key in (
            ("moodtheme_head", "mood_probs"),
            ("danceability_head", "dance_probs"),
        ):
            sess, in_name, out_name, in_rank = heads[head_key]
            for name, pooled_onnx in pooled_by_clip.items():
                x = pooled_onnx[None, :] if in_rank == 2 else pooled_onnx
                probs = sess.run(out_name, {in_name: x})[0].astype(np.float32)
                ref = goldens[f"{name}__{prob_key}"]
                if probs.shape != ref.shape:
                    dump_diagnostics(f"{name}/{prob_key}", probs=probs, ref=ref)
                    print(f"FAIL {name} {prob_key}: shape {probs.shape} vs ref {ref.shape}")
                    all_pass = False
                    continue
                max_abs = float(np.abs(probs - ref).max())
                ok = max_abs <= PROB_ABS_TOL
                all_pass = all_pass and ok
                status = "PASS" if ok else "FAIL"
                print(f"{status} {name} {prob_key}: max_abs_diff={max_abs:.3e}")
                if not ok:
                    dump_diagnostics(f"{name}/{prob_key}", probs=probs, ref=ref)

    verdict = "PASS" if all_pass else "FAIL"
    print(f"\nGATE VERDICT: {verdict}")
    (goldens_dir / "onnx_equivalence_results.json").write_text(
        json.dumps({"verdict": verdict, "tolerances": {"emb_abs": EMB_ABS_TOL, "prob_abs": PROB_ABS_TOL}, "results": results}, indent=2)
    )
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()