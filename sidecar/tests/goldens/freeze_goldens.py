#!/usr/bin/env python3
"""Freeze Phase 0 goldens using the TensorFlow reference path.

MUST run in a glibc environment (ubuntu-latest / python:3.12-slim) with:
    pip install essentia-tensorflow tensorflow-cpu numpy

For each clip (from generate_clips.py) this script:
  1. Computes the log-mel frames via TensorflowInputMusiCNN (the ONLY
     front-end algorithm; effnet predict reuses it internally).
  2. Patches with patchSize=128, patchHopSize=62, lastPatchMode="repeat"
     (TensorflowPredictEffnetDiscogs defaults).
  3. Runs discogs-effnet-bs64-1.pb -> per-patch embeddings [n, 1280]
     (selected by SHAPE: trailing dim 1280, not by TF output name).
  4. Pools per-patch embeddings via mean -> single 1280-d vector.
  5. Runs the classification heads on the POOLED vector:
       - mtg_jamendo_moodtheme-discogs-effnet-1  -> [56]
       - danceability-discogs-effnet-1           -> [2]
  6. Stores per-clip: pooled embedding (float32), patch count, head
     probabilities, sha256 of raw mel patch 0 float32 bytes, and a small
     strided subsample of mel patch 0 (for tolerance comparison in CI;
     checksums guard bit-exactness).

Outputs (written to --out):
    goldens.npz        - the tensors
    goldens_meta.json  - model URLs/sizes/sha256, front-end specs, versions

Model weights are downloaded to a temp dir and NEVER committed
(CC BY-NC-SA licensing).
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

from phase0_common import (
    BATCH_SIZE_EFFNET,
    BANDS_TYPE,
    COMPRESSION,
    DANCEABILITY_DIM,
    EMB_DIM,
    EFFNET_PATCH_SPEC,
    FRAME_SIZE,
    HIGH_FREQUENCY_BOUND,
    HOP_SIZE,
    LAST_BATCH_MODE_EFFNET,
    MOODTHEME_DIM,
    MUSICNN_PATCH_SPEC,
    NORMALIZE,
    NUMBER_BANDS,
    PATCH_SIZE_EFFNET,
    PROB_DIM,
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

BASE = "https://essentia.upf.edu/models"
MODELS = {
    "effnet_pb": (
        f"{BASE}/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",
        "discogs-effnet-bs64-1.pb",
    ),
    "moodtheme_head": (
        f"{BASE}/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb",
        "mtg_jamendo_moodtheme-discogs-effnet-1.pb",
    ),
    "danceability_head": (
        f"{BASE}/classification-heads/danceability/danceability-discogs-effnet-1.pb",
        "danceability-discogs-effnet-1.pb",
    ),
    "effnet_onnx": (
        f"{BASE}/feature-extractors/discogs-effnet/discogs-effnet-bsdynamic-1.onnx",
        "discogs-effnet-bsdynamic-1.onnx",
    ),
}

# Subsample of mel patches stored for tolerance comparison:
# 16 evenly spaced frames x all bands, float32. A few KB per clip.
PATCH_SUBSAMPLE_FRAMES = 16


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_models(model_dir: Path) -> dict:
    """Download all models; verify non-empty. Returns metadata dict."""
    model_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    for key, (url, fname) in MODELS.items():
        dest = model_dir / fname
        if not dest.exists():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, dest)  # noqa: S310
        size = dest.stat().st_size
        digest = sha256_file(dest)
        print(f"  {fname}: {size} bytes sha256={digest[:16]}...")
        meta[key] = {"url": url, "filename": fname, "size": size, "sha256": digest}
    return meta


def load_graph(path: Path, batch: int) -> tuple:
    """Load a frozen TF graph. Returns (graph, tensor_names_by_trailing_dim).

    Tensor selection is by SHAPE, not by name: we enumerate op outputs with
    a statically known trailing dim, EXCLUDING constants/variables (a
    [400,1280] dense-kernel transpose would otherwise shadow the real
    [64,1280] output) and requiring the leading dim to match the batch for
    rank-2 activations. The LAST candidate wins (graph outputs come last).
    """
    import tensorflow as tf

    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(str(path), "rb") as f:
            od_graph_def.ParseFromString(f.read())
        tf.import_graph_def(od_graph_def, name="")

    skip_ops = {"Const", "ConstV2", "Variable", "VariableV2", "StatefulVariableOp",
                "Placeholder", "PlaceholderWithDefault", "PlaceholderV2"}
    candidates: dict[int, list[str]] = {}
    for op in graph.get_operations():
        if op.type in skip_ops:
            continue
        for out in op.outputs:
            shape = out.shape
            if shape.rank is None or shape.rank < 1:
                continue
            dims = shape.as_list()
            if not dims or dims[-1] is None or dims[-1] <= 1:
                continue
            if shape.rank >= 2 and dims[0] is not None and dims[0] != batch:
                continue
            candidates.setdefault(dims[-1], []).append(out.name)
    by_trailing = {d: names[-1] for d, names in candidates.items()}
    print(f"  {path.name}: tensors by trailing dim: {sorted(by_trailing)}")
    return graph, by_trailing


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", required=True, help="dir with generated WAVs")
    parser.add_argument("--out", required=True, help="output dir for goldens")
    parser.add_argument("--models", required=True, help="dir to cache model files")
    args = parser.parse_args()

    import essentia
    import essentia.standard as es
    import tensorflow as tf

    clips_dir = Path(args.clips)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.models)

    model_meta = download_models(model_dir)

    # --- Load TF graphs (shape-indexed) -------------------------------------
    effnet_graph, effnet_tensors = load_graph(model_dir / "discogs-effnet-bs64-1.pb", BATCH_SIZE_EFFNET)
    mood_graph, mood_tensors = load_graph(model_dir / "mtg_jamendo_moodtheme-discogs-effnet-1.pb", 1)
    dance_graph, dance_tensors = load_graph(model_dir / "danceability-discogs-effnet-1.pb", 1)

    # --- Assumption guards (before heavy work) ------------------------------
    # 1. The front-end algorithm must exist.
    if not hasattr(es, "TensorflowInputMusiCNN"):
        print("FATAL: essentia.standard has no TensorflowInputMusiCNN", file=sys.stderr)
        sys.exit(1)
    # 2. The effnet graph must expose 1280-dim (embeddings) and 400-dim
    #    (predictions) outputs.
    if EMB_DIM not in effnet_tensors or PROB_DIM not in effnet_tensors:
        print(
            f"FATAL: effnet graph missing expected outputs (need trailing dims "
            f"{EMB_DIM} and {PROB_DIM}); found {sorted(effnet_tensors)}",
            file=sys.stderr,
        )
        sys.exit(1)
    # 3. Heads must expose their expected output dims.
    if MOODTHEME_DIM not in mood_tensors:
        print(f"FATAL: moodtheme head missing {MOODTHEME_DIM}-dim output; found {sorted(mood_tensors)}", file=sys.stderr)
        sys.exit(1)
    if DANCEABILITY_DIM not in dance_tensors:
        print(f"FATAL: danceability head missing {DANCEABILITY_DIM}-dim output; found {sorted(dance_tensors)}", file=sys.stderr)
        sys.exit(1)

    emb_t = effnet_tensors[EMB_DIM]
    mood_out_t = mood_tensors[MOODTHEME_DIM]
    dance_out_t = dance_tensors[DANCEABILITY_DIM]

    # Head inputs: the Placeholder ops (1280-dim embeddings go in).
    def placeholder_of(graph) -> str:
        for op in graph.get_operations():
            if op.type == "Placeholder":
                return op.name + ":0"
        raise RuntimeError("no Placeholder op found")

    mood_in_t = placeholder_of(mood_graph)
    dance_in_t = placeholder_of(dance_graph)

    sess = tf.compat.v1.Session(graph=effnet_graph)
    sess_mood = tf.compat.v1.Session(graph=mood_graph)
    sess_dance = tf.compat.v1.Session(graph=dance_graph)

    # 4. Probe the effnet graph once: fetch BOTH outputs and verify shapes.
    # The bs64 graph input is FIXED [64, 128, 96] (3D: batch, patch, bands;
    # the channel dim is squeezed by Essentia's predict wrapper).
    probe_in_shape = effnet_graph.get_tensor_by_name(
        placeholder_of(effnet_graph)
    ).shape.as_list()
    print(f"  effnet graph input shape: {probe_in_shape}")
    probe = np.zeros(probe_in_shape, dtype=np.float32)
    try:
        probe_outs = sess.run([emb_t, effnet_tensors[PROB_DIM]], {placeholder_of(effnet_graph): probe})
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: effnet probe inference failed: {exc}", file=sys.stderr)
        sys.exit(1)
    emb_probe, prob_probe = probe_outs
    if emb_probe.shape[-1] != EMB_DIM or prob_probe.shape[-1] != PROB_DIM:
        dump_diagnostics("effnet probe", emb=emb_probe, probs=prob_probe)
        sys.exit(1)
    print(f"  effnet probe OK: emb {emb_probe.shape}, probs {prob_probe.shape}")

    # --- Front end (reference path: TensorflowInputMusiCNN) -----------------
    # TensorflowInputMusiCNN accepts 512-sample FRAMES, not raw audio: the
    # predict algorithms wire FrameCutter(frameSize=512, hopSize=256,
    # startFromZero=false) -> TensorflowInputMusiCNN internally. We do the
    # same via FrameGenerator.
    fe = es.TensorflowInputMusiCNN()

    def logmel_frames(audio) -> np.ndarray:
        gen = es.FrameGenerator(audio, frameSize=FRAME_SIZE, hopSize=HOP_SIZE, startFromZero=False)
        return np.stack([fe(frame) for frame in gen]).astype(np.float32)

    # --- Per-clip processing -------------------------------------------------
    wavs = sorted(clips_dir.glob("*.wav"))
    if not wavs:
        print(f"ERROR: no .wav clips found in {clips_dir}", file=sys.stderr)
        sys.exit(1)

    store: dict[str, np.ndarray] = {}
    clip_meta: dict[str, dict] = {}

    for wav in wavs:
        name = wav.stem
        audio = es.MonoLoader(filename=str(wav), sampleRate=SAMPLE_RATE)()
        print(f"processing {name}: {audio.shape[0]} samples")

        # log-mel frames (reference front end)
        logmel = logmel_frames(audio)  # (n_frames, 96) float32
        print(f"  logmel frames: {logmel.shape}")

        # effnet patching: 128 frames, hop 62, repeat tail
        patches = EFFNET_PATCH_SPEC.make_patches(logmel)  # (n, 128, 96)
        n_patches = patches.shape[0]
        print(f"  patches: {patches.shape} (patchSize=128, patchHopSize=62, repeat)")

        # bs64 embeddings per patch, selected by SHAPE (trailing dim 1280).
        # The bs64 graph has a FIXED batch of 64: pad the last batch with
        # repeated patches and keep only the first n outputs (batch entries
        # are independent, so padding does not affect the kept outputs).
        batch = probe_in_shape[0]
        embs = []
        for start in range(0, n_patches, batch):
            chunk = patches[start : start + batch]
            n_keep = chunk.shape[0]
            if chunk.shape[0] < batch:
                pad = np.repeat(chunk[-1:], batch - chunk.shape[0], axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)
            outs = sess.run([emb_t], {placeholder_of(effnet_graph): chunk})
            embs.append(outs[0].reshape(batch, EMB_DIM)[:n_keep])
        embeddings = np.concatenate(embs, axis=0).astype(np.float32)  # (n, 1280)

        # POOL: mean over patches -> single 1280-d vector
        pooled = embeddings.mean(axis=0).astype(np.float32)  # (1280,)

        # head probabilities on the POOLED vector. Head placeholders may be
        # 1D [1280] or 2D [?, 1280]; feed the shape the graph expects.
        def head_feed(sess, graph, in_t, out_t, vec):
            in_shape = graph.get_tensor_by_name(in_t).shape.as_list()
            x = vec[None, :] if len(in_shape) == 2 else vec
            return sess.run(out_t, {in_t: x})[0].astype(np.float32)

        probs_mood = head_feed(sess_mood, mood_graph, mood_in_t, mood_out_t, pooled)
        probs_dance = head_feed(sess_dance, dance_graph, dance_in_t, dance_out_t, pooled)

        # checksums over the raw first patch (bit-exactness guard)
        cksum_e = sha256_bytes(patches[0])

        # musicnn spec: same front end, patch 187 contiguous (v1.1 note)
        patches_m = MUSICNN_PATCH_SPEC.make_patches(logmel)
        cksum_m = sha256_bytes(patches_m[0])

        # strided subsample of first patch for tolerance comparison
        idx = np.linspace(0, PATCH_SIZE_EFFNET - 1, PATCH_SUBSAMPLE_FRAMES, dtype=int)
        sub_e = patches[0][idx, :].astype(np.float32)  # (16, 96)

        store[f"{name}__pooled_emb"] = pooled
        store[f"{name}__mood_probs"] = probs_mood
        store[f"{name}__dance_probs"] = probs_dance
        store[f"{name}__patch_effnet_sub"] = sub_e

        clip_meta[name] = {
            "n_frames": int(logmel.shape[0]),
            "n_patches_effnet": int(n_patches),
            "n_patches_musicnn": int(patches_m.shape[0]),
            "patch_effnet_sha256": cksum_e,
            "patch_musicnn_sha256": cksum_m,
            "embedding_dim": EMB_DIM,
            "pooled_emb_norm": float(np.linalg.norm(pooled)),
        }
        print(
            f"  pooled emb {pooled.shape} (norm {clip_meta[name]['pooled_emb_norm']:.4f}), "
            f"mood probs {probs_mood.shape}, dance probs {probs_dance.shape}"
        )

    np.savez_compressed(out_dir / "goldens.npz", **store)

    meta = {
        "generated_with": {
            "essentia": essentia.__version__,
            "tensorflow": tf.__version__,
            "python": sys.version.split()[0],
        },
        "models": model_meta,
        "front_end_specs": {
            "effnet": {
                "component": "TensorflowInputMusiCNN (shared with effnet predict)",
                "constants_used": {
                    "sampleRate": SAMPLE_RATE,
                    "frameSize": FRAME_SIZE,
                    "hopSize": HOP_SIZE,
                    "numberBands": NUMBER_BANDS,
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
                    "patchSize": EFFNET_PATCH_SPEC.patch_size,
                    "patchHopSize": EFFNET_PATCH_SPEC.patch_hop_size,
                    "lastPatchMode": EFFNET_PATCH_SPEC.last_patch_mode,
                    "batchSize": BATCH_SIZE_EFFNET,
                    "lastBatchMode": LAST_BATCH_MODE_EFFNET,
                },
            },
            "musicnn": {
                "component": "TensorflowInputMusiCNN",
                "constants_used": {
                    "sampleRate": SAMPLE_RATE,
                    "frameSize": FRAME_SIZE,
                    "hopSize": HOP_SIZE,
                    "numberBands": NUMBER_BANDS,
                    "patchSize": MUSICNN_PATCH_SPEC.patch_size,
                    "patchHopSize": MUSICNN_PATCH_SPEC.patch_hop_size,
                    "lastPatchMode": MUSICNN_PATCH_SPEC.last_patch_mode,
                },
            },
        },
        "pooling": "mean over per-patch embeddings [n,1280] -> single 1280-d vector; heads run on the pooled vector",
        "clips": clip_meta,
        "patch_subsample": {
            "frames": PATCH_SUBSAMPLE_FRAMES,
            "note": "strided subsample of first patch; full-patch sha256 in clips[].patch_effnet_sha256",
        },
    }
    with open(out_dir / "goldens_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out_dir}/goldens.npz and goldens_meta.json")


if __name__ == "__main__":
    main()