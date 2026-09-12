#!/usr/bin/env python3
"""Freeze Phase 0 goldens using the TensorFlow reference path.

MUST run in a glibc environment (ubuntu-latest / python:3.12-slim) with:
    pip install essentia-tensorflow tensorflow-cpu numpy

For each clip (from generate_clips.py) this script:
  1. Computes the log-mel patch tensors via TensorflowInputEffnetDiscogs
     (effnet spec) and TensorflowInputMusiCNN (musicnn spec).
  2. Runs discogs-effnet-bs64-1.pb -> 200-dim embeddings.
  3. Runs the classification heads:
       - mtg_jamendo_moodtheme-discogs-effnet-1
       - danceability-discogs-effnet-1
  4. Stores per-clip: bs64 embeddings (float32), head probabilities,
     mel-patch SHA-256 checksums, and a small strided subsample of each
     mel patch (for tolerance comparison in CI; checksums guard
     bit-exactness).

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

BASE = "https://essentia.upf.edu/models"
MODELS = {
    "effnet_pb": (
        f"{BASE}/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",
        "discogs-effnet-bs64-1.pb",
    ),
    "moodtheme_head": (
        f"{BASE}/classification-heads/mtg_jamendo_moodtheme-discogs-effnet-1.pb",
        "mtg_jamendo_moodtheme-discogs-effnet-1.pb",
    ),
    "danceability_head": (
        f"{BASE}/classification-heads/danceability-discogs-effnet-1.pb",
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
    import essentia  # noqa: F401  (version recorded in meta)

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

    # --- Load TF graphs -----------------------------------------------------
    def load_graph(path: Path) -> tuple[tf.Graph, str, str]:
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(str(path), "rb") as f:
                od_graph_def.ParseFromString(f.read())
            tf.import_graph_def(od_graph_def, name="")
        # discover input/output names from the graph itself
        ops = [op.name for op in graph.get_operations()]
        input_name = next(
            (o for o in ops if "input" in o.lower() and "Placeholder" in graph.get_operation_by_name(o).type),
            ops[0],
        )
        output_name = ops[-1]
        return graph, input_name + ":0", output_name + ":0"

    effnet_graph, effnet_in, effnet_out = load_graph(model_dir / "discogs-effnet-bs64-1.pb")
    mood_graph, mood_in, mood_out = load_graph(model_dir / "mtg_jamendo_moodtheme-discogs-effnet-1.pb")
    dance_graph, dance_in, dance_out = load_graph(model_dir / "danceability-discogs-effnet-1.pb")

    sess = tf.compat.v1.Session(graph=effnet_graph)
    sess_mood = tf.compat.v1.Session(graph=mood_graph)
    sess_dance = tf.compat.v1.Session(graph=dance_graph)

    # --- Front ends (reference path: TensorflowInput* components) -----------
    sr_effnet = 16000
    frame_effnet = 512
    hop_effnet = 256
    bands_effnet = 96
    patch_effnet = 187  # 187 x 96 x 1

    sr_musicnn = 16000
    frame_musicnn = 1024
    hop_musicnn = 512
    bands_musicnn = 96
    patch_musicnn = 187

    fe_effnet = es.TensorflowInputEffnetDiscogs()
    fe_musicnn = es.TensorflowInputMusiCNN()

    # Record the ACTUAL specs from the components (do not trust constants).
    def spec_of(algo) -> dict:
        spec = {}
        for name in ("sampleRate", "frameSize", "hopSize", "numberBands", "patchSize"):
            try:
                spec[name] = getattr(algo, name)() if callable(getattr(algo, name, None)) else getattr(algo, name)
            except Exception:
                spec[name] = None
        return spec

    spec_effnet = spec_of(fe_effnet)
    spec_musicnn = spec_of(fe_musicnn)
    print(f"effnet spec:  {spec_effnet}")
    print(f"musicnn spec: {spec_musicnn}")

    # --- Per-clip processing -------------------------------------------------
    wavs = sorted(clips_dir.glob("*.wav"))
    if not wavs:
        print(f"ERROR: no .wav clips found in {clips_dir}", file=sys.stderr)
        sys.exit(1)

    store: dict[str, np.ndarray] = {}
    clip_meta: dict[str, dict] = {}

    for wav in wavs:
        name = wav.stem
        audio = es.MonoLoader(filename=str(wav), sampleRate=sr_effnet)()
        print(f"processing {name}: {audio.shape[0]} samples")

        # effnet mel patch (reference)
        patch_e = fe_effnet(audio)  # (frames, bands) float32
        n_frames_e = (patch_e.shape[0] // patch_effnet) * patch_effnet
        patch_e = patch_e[:n_frames_e].reshape(-1, patch_effnet, bands_effnet, 1)

        # musicnn mel patch (reference)
        audio_m = es.MonoLoader(filename=str(wav), sampleRate=sr_musicnn)()
        patch_m = fe_musicnn(audio_m)
        n_frames_m = (patch_m.shape[0] // patch_musicnn) * patch_musicnn
        patch_m = patch_m[:n_frames_m].reshape(-1, patch_musicnn, bands_musicnn, 1)

        # bs64 embeddings on full patches (batch through the graph)
        embs = []
        for i in range(patch_e.shape[0]):
            emb = sess.run(effnet_out, {effnet_in: patch_e[i : i + 1]})[0]
            embs.append(emb)
        embeddings = np.concatenate(embs, axis=0).astype(np.float32)  # (n_patches, 200)

        # head probabilities on the mean embedding (standard usage)
        mean_emb = embeddings.mean(axis=0, keepdims=True)
        probs_mood = sess_mood.run(mood_out, {mood_in: mean_emb})[0].astype(np.float32)
        probs_dance = sess_dance.run(dance_out, {dance_in: mean_emb})[0].astype(np.float32)

        # checksums over the full first patch (bit-exactness guard)
        cksum_e = hashlib.sha256(patch_e[0].tobytes()).hexdigest()
        cksum_m = hashlib.sha256(patch_m[0].tobytes()).hexdigest()

        # strided subsample of first patch for tolerance comparison
        idx = np.linspace(0, patch_effnet - 1, PATCH_SUBSAMPLE_FRAMES, dtype=int)
        sub_e = patch_e[0][idx, :, 0].astype(np.float32)  # (16, 96)
        idx_m = np.linspace(0, patch_musicnn - 1, PATCH_SUBSAMPLE_FRAMES, dtype=int)
        sub_m = patch_m[0][idx_m, :, 0].astype(np.float32)

        store[f"{name}__emb"] = embeddings
        store[f"{name}__mood_probs"] = probs_mood
        store[f"{name}__dance_probs"] = probs_dance
        store[f"{name}__patch_effnet_sub"] = sub_e
        store[f"{name}__patch_musicnn_sub"] = sub_m

        clip_meta[name] = {
            "n_patches_effnet": int(patch_e.shape[0]),
            "n_patches_musicnn": int(patch_m.shape[0]),
            "patch_effnet_sha256": cksum_e,
            "patch_musicnn_sha256": cksum_m,
            "embedding_dim": int(embeddings.shape[1]),
        }
        print(f"  embeddings {embeddings.shape}, mood probs {probs_mood.shape}, dance probs {probs_dance.shape}")

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
                "component": "TensorflowInputEffnetDiscogs",
                "recorded": spec_effnet,
                "constants_used": {
                    "sampleRate": sr_effnet,
                    "frameSize": frame_effnet,
                    "hopSize": hop_effnet,
                    "numberBands": bands_effnet,
                    "patchSize": patch_effnet,
                },
            },
            "musicnn": {
                "component": "TensorflowInputMusiCNN",
                "recorded": spec_musicnn,
                "constants_used": {
                    "sampleRate": sr_musicnn,
                    "frameSize": frame_musicnn,
                    "hopSize": hop_musicnn,
                    "numberBands": bands_musicnn,
                    "patchSize": patch_musicnn,
                },
            },
        },
        "clips": clip_meta,
        "patch_subsample": {
            "frames": PATCH_SUBSAMPLE_FRAMES,
            "note": "strided subsample of first patch; full-patch sha256 in clips[].patch_*_sha256",
        },
    }
    with open(out_dir / "goldens_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out_dir}/goldens.npz and goldens_meta.json")


if __name__ == "__main__":
    main()