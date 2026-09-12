#!/usr/bin/env python3
"""Shared Phase 0 constants and patching helpers.

Every constant here is confirmed from the Essentia source (master branch):

- src/algorithms/spectral/tensorflowinputmusicnn.cpp  (front end)
- src/algorithms/machinelearning/tensorflowpredicteffnetdiscogs.cpp (patching)
- src/algorithms/standard/vectorrealtotensor.h (patch semantics)

There is NO `TensorflowInputEffnetDiscogs` algorithm. Effnet predict
internally creates `TensorflowInputMusiCNN` for mel-spectrogram extraction;
the only difference is the patching (patchSize 128 vs 187 for musicnn).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

# --- Front end (TensorflowInputMusiCNN::configure, hardcoded) ---------------
SAMPLE_RATE = 16000
FRAME_SIZE = 512
HOP_SIZE = 256
NUMBER_BANDS = 96
LOW_FREQUENCY_BOUND = 0.0  # MelBands default
HIGH_FREQUENCY_BOUND = 8000.0  # sampleRate / 2
WARPING_FORMULA = "slaneyMel"
WEIGHTING = "linear"
NORMALIZE = "unit_tri"
BANDS_TYPE = "power"  # MelBands default
WINDOW_TYPE = "hann"  # Windowing default (only `normalized=false` is set)
WINDOW_NORMALIZED = False
WINDOW_ZERO_PHASE = True  # Windowing default
WINDOW_ZERO_PADDING = 0  # Windowing default
SHIFT = 1.0
SCALE = 10000.0
COMPRESSION = "log10"
# Final transform: log10(mel * 10000 + 1)

# --- FrameCutter (TensorflowPredictEffnetDiscogs / TensorflowPredictMusiCNN)
FRAME_CUTTER_START_FROM_ZERO = False  # FrameCutter default (zero-centered)

# --- Patching (TensorflowPredictEffnetDiscogs defaults) ---------------------
PATCH_SIZE_EFFNET = 128
PATCH_HOP_SIZE_EFFNET = 62  # overlapping patches -> 1.008 Hz prediction rate
LAST_PATCH_MODE_EFFNET = "repeat"
BATCH_SIZE_EFFNET = 64
LAST_BATCH_MODE_EFFNET = "discard"  # hardcoded in streaming configure()

# --- Patching (TensorflowPredictMusiCNN defaults) ---------------------------
PATCH_SIZE_MUSICNN = 187
PATCH_HOP_SIZE_MUSICNN = 187  # contiguous (VectorRealToTensor default 0 -> no overlap)
LAST_PATCH_MODE_MUSICNN = "repeat"

# --- Model I/O shapes (from model schema JSONs) -----------------------------
EMB_DIM = 1280  # effnet embeddings (PartitionedCall:1 / Flatten)
PROB_DIM = 400  # effnet predictions (PartitionedCall:0 / Sigmoid)
MOODTHEME_DIM = 56
DANCEABILITY_DIM = 2


@dataclass(frozen=True)
class PatchSpec:
    """How a (n_frames, 96) log-mel matrix becomes model input patches."""

    patch_size: int
    patch_hop_size: int
    last_patch_mode: str  # "repeat" | "discard"

    def n_patches(self, n_frames: int) -> int:
        """Number of patches VectorRealToTensor emits for n_frames frames."""
        if n_frames <= 0:
            return 0
        if n_frames <= self.patch_size:
            return 1
        return 1 + (n_frames - self.patch_size) // self.patch_hop_size

    def make_patches(self, logmel: np.ndarray) -> np.ndarray:
        """Replicate VectorRealToTensor patching: (n_patches, patch, 96).

        patchHopSize=0 means contiguous patches; >0 means overlapping.
        lastPatchMode="repeat" pads the tail by repeating the last frame;
        "discard" drops the trailing frames instead.
        """
        n_frames = logmel.shape[0]
        if n_frames == 0:
            return np.zeros((0, self.patch_size, NUMBER_BANDS), dtype=np.float32)

        if self.last_patch_mode == "repeat":
            n = self.n_patches(n_frames)
            needed = (n - 1) * self.patch_hop_size + self.patch_size
            if needed > n_frames:
                pad = np.repeat(logmel[-1:], needed - n_frames, axis=0)
                logmel = np.concatenate([logmel, pad], axis=0)
        else:  # discard
            n = self.n_patches(n_frames)
            needed = (n - 1) * self.patch_hop_size + self.patch_size
            logmel = logmel[:needed]

        patches = []
        for i in range(n):
            start = i * self.patch_hop_size
            patches.append(logmel[start : start + self.patch_size])
        return np.stack(patches).astype(np.float32)


EFFNET_PATCH_SPEC = PatchSpec(
    patch_size=PATCH_SIZE_EFFNET,
    patch_hop_size=PATCH_HOP_SIZE_EFFNET,
    last_patch_mode=LAST_PATCH_MODE_EFFNET,
)

MUSICNN_PATCH_SPEC = PatchSpec(
    patch_size=PATCH_SIZE_MUSICNN,
    patch_hop_size=PATCH_HOP_SIZE_MUSICNN,
    last_patch_mode=LAST_PATCH_MODE_MUSICNN,
)


def sha256_bytes(arr: np.ndarray) -> str:
    """SHA-256 of the raw float32 bytes of an array (C order)."""
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).hexdigest()


def select_by_shape(outputs: list[np.ndarray], trailing_dim: int) -> np.ndarray:
    """Pick the output whose trailing dim matches (embeddings=1280, probs=400).

    TF names like `PartitionedCall:1` may not survive ONNX export, so tensor
    selection must be by SHAPE, not by name.
    """
    candidates = [o for o in outputs if o.ndim >= 2 and o.shape[-1] == trailing_dim]
    if not candidates:
        shapes = [list(o.shape) for o in outputs]
        raise ValueError(f"no output with trailing dim {trailing_dim}; shapes={shapes}")
    return candidates[0]


def dump_diagnostics(label: str, **arrays: np.ndarray) -> None:
    """On failure: dump shapes + first 8 values of each array."""
    print(f"--- DIAGNOSTICS [{label}] ---")
    for name, arr in arrays.items():
        a = np.asarray(arr)
        flat = a.reshape(-1)
        head = np.array2string(flat[:8], precision=6, max_line_width=120)
        print(f"  {name}: shape={list(a.shape)} dtype={a.dtype} first8={head}")
    print("--- END DIAGNOSTICS ---")