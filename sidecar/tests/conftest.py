"""Sidecar test bootstrap: goldens, clip regeneration, skip-guards."""

from __future__ import annotations

import hashlib
import os
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

# Make the package importable from the tests directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GOLDENS_DIR = Path(__file__).resolve().parents[1] / "tests" / "goldens"


def require_onnx() -> None:
    """Skip-guard for real-ONNX tests (kills the vacuous-pass failure mode).

    Local (musl workspace): onnxruntime unavailable → module-level skip.
    CI (AXBPM_REQUIRE_ONNX=1): missing onnxruntime is a HARD FAIL.
    """
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        if os.environ.get("AXBPM_REQUIRE_ONNX"):
            pytest.fail("AXBPM_REQUIRE_ONNX=1 but onnxruntime is not importable")
        pytest.skip(
            "onnxruntime unavailable on musl workspace", allow_module_level=True
        )


# ---------------------------------------------------------------------------
# Deterministic clip regeneration (mirrors goldens/generate_clips.py)
# ---------------------------------------------------------------------------
CLIP_SAMPLE_RATE = 44100
CLIP_DURATION = 30.0
CLIP_N_SAMPLES = int(CLIP_SAMPLE_RATE * CLIP_DURATION)
CLIP_SEED = 20260912


def _write_wav(path: Path, samples: np.ndarray) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(CLIP_SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def _click_track(bpm: float) -> np.ndarray:
    rng = np.random.default_rng(CLIP_SEED)
    out = np.zeros(CLIP_N_SAMPLES)
    period = int(CLIP_SAMPLE_RATE * 60.0 / bpm)
    click_len = int(0.05 * CLIP_SAMPLE_RATE)
    t = np.arange(click_len) / CLIP_SAMPLE_RATE
    body = np.sin(2 * np.pi * 1000.0 * t) * np.exp(-t / 0.008)
    for start in range(0, CLIP_N_SAMPLES - click_len, period):
        out[start : start + click_len] += body
    out += rng.normal(0.0, 1e-6, CLIP_N_SAMPLES)
    return out / np.max(np.abs(out))


def _filtered_noise() -> np.ndarray:
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(CLIP_SEED + 1)
    sos = butter(4, 2000.0, btype="low", fs=CLIP_SAMPLE_RATE, output="sos")
    out = sosfilt(sos, rng.normal(0.0, 1.0, CLIP_N_SAMPLES))
    return out / np.max(np.abs(out))


def _chirp_sweep() -> np.ndarray:
    f0, f1 = 40.0, 12000.0
    t = np.arange(CLIP_N_SAMPLES) / CLIP_SAMPLE_RATE
    beta = np.log(f1 / f0) / CLIP_DURATION
    phase = 2 * np.pi * f0 * (np.exp(beta * t) - 1.0) / beta
    out = 0.8 * np.sin(phase)
    fade = int(0.01 * CLIP_SAMPLE_RATE)
    out[:fade] *= np.linspace(0, 1, fade)
    out[-fade:] *= np.linspace(1, 0, fade)
    return out


def generate_clips(out_dir: Path) -> dict[str, np.ndarray]:
    """Regenerate the 4 deterministic golden clips (44.1 kHz mono float64)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = {
        "click_90bpm": _click_track(90.0),
        "click_160bpm": _click_track(160.0),
        "pink_noise": _filtered_noise(),
        "chirp_sweep": _chirp_sweep(),
    }
    for name, samples in clips.items():
        _write_wav(out_dir / f"{name}.wav", samples)
    return clips


def sha256_bytes(arr: np.ndarray) -> str:
    """SHA-256 of the raw float32 bytes of an array (C order)."""
    return hashlib.sha256(
        np.ascontiguousarray(arr, dtype=np.float32).tobytes()
    ).hexdigest()


def resample_libsamplerate(
    samples: np.ndarray, src_rate: int, dst_rate: int, quality: int = 1
) -> np.ndarray:
    """Resample via libsamplerate — replicating essentia's streaming Resample.

    Essentia's MonoLoader wraps the STREAMING Resample algorithm, which uses
    a stateful src_new/src_process converter fed in _preferredSize=4096
    chunks (resample.h: "_preferredSize = 4096; // arbitrary"), with
    end_of_input=1 flushed at the end. The streaming path differs from
    one-shot src_simple by a transport delay + end zero-padding
    (resample.cpp comment; libsamplerate FAQ Q006) — so the chunked
    src_process replication is required for bit-exactness.

    Quality mapping (resample.h): "0 for best quality, 4 for fast linear
    approximation" — MonoLoader's resampleQuality DEFAULT is 1
    (SRC_SINC_MEDIUM_QUALITY). This is NOT libsamplerate's enum order.
    """
    import ctypes
    import ctypes.util

    lib_name = ctypes.util.find_library("samplerate") or "libsamplerate.so.0"
    lib = ctypes.CDLL(lib_name)

    class SRCData(ctypes.Structure):
        _fields_ = [
            ("data_in", ctypes.POINTER(ctypes.c_float)),
            ("data_out", ctypes.POINTER(ctypes.c_float)),
            ("input_frames", ctypes.c_long),
            ("output_frames", ctypes.c_long),
            ("input_frames_used", ctypes.c_long),
            ("output_frames_gen", ctypes.c_long),
            ("end_of_input", ctypes.c_int),
            ("src_ratio", ctypes.c_double),
        ]

    # Explicit signatures — without restype, ctypes truncates the src_new
    # pointer to 32 bits and src_process segfaults on 64-bit platforms.
    lib.src_new.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.src_new.restype = ctypes.c_void_p
    lib.src_process.argtypes = [ctypes.c_void_p, ctypes.POINTER(SRCData)]
    lib.src_process.restype = ctypes.c_int
    lib.src_delete.argtypes = [ctypes.c_void_p]
    lib.src_delete.restype = None

    err = ctypes.c_int(0)
    state = lib.src_new(quality, 1, ctypes.byref(err))
    if not state or err.value:
        raise RuntimeError(f"src_new failed: {err.value}")

    data = np.ascontiguousarray(samples, dtype=np.float32)
    ratio = dst_rate / src_rate
    chunk = 4096  # Resample::_preferredSize
    out_parts: list[np.ndarray] = []

    def _process(chunk_data: np.ndarray, end_of_input: int) -> np.ndarray:
        out_len = int(len(chunk_data) * ratio) + 100
        out = np.zeros(out_len, dtype=np.float32)
        src = SRCData()
        src.data_in = chunk_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        src.input_frames = len(chunk_data)
        src.data_out = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        src.output_frames = out_len
        src.end_of_input = end_of_input
        src.src_ratio = ratio
        error = lib.src_process(state, ctypes.byref(src))
        if error:
            raise RuntimeError(f"libsamplerate error {error}")
        return out[: src.output_frames_gen].copy()

    for start in range(0, len(data), chunk):
        part = _process(data[start : start + chunk], 0)
        out_parts.append(part)
    # Flush: essentia feeds the tail with end_of_input=1.
    out_parts.append(_process(np.zeros(0, dtype=np.float32), 1))

    lib.src_delete(state)
    return np.concatenate(out_parts) if out_parts else np.zeros(0, dtype=np.float32)