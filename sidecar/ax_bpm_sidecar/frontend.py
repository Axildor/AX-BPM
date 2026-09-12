"""Pure-NumPy log-mel front end — bit-exact replica of TensorflowInputMusiCNN.

NO essentia in the sidecar. Every constant and algorithm here is a faithful
port of the Essentia C++ source (master branch), confirmed against the
Phase 0 goldens (per-clip patch sha256s in goldens_meta.json):

- Windowing (windowing.cpp): symmetric hann `0.5 - 0.5*cos(2*pi*i/(N-1))`,
  normalized=false, zeroPhase=true (rotate the frame so the window center
  is at the frame boundary), zeroPadding=0.
- Spectrum (spectrum.cpp): magnitude of the real FFT, size N/2+1.
- MelBands (melbands.cpp + triangularbands.cpp): 96 triangular bands,
  0–8000 Hz, slaneyMel warping (essentiamath.h mel2hzSlaney/hz2melSlaney),
  linear weighting (hz2hz), unit_tri normalization (theoretical triangular
  area, NOT the sum of weights), power bands (spectrum²).
- UnaryOperator chain: y = x*10000 + 1, then log10 → log10(10000*mel + 1).
- FrameCutter (framecutter.cpp): startFromZero=false (zero-centered
  frames), frameSize 512, hopSize 256.
- Patching (VectorRealToTensor semantics, phase0_common.PatchSpec):
  patch 128, hop 62, lastPatchMode repeat.

BIT-EXACTNESS SCOPE: bit-exact applies to the regenerated 16 kHz golden
clips only (where the resample step is a no-op or the exact libsamplerate
path). Production decode/resample (miniaudio/ffmpeg chain) is NOT
essentia's MonoLoader resampleQuality=4 — live-audio outputs are
tolerance-bounded, never claimed bit-exact.
"""

from __future__ import annotations

import numpy as np

from . import config as cfg

# ---------------------------------------------------------------------------
# Slaney mel warping (essentiamath.h, verbatim)
# ---------------------------------------------------------------------------
_LIN_SLOPE = 3.0 / 200.0
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ * _LIN_SLOPE
_LOG_STEP = np.log(6.4) / 27.0


def hz2mel_slaney(hz: np.ndarray) -> np.ndarray:
    """Hz → mel, Slaney's MATLAB Auditory Toolbox formula."""
    hz = np.asarray(hz, dtype=np.float64)
    out = np.where(
        hz < _MIN_LOG_HZ,
        hz * _LIN_SLOPE,
        _MIN_LOG_MEL + np.log(np.maximum(hz, _MIN_LOG_HZ) / _MIN_LOG_HZ) / _LOG_STEP,
    )
    return out


def mel2hz_slaney(mel: np.ndarray) -> np.ndarray:
    """Mel → Hz, Slaney's formula (inverse)."""
    mel = np.asarray(mel, dtype=np.float64)
    return np.where(
        mel < _MIN_LOG_MEL,
        mel / _LIN_SLOPE,
        _MIN_LOG_HZ * np.exp((mel - _MIN_LOG_MEL) * _LOG_STEP),
    )


# ---------------------------------------------------------------------------
# Filterbank construction (MelBands::calculateFilterFrequencies +
# TriangularBands::createFilters, verbatim)
# ---------------------------------------------------------------------------
def _filter_frequencies() -> np.ndarray:
    """(numBands + 2,) band edge frequencies, linearly spaced in mel."""
    n = cfg.NUMBER_BANDS
    low_mel = hz2mel_slaney(np.array([cfg.LOW_FREQUENCY_BOUND]))[0]
    high_mel = hz2mel_slaney(np.array([cfg.HIGH_FREQUENCY_BOUND]))[0]
    increment = (high_mel - low_mel) / (n + 1)
    mel_freqs = low_mel + increment * np.arange(n + 2, dtype=np.float64)
    return mel2hz_slaney(mel_freqs)


def _triangular_filterbank(spectrum_size: int) -> np.ndarray:
    """(numBands, spectrum_size) filter coefficients, unit_tri normalized.

    Port of TriangularBands::createFilters with weighting=linear (hz2hz),
    normalize=unit_tri (theoretical triangular area (fstep1+fstep2)/2),
    type=power applied at compute time.
    """
    freqs = _filter_frequencies()
    n_bands = len(freqs) - 2
    frequency_scale = (cfg.SAMPLE_RATE / 2.0) / (spectrum_size - 1)

    coeffs = np.zeros((n_bands, spectrum_size), dtype=np.float64)
    for i in range(n_bands):
        fstep1 = freqs[i + 1] - freqs[i]
        fstep2 = freqs[i + 2] - freqs[i + 1]

        jbegin = int(np.ceil(freqs[i] / frequency_scale))
        jend = int(np.floor(freqs[i + 2] / frequency_scale))
        if jend >= spectrum_size:
            raise ValueError(
                f"band {i} exceeds Nyquist: {freqs[i + 2]} Hz"
            )

        weight = 0.0
        for j in range(jbegin, jend + 1):
            binfreq = j * frequency_scale
            if binfreq < freqs[i + 1]:
                coeffs[i, j] = (binfreq - freqs[i]) / fstep1
            else:
                coeffs[i, j] = (freqs[i + 2] - binfreq) / fstep2
            weight += coeffs[i, j]

        if weight == 0.0:
            raise ValueError(
                "insufficient spectrum bins for the mel filterbank"
            )

        # unit_tri: theoretical triangular area instead of the actual sum.
        if cfg.NORMALIZE == "unit_tri":
            weight = (fstep1 + fstep2) / 2.0
        if cfg.NORMALIZE in ("unit_sum", "unit_tri"):
            coeffs[i, jbegin : jend + 1] /= weight

    return coeffs


_FILTER_CACHE: dict[int, np.ndarray] = {}


def _filterbank(spectrum_size: int) -> np.ndarray:
    if spectrum_size not in _FILTER_CACHE:
        _FILTER_CACHE[spectrum_size] = _triangular_filterbank(spectrum_size)
    return _FILTER_CACHE[spectrum_size]


# ---------------------------------------------------------------------------
# Windowing (Windowing::hann + zeroPhase rotation, verbatim)
# ---------------------------------------------------------------------------
def _hann_window(size: int) -> np.ndarray:
    """Symmetric hann, NOT normalized (Windowing defaults, normalized=false)."""
    i = np.arange(size, dtype=np.float64)
    return 0.5 - 0.5 * np.cos((2.0 * np.pi * i) / (size - 1.0))


def _zero_phase_rotate(frames: np.ndarray) -> np.ndarray:
    """Windowing::compute zeroPhase branch.

    First half of the output is the second half of the frame, second half
    is the first half of the frame (rotation by frameSize/2).
    """
    half = frames.shape[-1] // 2
    return np.concatenate([frames[..., half:], frames[..., :half]], axis=-1)


# ---------------------------------------------------------------------------
# FrameCutter (startFromZero=false → zero-centered frames)
# ---------------------------------------------------------------------------
def _make_frames(audio: np.ndarray) -> np.ndarray:
    """Zero-centered frames: (n_frames, frameSize).

    FrameCutter with startFromZero=false: first frame starts at
    -(frameSize+1)//2 (zero-padded), last frame's center must be within
    the buffer. Frames are zero-padded at the edges.
    """
    n = len(audio)
    frame_size = cfg.FRAME_SIZE
    hop = cfg.HOP_SIZE
    # C++ integer division truncates toward zero: -(513)/2 = -256.
    # Python floor division would give -257 — a 1-sample shift. Use
    # -( (frame_size+1)//2 ) to replicate the C++ semantics.
    start_index = -((frame_size + 1) // 2)

    # FrameCutter (startFromZero=false): frames are emitted while the frame
    # START is within the buffer (start_index + k*hop < n); the final frame
    # — whose center is at/past the end — IS emitted zero-padded, then
    # _lastFrame stops the cutter. So n_frames = ceil((n - start_index)/hop).
    n_frames = max(0, (n - start_index + hop - 1) // hop)
    if n_frames <= 0:
        return np.zeros((0, frame_size), dtype=np.float64)

    idx = np.arange(n_frames) * hop + start_index
    frames = np.zeros((n_frames, frame_size), dtype=np.float64)
    for k, s in enumerate(idx):
        lo_src = max(0, s)
        hi_src = min(n, s + frame_size)
        lo_dst = max(0, -s)
        if hi_src > lo_src:
            frames[k, lo_dst : lo_dst + (hi_src - lo_src)] = audio[lo_src:hi_src]
    return frames


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def logmel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """(n_frames, 96) log-mel frames — TensorflowInputMusiCNN equivalent.

    audio must already be mono float at cfg.SAMPLE_RATE (16 kHz).
    """
    audio = np.asarray(audio, dtype=np.float64)
    frames = _make_frames(audio)
    if frames.shape[0] == 0:
        return np.zeros((0, cfg.NUMBER_BANDS), dtype=np.float32)

    # Windowing (Windowing::compute, zeroPhase branch): the window is
    # indexed by the ORIGINAL frame position j — `signal[j] * _window[j]` —
    # and the OUTPUT is rotated. The rotation permutes the frame by
    # frameSize/2, which multiplies the DFT by a unit-modulus phase factor
    # e^(±jπk) — the MAGNITUDE spectrum is invariant to it. The goldens
    # confirm: feeding the rotated or unrotated windowed frame to the
    # magnitude FFT yields identical mel patches. We therefore skip the
    # rotation entirely (it would be dead work for a magnitude-only path).
    window = _hann_window(cfg.FRAME_SIZE)
    windowed = frames * window

    # Spectrum: magnitude of the real FFT → (n_frames, 257).
    spectrum = np.abs(np.fft.rfft(windowed, n=cfg.FRAME_SIZE, axis=-1))

    # MelBands: power bands (spectrum²) through the unit_tri filterbank.
    fb = _filterbank(spectrum.shape[-1])
    mel = (spectrum**2) @ fb.T

    # UnaryOperator chain: y = x*10000 + 1, then log10.
    logmel = np.log10(mel * cfg.SCALE + cfg.SHIFT)

    return logmel.astype(np.float32)


def make_patches(logmel: np.ndarray) -> np.ndarray:
    """VectorRealToTensor patching: (n_patches, 128, 96).

    patchHopSize=62 (overlapping), lastPatchMode="repeat" pads the tail by
    repeating the last frame. Port of phase0_common.PatchSpec.
    """
    n_frames = logmel.shape[0]
    if n_frames == 0:
        return np.zeros((0, cfg.PATCH_SIZE, cfg.NUMBER_BANDS), dtype=np.float32)

    def n_patches(n: int) -> int:
        if n <= cfg.PATCH_SIZE:
            return 1
        return 1 + (n - cfg.PATCH_SIZE) // cfg.PATCH_HOP_SIZE

    n = n_patches(n_frames)
    needed = (n - 1) * cfg.PATCH_HOP_SIZE + cfg.PATCH_SIZE
    if cfg.LAST_PATCH_MODE == "repeat" and needed > n_frames:
        pad = np.repeat(logmel[-1:], needed - n_frames, axis=0)
        logmel = np.concatenate([logmel, pad], axis=0)
    elif cfg.LAST_PATCH_MODE == "discard":
        logmel = logmel[:needed]

    patches = np.stack(
        [logmel[i * cfg.PATCH_HOP_SIZE : i * cfg.PATCH_HOP_SIZE + cfg.PATCH_SIZE] for i in range(n)]
    )
    return patches.astype(np.float32)


def front_end(audio: np.ndarray) -> np.ndarray:
    """Full front end: (n_patches, 128, 96) float32 model input."""
    return make_patches(logmel_spectrogram(audio))