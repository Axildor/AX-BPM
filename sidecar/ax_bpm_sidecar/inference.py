"""ONNX inference: warm sessions, mean pooling, pinned tensor selection.

All sessions load once at startup and stay warm (~25 MB weights total —
no load-on-demand). Tensor selection is by SHAPE (trailing dim 1280 =
embeddings) EXCEPT the jamendo moodtheme head, which has TWO 56-d outputs
(model/Sigmoid predictions vs model/dense_1/BiasAdd logits) and therefore
loads by the PINNED graph-output index (config.MOODTHEME_PROB_OUTPUT_INDEX).

mood_scores is ATOMIC: emitted only when all five mood heads are healthy —
a missing head read as 0.0 would mean "maximally non-X" and bias octave
gating toward intensity during partial failure.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from . import config as cfg
from . import frontend
from .models import ModelManager

_LOGGER = logging.getLogger(__name__)


class InferenceEngine:
    """Warm ONNX Runtime sessions + the analyze() pipeline."""

    def __init__(self, models: ModelManager) -> None:
        self._models = models
        self._sessions: dict[str, object] = {}

    @property
    def sessions_loaded(self) -> bool:
        return bool(self._sessions)

    def load_sessions(self) -> None:
        """Create ONNX Runtime CPU sessions for every verified model.

        Thread-capped: intra_op default 2 (add-on option), inter_op 1 —
        the sidecar shares host CPU with HA core; unbounded thread use can
        stutter core.
        """
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = cfg.INTRA_OP_THREADS
        opts.inter_op_num_threads = cfg.INTER_OP_THREADS
        opts.log_severity_level = 3

        for name, state in self._models.states.items():
            if state != "ok":
                continue
            path = self._models.model_path(name)
            try:
                self._sessions[name] = ort.InferenceSession(
                    str(path), sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                _LOGGER.info("AX-BPM sidecar: session loaded for %s", name)
            except Exception as err:  # noqa: BLE001 — degrade, never crash
                _LOGGER.error(
                    "AX-BPM sidecar: session load failed for %s: %s", name, err
                )
                self._models.states[name] = "error"

    def _run(self, name: str, patches: np.ndarray) -> list[np.ndarray]:
        """Run one session; returns all outputs."""
        session = self._sessions[name]
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: patches})
        return [np.asarray(o) for o in outputs]

    @staticmethod
    def _select_by_shape(
        outputs: list[np.ndarray], trailing_dim: int
    ) -> np.ndarray:
        """Pick the output whose trailing dim matches (embeddings=1280).

        TF names like `PartitionedCall:1` may not survive ONNX export —
        selection must be by SHAPE, not by name.
        """
        candidates = [
            o for o in outputs if o.ndim >= 2 and o.shape[-1] == trailing_dim
        ]
        if not candidates:
            shapes = [list(o.shape) for o in outputs]
            raise ValueError(
                f"no output with trailing dim {trailing_dim}; shapes={shapes}"
            )
        return candidates[0]

    def _embeddings(self, patches: np.ndarray) -> np.ndarray:
        """effnet bsdynamic → per-patch embeddings → mean pool (1280-d)."""
        outputs = self._run("effnet", patches)
        emb = InferenceEngine._select_by_shape(outputs, 1280)
        pooled = emb.reshape(-1, emb.shape[-1]).mean(axis=0)
        return pooled.astype(np.float32)

    def _head(self, name: str, pooled: np.ndarray) -> np.ndarray | None:
        """Run a classification head on the pooled embedding.

        2-class heads: shape-based selection ([2] predictions vs [100]
        penultimate). moodtheme: PINNED output index (dual 56-d outputs).
        """
        if name not in self._sessions:
            return None
        outputs = self._run(name, pooled.reshape(1, -1))
        if name == "moodtheme":
            idx = cfg.MOODTHEME_PROB_OUTPUT_INDEX
            out = outputs[idx].reshape(-1)
            # Pinned output must be probabilities in [0,1] — logits fail.
            if out.min() < 0.0 or out.max() > 1.0 + 1e-6:
                _LOGGER.error(
                    "AX-BPM sidecar: pinned moodtheme output index %d is "
                    "not in [0,1] (range [%s, %s]) — wrong pin?",
                    idx, out.min(), out.max(),
                )
                return None
            return out
        candidates = [o for o in outputs if o.ndim >= 2 and o.shape[-1] == 2]
        if not candidates:
            return None
        return candidates[0].reshape(-1)

    def analyze(self, audio: np.ndarray) -> dict[str, Any] | None:
        """Full analysis: front end → effnet → heads → payload dict.

        Returns None when effnet is unavailable (nothing can be computed).
        """
        if "effnet" not in self._sessions:
            return None

        patches = frontend.front_end(audio)
        if patches.shape[0] == 0:
            return None
        pooled = self._embeddings(patches)

        payload: dict[str, Any] = {
            "valence": None,
            "arousal": None,
            "valence_std": None,
            "arousal_std": None,
            "analyzed_seconds": round(len(audio) / cfg.SAMPLE_RATE, 3),
            "model_versions": {
                name: f"v{pin['version']} ({pin['release_date']})"
                for name, pin in cfg.MODEL_PINS.items()
                if self._models.is_loaded(name)
            },
        }

        # mood_tags — attribute layer only (moodtheme PR-AUC 0.14).
        theme = self._head("moodtheme", pooled)
        if theme is not None and len(theme) == len(cfg.MOODTHEME_CLASSES):
            tags = [
                {"tag": cls, "score": round(float(score), 4)}
                for cls, score in zip(cfg.MOODTHEME_CLASSES, theme)
                if float(score) >= cfg.MOOD_TAG_THRESHOLD
            ]
            tags.sort(key=lambda t: t["score"], reverse=True)
            payload["mood_tags"] = tags[: cfg.MOOD_TAG_TOP_N]
        else:
            payload["mood_tags"] = []

        # danceability — degrades independently (attribute only).
        dance = self._head("danceability", pooled)
        if dance is not None and len(dance) == 2:
            payload["danceability"] = round(
                float(dance[cfg.POSITIVE_CLASS_INDEX["danceability"]]), 4
            )

        # mood_scores — ATOMIC: all five heads or omit entirely.
        head_outputs: dict[str, np.ndarray] = {}
        for head in cfg.MOOD_HEADS:
            out = self._head(f"mood_{head}", pooled)
            if out is not None and len(out) == 2:
                head_outputs[head] = out
        if len(head_outputs) == len(cfg.MOOD_HEADS):
            payload["mood_scores"] = {
                head: round(
                    float(head_outputs[head][cfg.POSITIVE_CLASS_INDEX[f"mood_{head}"]]),
                    4,
                )
                for head in cfg.MOOD_HEADS
            }
        # else: key omitted entirely — the integration falls back to
        # genre-only gating (never substitutes 0.0).

        return payload