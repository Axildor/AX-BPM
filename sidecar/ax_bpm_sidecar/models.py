"""Model manager: first-start download, sha256+size verify, per-model state.

Weights are NEVER committed (CC BY-NC-SA). On first start each pinned
model is downloaded into the data dir, verified against its sha256 + size
pin, and retried with backoff. A model that fails to download degrades
health (per-model state) — it never crashes the service.
"""

from __future__ import annotations

import hashlib
import logging
import time
import urllib.request
from pathlib import Path

from . import config as cfg

_LOGGER = logging.getLogger(__name__)


class ModelManager:
    """Downloads, verifies, and tracks per-model state."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir else cfg.DATA_DIR
        self._states: dict[str, str] = {
            name: "pending" for name in cfg.MODEL_PINS
        }

    @property
    def states(self) -> dict[str, str]:
        """Per-model state: pending | ok | error (live dict, not a copy)."""
        return self._states

    @property
    def status(self) -> str:
        """Aggregate status: ok (all), degraded (some), error (effnet)."""
        states = list(self._states.values())
        if all(s == "ok" for s in states):
            return "ok"
        if self._states.get("effnet") != "ok":
            return "error"
        return "degraded"

    def model_path(self, name: str) -> Path:
        return self._data_dir / cfg.MODEL_PINS[name]["filename"]

    def is_loaded(self, name: str) -> bool:
        return self._states.get(name) == "ok"

    def verify_existing(self) -> None:
        """Verify already-downloaded weights (restart path)."""
        for name, pin in cfg.MODEL_PINS.items():
            path = self._data_dir / pin["filename"]
            if not path.is_file():
                self._states[name] = "pending"
                continue
            self._states[name] = (
                "ok" if self._verify_file(path, pin) else "error"
            )

    @staticmethod
    def _verify_file(path: Path, pin: dict) -> bool:
        if path.stat().st_size != pin["size"]:
            return False
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return digest == pin["sha256"]

    def ensure_all(self) -> None:
        """Download any missing/invalid model, retry with backoff.

        Failure degrades the per-model state; never raises.
        """
        self._data_dir.mkdir(parents=True, exist_ok=True)
        for name, pin in cfg.MODEL_PINS.items():
            path = self._data_dir / pin["filename"]
            if path.is_file() and self._verify_file(path, pin):
                self._states[name] = "ok"
                continue
            self._states[name] = "pending"
            if self._download(name, path, pin):
                self._states[name] = "ok"
            else:
                self._states[name] = "error"
                _LOGGER.error(
                    "AX-BPM sidecar: model %s failed to download after %d "
                    "attempts — degraded health (mood features using this "
                    "model are disabled)",
                    name,
                    cfg.DOWNLOAD_RETRIES,
                )

    def _download(self, name: str, path: Path, pin: dict) -> bool:
        for attempt in range(1, cfg.DOWNLOAD_RETRIES + 1):
            try:
                _LOGGER.info(
                    "AX-BPM sidecar: downloading model %s (%d bytes) from %s",
                    name, pin["size"], pin["url"],
                )
                tmp = path.with_suffix(path.suffix + ".tmp")
                urllib.request.urlretrieve(pin["url"], tmp)
                if self._verify_file(tmp, pin):
                    tmp.replace(path)
                    _LOGGER.info("AX-BPM sidecar: model %s verified (sha256 ok)", name)
                    return True
                _LOGGER.warning(
                    "AX-BPM sidecar: model %s sha256/size mismatch after "
                    "download — retrying", name,
                )
                tmp.unlink(missing_ok=True)
            except Exception as err:  # noqa: BLE001 — download failures retry
                _LOGGER.warning(
                    "AX-BPM sidecar: model %s download attempt %d failed: %s",
                    name, attempt, err,
                )
            if attempt < cfg.DOWNLOAD_RETRIES:
                time.sleep(cfg.DOWNLOAD_BACKOFF_SECONDS * attempt)
        return False