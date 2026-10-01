"""AX BPM Analyzer client (tempo + mood) — integration side.

Talks HTTP to the AX BPM Analyzer add-on: `POST /analyze` with a
multipart preview buffer, `GET /health` for per-model state + tempo
availability. ONE /analyze call returns BOTH the aubio `bpm`
(+ `bpm_confidence`) and the atomic `mood_scores` — the pipeline's local
path makes a single call when the analyzer is healthy.

Contract (from the parent plan):
- Hard timeout, single attempt, NO retry. Any failure returns None —
  never raises into the pipeline.
- URL auto-detect order: manual `analyzer_url` override → discovered URL
  (HA-native Supervisor discovery) → `http://homeassistant.local:8099`
  fallback (external-analyzer mode).
- Empty manual URL = auto-detect only; feature fully off when the mode
  dropdown excludes mood.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import aiohttp

from .const import (
    ANALYZE_PATH,
    ANALYZER_TIMEOUT,
    ANALYZER_URLS,
    HEALTH_PATH,
)

_LOGGER = logging.getLogger(__name__)

# Health probe timeout — shorter than analysis; used by config flow + setup.
HEALTH_TIMEOUT = 3.0

# Re-probe cooldown: when the analyzer was never detected (or the resolved
# URL stopped answering), async_analyze may retry detection at most once per
# this window. Covers startup ordering (add-on boots after HA core) without
# hammering the network on every track change.
REPROBE_COOLDOWN = 300.0


def _read_file(path: str) -> bytes:
    """Blocking file read — must run in an executor, never on the loop."""
    with open(path, "rb") as fh:
        return fh.read()


class AnalyzerClient:
    """Async client for the AX BPM Analyzer service (tempo + mood).

    /analyze carries an optional Bearer shared-secret token (set the same
    token in the add-on config and the integration options); /health stays
    unauthenticated (auto-detect + config-flow status line need it).
    """

    def __init__(
        self,
        session: aiohttp.ClientSession,
        manual_url: str | None,
        api_token: str | None = None,
        discovered_url: str | None = None,
    ) -> None:
        self._session = session
        self._manual_url = (manual_url or "").strip().rstrip("/") or None
        self._discovered_url = (discovered_url or "").strip().rstrip("/") or None
        self._api_token = (api_token or "").strip() or None
        self._resolved_url: str | None = None
        self._probed = False
        self._last_failed_detect: float | None = None

    @property
    def base_url(self) -> str | None:
        """The resolved analyzer base URL, or None when not detected."""
        return self._resolved_url

    async def async_detect(self, force: bool = False) -> str | None:
        """Resolve the analyzer URL (cached; cooldown-bounded re-probe).

        Auto-detect order: manual override → discovered URL (Supervisor
        discovery) → homeassistant.local fallback. Returns the working
        base URL or None.

        force=True bypasses the cooldown (used by async_analyze when the
        resolved URL stopped answering). Without force, a failed detection
        is retried at most once per REPROBE_COOLDOWN seconds so startup
        ordering (add-on boots after HA core) self-heals without
        hammering the network on every track change.
        """
        if self._probed and not force:
            if self._resolved_url is not None:
                return self._resolved_url
            if (
                self._last_failed_detect is not None
                and time.monotonic() - self._last_failed_detect
                < REPROBE_COOLDOWN
            ):
                return None
        self._probed = True

        candidates: list[str] = []
        if self._manual_url:
            candidates.append(self._manual_url)
        if self._discovered_url and self._discovered_url not in candidates:
            candidates.append(self._discovered_url)
        candidates.extend(ANALYZER_URLS)

        for url in candidates:
            if await self._async_health(url):
                self._resolved_url = url
                self._last_failed_detect = None
                _LOGGER.info("AX BPM: analyzer detected at %s", url)
                return url
        self._resolved_url = None
        self._last_failed_detect = time.monotonic()
        if self._manual_url:
            _LOGGER.info(
                "AX BPM: analyzer not reachable at %s — mood attributes "
                "disabled and local tempo degrades to the NumPy floor",
                self._manual_url,
            )
        else:
            _LOGGER.info(
                "AX BPM: analyzer not detected — mood attributes disabled "
                "and local tempo degrades to the NumPy floor"
            )
        return None

    async def _async_health(self, base_url: str) -> dict[str, Any] | None:
        """GET /health. Returns the JSON body or None on any failure."""
        try:
            async with self._session.get(
                f"{base_url}{HEALTH_PATH}",
                timeout=aiohttp.ClientTimeout(total=HEALTH_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    return None
                return await resp.json(content_type=None)
        except (TimeoutError, aiohttp.ClientError, ValueError):
            return None

    async def async_health(self) -> dict[str, Any] | None:
        """Public health probe against the resolved (or manual) URL."""
        base = self._resolved_url or self._manual_url or self._discovered_url
        if not base:
            return None
        return await self._async_health(base)

    async def async_analyze_file(self, path: str) -> dict[str, Any] | None:
        """Read a preview file (executor) and POST it to /analyze."""
        loop = asyncio.get_running_loop()
        try:
            preview = await loop.run_in_executor(None, _read_file, path)
        except OSError as err:
            _LOGGER.debug("AX BPM analyzer preview read failed: %s", err)
            return None
        if not preview:
            return None
        return await self.async_analyze(preview)

    async def async_analyze(self, preview: bytes) -> dict[str, Any] | None:
        """POST the preview buffer to /analyze (multipart field "file").

        Returns the payload dict (bpm + bpm_confidence + mood_scores +
        tags) on success, None on any failure (timeout, HTTP error, 5xx,
        unreachable). Single attempt per URL; on a transport failure the
        client force-re-detects once (cooldown-free) so a moved/restarted
        analyzer is picked up on the next call.
        """
        base = self._resolved_url or self._manual_url or self._discovered_url
        if not base:
            # Never detected (or cooldown-blocked): allow one cooldown-bounded
            # re-detect so startup ordering self-heals without user action.
            await self.async_detect()
            base = self._resolved_url
            if not base:
                return None
        form = aiohttp.FormData()
        form.add_field(
            "file", preview, filename="preview.mp3", content_type="audio/mpeg"
        )

        headers = (
            {"Authorization": f"Bearer {self._api_token}"}
            if self._api_token
            else {}
        )

        async def _post() -> dict[str, Any] | None:
            async with self._session.post(
                f"{base}{ANALYZE_PATH}",
                data=form,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=ANALYZER_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    _LOGGER.debug(
                        "AX BPM analyzer /analyze returned HTTP %s", resp.status
                    )
                    return None
                return await resp.json(content_type=None)

        try:
            # Hard outer timeout: guarantees the single attempt can never
            # hang past ANALYZER_TIMEOUT regardless of transport behavior.
            return await asyncio.wait_for(_post(), timeout=ANALYZER_TIMEOUT)
        except (TimeoutError, aiohttp.ClientError, OSError, ValueError) as err:
            _LOGGER.debug("AX BPM analyzer /analyze failed: %s", err)
            # The resolved URL stopped answering — force one re-detect so
            # the NEXT call can target a recovered/moved analyzer.
            await self.async_detect(force=True)
            return None
