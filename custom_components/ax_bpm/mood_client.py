"""Sidecar mood analyzer client (Phase 1 — integration side).

Talks HTTP to the AX BPM sidecar add-on (Phase 2): `POST /analyze` with a
multipart preview buffer, `GET /health` for per-model state.

Contract (from the parent plan):
- Hard timeout, single attempt, NO retry. Any failure returns None —
  never raises into the pipeline.
- URL auto-detect order: add-on internal hostname →
  `http://homeassistant.local:8099` → manual `mood_analyzer_url` override.
- Empty manual URL = auto-detect only; feature fully off when the mode
  dropdown excludes mood.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

from .const import (
    MOOD_TIMEOUT,
    SIDECAR_ANALYZE_PATH,
    SIDECAR_HEALTH_PATH,
    SIDECAR_URLS,
)

_LOGGER = logging.getLogger(__name__)

# Health probe timeout — shorter than analysis; used by config flow + setup.
HEALTH_TIMEOUT = 3.0


def _read_file(path: str) -> bytes:
    """Blocking file read — must run in an executor, never on the loop."""
    with open(path, "rb") as fh:
        return fh.read()


class MoodClient:
    """Async client for the sidecar mood analyzer service.

    /analyze carries an optional Bearer shared-secret token (set the same
    token in the add-on config and the integration options); /health stays
    unauthenticated (auto-detect + config-flow status line need it).
    """

    def __init__(
        self,
        session: aiohttp.ClientSession,
        manual_url: str | None,
        api_token: str | None = None,
    ) -> None:
        self._session = session
        self._manual_url = (manual_url or "").strip().rstrip("/") or None
        self._api_token = (api_token or "").strip() or None
        self._resolved_url: str | None = None
        self._probed = False

    @property
    def base_url(self) -> str | None:
        """The resolved sidecar base URL, or None when not detected."""
        return self._resolved_url

    async def async_detect(self) -> str | None:
        """Resolve the sidecar URL once. Manual override wins when set.

        Auto-detect order: manual URL → add-on internal hostname →
        homeassistant.local. Returns the working base URL or None.
        """
        if self._probed:
            return self._resolved_url
        self._probed = True

        candidates: list[str] = []
        if self._manual_url:
            candidates.append(self._manual_url)
        candidates.extend(SIDECAR_URLS)

        for url in candidates:
            if await self._async_health(url):
                self._resolved_url = url
                _LOGGER.info("AX BPM: sidecar mood analyzer detected at %s", url)
                return url
        self._resolved_url = None
        if self._manual_url:
            _LOGGER.info(
                "AX BPM: sidecar mood analyzer not reachable at %s — mood "
                "attributes disabled (genre-only octave disambiguation)",
                self._manual_url,
            )
        else:
            _LOGGER.info(
                "AX BPM: sidecar mood analyzer not detected — mood "
                "attributes disabled (genre-only octave disambiguation)"
            )
        return None

    async def _async_health(self, base_url: str) -> dict[str, Any] | None:
        """GET /health. Returns the JSON body or None on any failure."""
        try:
            async with self._session.get(
                f"{base_url}{SIDECAR_HEALTH_PATH}",
                timeout=aiohttp.ClientTimeout(total=HEALTH_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    return None
                return await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return None

    async def async_health(self) -> dict[str, Any] | None:
        """Public health probe against the resolved (or manual) URL."""
        base = self._resolved_url or self._manual_url
        if not base:
            return None
        return await self._async_health(base)

    async def async_analyze_file(self, path: str) -> dict[str, Any] | None:
        """Read a preview file (executor) and POST it to /analyze."""
        loop = asyncio.get_running_loop()
        try:
            preview = await loop.run_in_executor(None, _read_file, path)
        except OSError as err:
            _LOGGER.debug("AX BPM sidecar preview read failed: %s", err)
            return None
        if not preview:
            return None
        return await self.async_analyze(preview)

    async def async_analyze(self, preview: bytes) -> dict[str, Any] | None:
        """POST the preview buffer to /analyze (multipart field "file").

        Returns the mood payload dict on success, None on any failure
        (timeout, HTTP error, 5xx, unreachable). Single attempt, no retry.
        """
        base = self._resolved_url or self._manual_url
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
                f"{base}{SIDECAR_ANALYZE_PATH}",
                data=form,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=MOOD_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    _LOGGER.debug(
                        "AX BPM sidecar /analyze returned HTTP %s", resp.status
                    )
                    return None
                return await resp.json(content_type=None)

        try:
            # Hard outer timeout: guarantees the single attempt can never
            # hang past MOOD_TIMEOUT regardless of transport behavior.
            return await asyncio.wait_for(_post(), timeout=MOOD_TIMEOUT)
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            OSError,
            ValueError,
        ) as err:
            _LOGGER.debug("AX BPM sidecar /analyze failed: %s", err)
            return None