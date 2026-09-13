"""Supervisor discovery announcement (HA-native add-on discovery).

When running as a Home Assistant add-on, the service announces itself to
the Supervisor so the AX BPM integration learns the REAL resolvable
hostname + port instead of guessing. The Supervisor routes the discovery
to the integration's `async_step_hassio` config-flow step.

Why this exists: HA derives an add-on's DNS name as `{REPO}_{SLUG}` with
underscores replaced by hyphens, where `{REPO}` is a hashed identifier
for GitHub-repo add-ons — NOT the plain slug. So a hardcoded
`http://ax-bpm-analyzer:8099` candidate does not resolve in production.
The add-on reads its own `hostname` from the Supervisor and announces it.

Contract:
- Best-effort ONLY. Any failure is logged once and ignored — discovery
  must never crash or delay the service (same degraded-never-crash
  philosophy as model loading).
- Skipped entirely when `SUPERVISOR_TOKEN` is absent (non-HAOS installs,
  CI) or when `AXBPM_SKIP_BOOTSTRAP` is set.
- Uses stdlib urllib in a worker thread — no extra runtime dependency.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request

from . import config as cfg

_LOGGER = logging.getLogger(__name__)

SUPERVISOR_BASE = "http://supervisor"
# Discovery service name — MUST match the integration domain so the
# Supervisor routes the flow to `ax_bpm`'s async_step_hassio.
DISCOVERY_SERVICE = "ax_bpm"
# Timeout for both Supervisor calls (local socket, should be instant).
_TIMEOUT = 5.0


def _supervisor_token() -> str | None:
    """The Supervisor token, or None when not running under Supervisor."""
    return os.environ.get("SUPERVISOR_TOKEN") or None


def _request(
    method: str, path: str, token: str, payload: dict | None = None
) -> dict | None:
    """Blocking Supervisor API call. Returns the parsed body or None."""
    url = f"{SUPERVISOR_BASE}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        body = resp.read()
    if not body:
        return {}
    return json.loads(body)


def _announce_sync(token: str) -> None:
    """Read our own hostname, then POST the discovery entry."""
    info = _request("GET", "/addons/self/info", token) or {}
    hostname = (info.get("data") or {}).get("hostname")
    if not hostname:
        _LOGGER.warning(
            "AX BPM Analyzer: Supervisor reported no hostname — "
            "skipping discovery announcement"
        )
        return
    _request(
        "POST",
        "/services/discovery",
        token,
        {"service": DISCOVERY_SERVICE, "config": {"host": hostname, "port": cfg.PORT}},
    )
    _LOGGER.info(
        "AX BPM Analyzer: announced discovery to Supervisor (%s:%s)",
        hostname,
        cfg.PORT,
    )


async def async_announce() -> None:
    """Announce this add-on to the Supervisor. Never raises."""
    if os.environ.get("AXBPM_SKIP_BOOTSTRAP"):
        return
    token = _supervisor_token()
    if not token:
        _LOGGER.debug(
            "AX BPM Analyzer: no SUPERVISOR_TOKEN — not running as an "
            "add-on, discovery announcement skipped"
        )
        return
    try:
        await asyncio.to_thread(_announce_sync, token)
    except (urllib.error.URLError, OSError, ValueError, KeyError) as err:
        # Best-effort: the integration still finds us via the
        # homeassistant.local:8099 fallback.
        _LOGGER.warning(
            "AX BPM Analyzer: discovery announcement failed (%s) — the "
            "integration will fall back to homeassistant.local:%s",
            err,
            cfg.PORT,
        )
