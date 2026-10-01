"""Discovery announcement tests: endpoint, payload, best-effort contract.

Regression for the 3.1.0 fix: the announcement was POSTed to
/services/discovery (nonexistent → HTTP 404) instead of the Supervisor's
real POST /discovery endpoint, so the integration never learned the
add-on's hostname and fell back to an unresolvable homeassistant.local.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from ax_bpm_analyzer import config as cfg
from ax_bpm_analyzer import discovery


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_announce_posts_to_discovery_endpoint():
    """The announcement POSTs to /discovery with service + host/port config."""
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["auth"] = req.get_header("Authorization")
        if req.data is not None:
            captured["payload"] = json.loads(req.data.decode())
        if req.full_url.endswith("/addons/self/info"):
            return _FakeResponse(
                json.dumps({"data": {"hostname": "abc123-ax-bpm-analyzer"}}).encode()
            )
        return _FakeResponse(b"{}")

    with patch.object(discovery.urllib.request, "urlopen", fake_urlopen):
        discovery._announce_sync("test-token")

    # The Supervisor's real discovery endpoint (POST /discovery) — NOT the
    # nonexistent /services/discovery that caused the 404 regression.
    assert captured["url"] == "http://supervisor/discovery"
    assert captured["method"] == "POST"
    assert captured["auth"] == "Bearer test-token"
    assert captured["payload"] == {
        "service": "ax_bpm",
        "config": {"host": "abc123-ax-bpm-analyzer", "port": cfg.PORT},
    }


def test_announce_hostname_read_from_self_info():
    """The hostname comes from GET /addons/self/info (correct endpoint)."""
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured.setdefault("urls", []).append(req.full_url)
        if req.full_url.endswith("/addons/self/info"):
            return _FakeResponse(
                json.dumps({"data": {"hostname": "my-addon-host"}}).encode()
            )
        return _FakeResponse(b"{}")

    with patch.object(discovery.urllib.request, "urlopen", fake_urlopen):
        discovery._announce_sync("tok")

    assert captured["urls"] == [
        "http://supervisor/addons/self/info",
        "http://supervisor/discovery",
    ]


@pytest.mark.asyncio
async def test_async_announce_swallows_errors():
    """Best-effort contract: any failure is swallowed, never raised."""
    def fake_urlopen(req, timeout=None):
        raise OSError("connection refused")

    with patch.object(discovery.urllib.request, "urlopen", fake_urlopen):
        # Must not raise.
        await discovery.async_announce()


@pytest.mark.asyncio
async def test_async_announce_skipped_without_supervisor_token(monkeypatch):
    """No SUPERVISOR_TOKEN → no announcement attempt (non-HAOS / CI)."""
    monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
    monkeypatch.delenv("AXBPM_SKIP_BOOTSTRAP", raising=False)
    called = False

    def fake_urlopen(req, timeout=None):
        nonlocal called
        called = True
        return _FakeResponse(b"{}")

    with patch.object(discovery.urllib.request, "urlopen", fake_urlopen):
        await discovery.async_announce()

    assert called is False


@pytest.mark.asyncio
async def test_async_announce_skipped_when_bootstrap_disabled(monkeypatch):
    """AXBPM_SKIP_BOOTSTRAP → no announcement attempt."""
    monkeypatch.setenv("AXBPM_SKIP_BOOTSTRAP", "1")
    called = False

    def fake_urlopen(req, timeout=None):
        nonlocal called
        called = True
        return _FakeResponse(b"{}")

    with patch.object(discovery.urllib.request, "urlopen", fake_urlopen):
        await discovery.async_announce()

    assert called is False
