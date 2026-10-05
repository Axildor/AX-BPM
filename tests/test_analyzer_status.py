"""Tests for the analyzer connectivity status (coordinator + sensor).

Covers:
- Coordinator poll: connected / disconnected / disabled states.
- Push path: the analyzer client's analyze-listener fires on /analyze
  success and failure, updating the status without waiting for a poll.
- Sensor: native_value mirrors coordinator data; attributes carry the
  resolved URL + health payload; disabled state suppresses push updates.
- Client hook: set_analyze_listener registration + failure isolation
  (a raising listener must never break the /analyze call).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ax_bpm.analyzer_client import AnalyzerClient
from ax_bpm.analyzer_status import (
    AnalyzerStatusCoordinator,
    AxBpmAnalyzerStatusSensor,
)
from ax_bpm.const import (
    DOMAIN,
    STATUS_CONNECTED,
    STATUS_DISABLED,
    STATUS_DISCONNECTED,
)

ENTRY_ID = "01M3T3Z3VYVXYKYZJP7PP1NRBX"


class _FakePipeline:
    """Minimal pipeline surface used by the coordinator."""

    def __init__(self, mood_enabled: bool = True) -> None:
        self._mood_enabled = mood_enabled
        self.client = AnalyzerClient(MagicMock(), None)
        self.analyzer_client = self.client

    @property
    def mood_enabled(self) -> bool:
        return self._mood_enabled


class _FakeHass:
    """Minimal hass for the coordinator (no HA runtime)."""

    def __init__(self) -> None:
        self.data = {}


class _Ctx:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *args):
        return False


def _post_ctx(status: int, body: dict | None = None) -> MagicMock:
    """A session.post returning a context manager with the given response."""
    resp = MagicMock()
    resp.status = status

    async def json(content_type=None):
        return body

    resp.json = json
    return MagicMock(return_value=_Ctx(resp))


# ---------------------------------------------------------------------------
# Coordinator — poll path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poll_connected():
    """Health 200 + detected URL → connected with url + health attrs."""
    pipeline = _FakePipeline()
    client = pipeline.client
    client.async_detect = AsyncMock(return_value="http://analyzer:8099")
    client.async_health = AsyncMock(
        return_value={
            "status": "ok",
            "models_loaded": {"effnet": "ok", "mood": "ok"},
            "tempo_available": True,
        }
    )
    coordinator = AnalyzerStatusCoordinator(_FakeHass(), pipeline)
    data = await coordinator._async_update_data()
    assert data["state"] == STATUS_CONNECTED
    assert data["url"] == "http://analyzer:8099"
    assert data["health"]["status"] == "ok"
    assert data["last_check"]


@pytest.mark.asyncio
async def test_poll_disconnected_when_health_fails():
    """Detected URL but /health fails → disconnected."""
    pipeline = _FakePipeline()
    client = pipeline.client
    client.async_detect = AsyncMock(return_value="http://analyzer:8099")
    client.async_health = AsyncMock(return_value=None)
    coordinator = AnalyzerStatusCoordinator(_FakeHass(), pipeline)
    data = await coordinator._async_update_data()
    assert data["state"] == STATUS_DISCONNECTED


@pytest.mark.asyncio
async def test_poll_disconnected_when_not_detected():
    """Detection fails → disconnected, no health probe attempted."""
    pipeline = _FakePipeline()
    client = pipeline.client
    client.async_detect = AsyncMock(return_value=None)
    client.async_health = AsyncMock()
    coordinator = AnalyzerStatusCoordinator(_FakeHass(), pipeline)
    data = await coordinator._async_update_data()
    assert data["state"] == STATUS_DISCONNECTED
    assert data["url"] is None
    client.async_health.assert_not_called()


@pytest.mark.asyncio
async def test_poll_disabled_when_mood_off():
    """Octave mode off → disabled, no network calls at all."""
    pipeline = _FakePipeline(mood_enabled=False)
    client = pipeline.client
    client.async_detect = AsyncMock()
    client.async_health = AsyncMock()
    coordinator = AnalyzerStatusCoordinator(_FakeHass(), pipeline)
    data = await coordinator._async_update_data()
    assert data["state"] == STATUS_DISABLED
    assert data["url"] is None
    client.async_detect.assert_not_called()
    client.async_health.assert_not_called()


# ---------------------------------------------------------------------------
# Push path — analyze listener
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_push_listener_fires_on_success():
    """The client fires the listener after a successful /analyze."""
    pipeline = _FakePipeline()
    coordinator = AnalyzerStatusCoordinator(_FakeHass(), pipeline)
    client = pipeline.client

    # Coordinator data must exist before a push (mirrors a prior poll).
    coordinator.data = {
        "state": STATUS_DISCONNECTED,
        "url": None,
        "health": None,
        "last_check": None,
    }

    client._resolved_url = "http://analyzer:8099"
    client._session = MagicMock()
    client._session.post = _post_ctx(200, {"bpm": 120.0})

    result = await client.async_analyze(b"mp3")
    assert result == {"bpm": 120.0}
    assert coordinator.data["state"] == STATUS_CONNECTED
    assert coordinator.data["last_analyze"] == "ok"
    assert coordinator.data["url"] == "http://analyzer:8099"


@pytest.mark.asyncio
async def test_push_listener_fires_on_failure():
    """A failed /analyze pushes disconnected immediately."""
    pipeline = _FakePipeline()
    coordinator = AnalyzerStatusCoordinator(_FakeHass(), pipeline)
    client = pipeline.client
    coordinator.data = {
        "state": STATUS_CONNECTED,
        "url": "http://analyzer:8099",
        "health": None,
        "last_check": None,
    }

    client._resolved_url = "http://analyzer:8099"
    client._session = MagicMock()
    client._session.post = _post_ctx(500)

    assert await client.async_analyze(b"mp3") is None
    assert coordinator.data["state"] == STATUS_DISCONNECTED
    assert coordinator.data["last_analyze"] == "failed"


@pytest.mark.asyncio
async def test_push_suppressed_when_disabled():
    """A /analyze outcome must not resurrect a disabled status."""
    pipeline = _FakePipeline(mood_enabled=False)
    coordinator = AnalyzerStatusCoordinator(_FakeHass(), pipeline)
    coordinator.data = {
        "state": STATUS_DISABLED,
        "url": None,
        "health": None,
        "last_check": None,
    }

    pipeline.client._notify_analyze_result(True)
    assert coordinator.data["state"] == STATUS_DISABLED


# ---------------------------------------------------------------------------
# Sensor entity
# ---------------------------------------------------------------------------


def _make_sensor(coordinator) -> AxBpmAnalyzerStatusSensor:
    class _Entry:
        entry_id = ENTRY_ID

    return AxBpmAnalyzerStatusSensor(_Entry(), coordinator)


def test_sensor_mirrors_coordinator_state():
    """native_value mirrors coordinator data; attrs carry url + health."""
    coordinator = MagicMock()
    coordinator.data = {
        "state": STATUS_CONNECTED,
        "url": "http://analyzer:8099",
        "health": {
            "status": "ok",
            "models_loaded": {"effnet": "ok"},
            "tempo_available": True,
        },
        "last_check": "2026-10-05T00:00:00+00:00",
    }
    sensor = _make_sensor(coordinator)
    assert sensor.native_value == STATUS_CONNECTED
    attrs = sensor.extra_state_attributes
    assert attrs["analyzer_url"] == "http://analyzer:8099"
    assert attrs["health_status"] == "ok"
    assert attrs["models_loaded"] == {"effnet": "ok"}
    assert attrs["tempo_available"] is True


def test_sensor_attrs_when_no_health():
    """No health payload → url + last_check only, no health keys."""
    coordinator = MagicMock()
    coordinator.data = {
        "state": STATUS_DISCONNECTED,
        "url": None,
        "health": None,
        "last_check": "2026-10-05T00:00:00+00:00",
    }
    sensor = _make_sensor(coordinator)
    assert sensor.native_value == STATUS_DISCONNECTED
    attrs = sensor.extra_state_attributes
    assert attrs["analyzer_url"] is None
    assert "health_status" not in attrs
    assert "models_loaded" not in attrs
    assert "tempo_available" not in attrs


def test_sensor_entity_metadata():
    """Entity-name mode + translation key + unique_id + device info."""
    coordinator = MagicMock()
    coordinator.data = {"state": STATUS_CONNECTED}
    sensor = _make_sensor(coordinator)
    assert sensor._attr_has_entity_name is True
    assert sensor._attr_translation_key == "analyzer_status"
    assert sensor._attr_unique_id == f"{ENTRY_ID}_analyzer_status"
    # The conftest DeviceInfo stub stores kwargs as attributes.
    assert sensor._attr_device_info.identifiers == {(DOMAIN, ENTRY_ID)}


# ---------------------------------------------------------------------------
# Client hook — registration + failure isolation
# ---------------------------------------------------------------------------


def test_listener_registration():
    """set_analyze_listener stores the callback."""
    client = AnalyzerClient(MagicMock(), None)
    listener = MagicMock()
    client.set_analyze_listener(listener)
    assert client._analyze_listener is listener


@pytest.mark.asyncio
async def test_listener_failure_isolated():
    """A raising listener never breaks the /analyze call."""
    client = AnalyzerClient(MagicMock(), None)
    client._resolved_url = "http://analyzer:8099"
    client.set_analyze_listener(MagicMock(side_effect=RuntimeError("boom")))

    client._session = MagicMock()
    client._session.post = _post_ctx(200, {"bpm": 120.0})

    assert await client.async_analyze(b"mp3") == {"bpm": 120.0}
