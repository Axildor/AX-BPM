"""Analyzer connectivity status — coordinator + sensor entity.

Reports whether the AX BPM Analyzer add-on is reachable:

- connected    /health answered 200 (or the last /analyze succeeded)
- disconnected analyzer unreachable (or the last /analyze failed)
- disabled     octave mode is not "Genre + mood" — the analyzer is
               never consulted in that mode

Two update paths:
- Poll: a DataUpdateCoordinator refreshes every 60 s with a cheap local
  /health probe (HA best practice for periodic status).
- Push: the analyzer client notifies the coordinator after every
  /analyze outcome, so the status reflects real usage immediately
  instead of waiting for the next poll.

The sensor never blocks setup: the first poll runs as a background task
so HA startup is never delayed by a network probe.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from homeassistant.components.sensor import SensorEntity
from homeassistant.core import callback
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
)

from .const import (
    ANALYZER_STATUS_POLL_SECONDS,
    DOMAIN,
    NAME,
    STATUS_CONNECTED,
    STATUS_DISABLED,
    STATUS_DISCONNECTED,
)

_LOGGER = logging.getLogger(__name__)


class AnalyzerStatusCoordinator(DataUpdateCoordinator):
    """Polls the analyzer /health endpoint; accepts push updates."""

    def __init__(self, hass, pipeline) -> None:
        super().__init__(
            hass,
            _LOGGER,
            name=f"{DOMAIN}_analyzer_status",
            update_interval=timedelta(seconds=ANALYZER_STATUS_POLL_SECONDS),
        )
        self._pipeline = pipeline
        # Push path: the client fires this after every /analyze outcome.
        pipeline.analyzer_client.set_analyze_listener(self._on_analyze_result)

    async def _async_update_data(self) -> dict:
        """One poll cycle: detect (cooldown-bounded) + /health probe."""
        if not self._pipeline.mood_enabled:
            return {
                "state": STATUS_DISABLED,
                "url": None,
                "health": None,
                "last_check": datetime.now(UTC).isoformat(),
            }
        client = self._pipeline.analyzer_client
        url = await client.async_detect()
        health = await client.async_health() if url else None
        state = (
            STATUS_CONNECTED if url and health is not None else STATUS_DISCONNECTED
        )
        return {
            "state": state,
            "url": url,
            "health": health,
            "last_check": datetime.now(UTC).isoformat(),
        }

    @callback
    def _on_analyze_result(self, success: bool) -> None:
        """Push update after a real /analyze outcome (never waits to poll).

        A failure maps to disconnected — a 503 (busy) also lands here,
        but the next 60 s poll corrects any transient flap.
        """
        data = dict(self.data or {})
        if data.get("state") == STATUS_DISABLED:
            return  # analyzer not in use; don't resurrect the status
        client = self._pipeline.analyzer_client
        data["state"] = STATUS_CONNECTED if success else STATUS_DISCONNECTED
        if success and client.base_url:
            data["url"] = client.base_url
        data["last_analyze"] = "ok" if success else "failed"
        data["last_analyze_at"] = datetime.now(UTC).isoformat()
        self.async_set_updated_data(data)


class AxBpmAnalyzerStatusSensor(CoordinatorEntity, SensorEntity):
    """Sensor reporting analyzer connectivity (connected/disconnected)."""

    _attr_has_entity_name = True
    _attr_translation_key = "analyzer_status"
    _attr_icon = "mdi:lan-connect"
    _attr_device_class = None  # custom states; no HA device class

    def __init__(self, entry, coordinator: AnalyzerStatusCoordinator) -> None:
        super().__init__(coordinator)
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_analyzer_status"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=NAME,
            entry_type=DeviceEntryType.SERVICE,
        )

    @property
    def native_value(self) -> str | None:
        """connected / disconnected / disabled."""
        data = self.coordinator.data or {}
        return data.get("state")

    @property
    def extra_state_attributes(self) -> dict:
        """Resolved URL + per-model health for diagnostics."""
        data = self.coordinator.data or {}
        attrs: dict = {
            "analyzer_url": data.get("url"),
            "last_check": data.get("last_check"),
        }
        health = data.get("health")
        if isinstance(health, dict):
            attrs["health_status"] = health.get("status")
            attrs["models_loaded"] = health.get("models_loaded")
            attrs["tempo_available"] = health.get("tempo_available")
        if data.get("last_analyze_at"):
            attrs["last_analyze"] = data.get("last_analyze")
            attrs["last_analyze_at"] = data.get("last_analyze_at")
        return attrs

    @callback
    def _handle_coordinator_update(self) -> None:
        """Write state on every coordinator push (poll or /analyze)."""
        self.async_write_ha_state()
