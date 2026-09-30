"""AX BPM button entities — thin wrappers over the services.

Four buttons on the same device as the sensor:

  Halve BPM        → ax_bpm.halve_bpm
  Double BPM       → ax_bpm.double_bpm
  Clear BPM cache  → ax_bpm.clear_cache
  Clear overrides  → ax_bpm.clear_overrides

Buttons carry no logic of their own: each press calls the corresponding
service via hass.services.async_call, so automations and button presses
share exactly one code path (services.py). A fifth conceptual action —
clearing the override for the current track — is service-only
(clear_override), since it's a per-track undo rather than a device-level
control.

Push-only (should_poll=False); availability follows the config entry.
"""

from __future__ import annotations

import logging

from homeassistant.components.button import ButtonEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, NAME

_LOGGER = logging.getLogger(__name__)

SERVICE_BY_BUTTON = {
    "halve": "halve_bpm",
    "double": "double_bpm",
    "clear_cache": "clear_cache",
    "clear_overrides": "clear_overrides",
}


async def async_setup_entry(
    hass: HomeAssistant,
    entry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the four AX BPM buttons from a config entry."""
    async_add_entities(
        [
            AxBpmButton(entry, "halve"),
            AxBpmButton(entry, "double"),
            AxBpmButton(entry, "clear_cache"),
            AxBpmButton(entry, "clear_overrides"),
        ]
    )


class AxBpmButton(ButtonEntity):
    """One AX BPM action button (service wrapper)."""

    _attr_should_poll = False

    def __init__(self, entry, action: str) -> None:
        self._entry = entry
        self._action = action
        self._attr_unique_id = f"{entry.entry_id}_btn_{action}"
        self._attr_translation_key = f"btn_{action}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=NAME,
            entry_type=DeviceEntryType.SERVICE,
        )

    async def async_press(self) -> None:
        """Call the wrapped service."""
        service = SERVICE_BY_BUTTON[self._action]
        _LOGGER.debug("AX BPM button %s → service %s", self._action, service)
        await self.hass.services.async_call(
            DOMAIN, service, blocking=True
        )
