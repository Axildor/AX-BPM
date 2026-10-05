"""Test bootstrap: register the integration as an importable package.

The integration's __init__.py imports homeassistant, which is not needed
for unit-testing the pure modules. We register a lightweight package
entry for `ax_bpm` pointing at the component directory (skipping
__init__.py execution) and stub the homeassistant modules that the
imported modules reference at import time.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

COMPONENT_DIR = Path(__file__).parent.parent / "custom_components" / "ax_bpm"

# --- Fake `ax_bpm` package (skips __init__.py execution) -------------------
if "ax_bpm" not in sys.modules:
    pkg = types.ModuleType("ax_bpm")
    pkg.__path__ = [str(COMPONENT_DIR)]
    pkg.__package__ = "ax_bpm"
    sys.modules["ax_bpm"] = pkg

# --- Stub homeassistant modules used at import time ------------------------


def _ensure_module(name: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    # Attach to parent.
    if "." in name:
        parent_name, _, child = name.rpartition(".")
        parent = _ensure_module(parent_name)
        setattr(parent, child, mod)
    return mod


def _stub_homeassistant() -> None:
    ha = _ensure_module("homeassistant")
    ha.__path__ = []  # mark as package

    core = _ensure_module("homeassistant.core")
    class _HomeAssistant:  # annotation-only usage
        pass
    core.HomeAssistant = _HomeAssistant
    core.callback = lambda fn: fn

    config_entries = _ensure_module("homeassistant.config_entries")
    class _ConfigEntry:  # annotation-only usage
        pass
    config_entries.ConfigEntry = _ConfigEntry
    config_entries.ConfigFlow = type("ConfigFlow", (), {"__init_subclass__": classmethod(lambda cls, **kwargs: None)})
    config_entries.OptionsFlow = type("OptionsFlow", (), {})

    helpers = _ensure_module("homeassistant.helpers")
    helpers.__path__ = []

    # Stub submodules referenced at import time by config_flow.py.
    selector = _ensure_module("homeassistant.helpers.selector")
    class _SelectorConfig:  # attribute-bag used only at schema-build time
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    selector.EntitySelector = lambda cfg=None: ("entity", cfg)
    selector.EntitySelectorConfig = _SelectorConfig
    selector.SelectSelector = lambda cfg=None: ("select", cfg)
    selector.SelectSelectorConfig = _SelectorConfig
    selector.SelectSelectorMode = type("SelectSelectorMode", (), {"DROPDOWN": "dropdown"})

    aiohttp_client = _ensure_module("homeassistant.helpers.aiohttp_client")
    aiohttp_client.async_get_clientsession = lambda hass: None

    storage = _ensure_module("homeassistant.helpers.storage")
    class Store:  # replaced/mocked in tests; never instantiated here
        def __init__(self, *args, **kwargs):
            pass
        def __class_getitem__(cls, item):
            # Support Store[dict[str, Any]] subscripting (real HA Store is
            # Generic); return the class itself so subclassing works.
            return cls
    storage.Store = Store

    # --- Stubs for sensor.py / button.py (entity regression tests) ----------
    const = _ensure_module("homeassistant.const")
    const.STATE_OFF = "off"
    const.STATE_PLAYING = "playing"

    components = _ensure_module("homeassistant.components")
    components.__path__ = []  # mark as package

    sensor_mod = _ensure_module("homeassistant.components.sensor")
    class _EntityBase:  # minimal base; tests never touch HA internals
        def async_on_remove(self, func):
            # Real HA stores the unsub and calls it on removal; tests only
            # need the registration to succeed.
            return func
    sensor_mod.SensorEntity = _EntityBase
    sensor_mod.SensorStateClass = type(
        "SensorStateClass", (), {"MEASUREMENT": "measurement"}
    )

    button_mod = _ensure_module("homeassistant.components.button")
    button_mod.ButtonEntity = _EntityBase

    device_registry = _ensure_module("homeassistant.helpers.device_registry")
    class _DeviceEntryType:
        SERVICE = "service"
    device_registry.DeviceEntryType = _DeviceEntryType
    class _DeviceInfo:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    device_registry.DeviceInfo = _DeviceInfo

    entity_platform = _ensure_module("homeassistant.helpers.entity_platform")
    entity_platform.AddEntitiesCallback = object  # annotation-only usage

    update_coordinator = _ensure_module("homeassistant.helpers.update_coordinator")
    class _DataUpdateCoordinator:  # minimal base; tests subclass/mock it
        def __init__(self, hass, logger, name=None, update_interval=None):
            self.hass = hass
            self.logger = logger
            self.name = name
            self.update_interval = update_interval
            self.data = None
        def async_set_updated_data(self, data):
            self.data = data
    update_coordinator.DataUpdateCoordinator = _DataUpdateCoordinator
    class _CoordinatorEntity:  # minimal mixin; tests set attributes directly
        def __init__(self, coordinator, **kwargs):
            super().__init__()
            self.coordinator = coordinator
        @property
        def available(self):
            return True
    update_coordinator.CoordinatorEntity = _CoordinatorEntity

    event = _ensure_module("homeassistant.helpers.event")
    def _track_state_change_event(hass, entities, action):
        # Record-free stub: returns an unsubscribe callable.
        return lambda: None
    event.async_track_state_change_event = _track_state_change_event
    def _call_later(hass, delay, action):
        # Fire-and-forget stub: returns a cancel callable (never fired —
        # the debounce path is not under test here).
        return lambda: None
    event.async_call_later = _call_later


_stub_homeassistant()
