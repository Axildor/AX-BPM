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
    storage.Store = Store


_stub_homeassistant()