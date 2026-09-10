"""Config flow for AX-BPM (UI only, no YAML)."""

from __future__ import annotations

import logging

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
    CONF_AUBIO_BINARY,
    CONF_GENRE_CORRECTION,
    CONF_GETSONGKEY_API_KEY,
    CONF_HELPER_SCRIPT,
    CONF_MEDIA_PLAYER,
    CONF_MOOD_CORRECTION,
    DOMAIN,
    NAME,
)
from .mood import is_available as essentia_available

_LOGGER = logging.getLogger(__name__)


def _build_schema(defaults: dict) -> vol.Schema:
    mood_default = defaults.get(CONF_MOOD_CORRECTION, True)
    schema = {
        vol.Required(CONF_MEDIA_PLAYER, default=defaults.get(CONF_MEDIA_PLAYER, "")): selector.selector(
            {"entity": {"domain": "media_player"}}
        ),
        vol.Required(CONF_GENRE_CORRECTION, default=defaults.get(CONF_GENRE_CORRECTION, True)): selector.selector(
            {"boolean": {}}
        ),
    }
    if essentia_available():
        schema[
            vol.Required(CONF_MOOD_CORRECTION, default=mood_default)
        ] = selector.selector({"boolean": {}})
    else:
        # Auto-disable mood correction with a warning when Essentia is
        # unavailable; the field is still shown (disabled) for visibility.
        schema[
            vol.Required(CONF_MOOD_CORRECTION, default=False)
        ] = selector.selector(
            {
                "boolean": {},
                "ui": {
                    "description": {
                        "suggested_value": False,
                    }
                },
            }
        )
    schema.update(
        {
            vol.Optional(CONF_AUBIO_BINARY, default=defaults.get(CONF_AUBIO_BINARY, "")): str,
            vol.Optional(CONF_HELPER_SCRIPT, default=defaults.get(CONF_HELPER_SCRIPT, "")): str,
            vol.Optional(CONF_GETSONGKEY_API_KEY, default=defaults.get(CONF_GETSONGKEY_API_KEY, "")): str,
        }
    )
    return vol.Schema(schema)


class AxBpmOptionsHandler(config_entries.OptionsFlow):
    """Handle options for AX-BPM."""

    async def async_step_init(self, user_input=None):
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)
        current = {**self.config_entry.data, **self.config_entry.options}
        return self.async_show_form(
            step_id="init",
            data_schema=_build_schema(current),
        )


class AxBpmConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Config flow for AX-BPM."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        errors = {}
        if user_input is not None:
            if not essentia_available() and user_input.get(CONF_MOOD_CORRECTION):
                user_input[CONF_MOOD_CORRECTION] = False
            # Empty optional strings → drop them.
            user_input = {
                k: v for k, v in user_input.items() if v not in ("", None)
            }
            return self.async_create_entry(title=NAME, data=user_input)

        return self.async_show_form(
            step_id="user",
            data_schema=_build_schema({}),
            errors=errors,
            description_placeholders={
                "mood_warning": (
                    "Essentia is not installed on this system. "
                    "Mood-based octave disambiguation is disabled; "
                    "genre-based correction still works."
                )
                if not essentia_available()
                else ""
            },
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return AxBpmOptionsHandler()