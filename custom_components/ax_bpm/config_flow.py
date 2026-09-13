"""Config flow for AX BPM (UI only, no YAML).

The legacy genre_correction / mood_correction toggle pair is replaced by
ONE `octave_disambiguation` dropdown (Off / Genre only / Genre + mood)
plus an optional manual `analyzer_url` override.

HA-native discovery: the AX BPM Analyzer add-on announces itself to the
Supervisor, which routes the flow to `async_step_hassio` — the discovered
host/port is stored as `discovered_analyzer_url` so the client can reach
the add-on without guessing its hostname. That key is INTERNAL: it is
never rendered as a form field.

Legacy entries are migrated on the next options save (or entry reload);
v1 → v2 key renames are handled by `async_migrate_entry` in __init__.py.
"""

from __future__ import annotations

import logging

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers import aiohttp_client, selector

from .analyzer_client import AnalyzerClient
from .const import (
    CONF_ANALYZER_API_TOKEN,
    CONF_ANALYZER_URL,
    CONF_DISCOVERED_ANALYZER_URL,
    CONF_MEDIA_PLAYER,
    CONF_OCTAVE_DISAMBIGUATION,
    DOMAIN,
    NAME,
    OCTAVE_GENRE_MOOD,
    OCTAVE_GENRE_ONLY,
    OCTAVE_MODES,
    OCTAVE_OFF,
)

_LOGGER = logging.getLogger(__name__)


def _migrate_legacy_toggles(merged: dict) -> str:
    """Derive the dropdown value from legacy genre/mood toggles.

    genre+mood → "Genre + mood"; genre-only → "Genre only"; both off →
    "Off". A stored dropdown value always wins over the legacy toggles.
    """
    stored = merged.get(CONF_OCTAVE_DISAMBIGUATION)
    if stored in OCTAVE_MODES:
        return stored
    genre = merged.get("genre_correction", True)
    mood = merged.get("mood_correction", False)
    if genre and mood:
        return OCTAVE_GENRE_MOOD
    if genre:
        return OCTAVE_GENRE_ONLY
    return OCTAVE_OFF


def _build_schema(defaults: dict, analyzer_detected: bool) -> vol.Schema:
    """Build the shared config/options schema.

    "Genre + mood" is disabled with helper text when no analyzer was
    detected; the user can still select it (the pipeline degrades to
    genre-only at runtime), but the UI makes the state visible.

    `discovered_analyzer_url` is deliberately NOT part of the schema —
    it is internal plumbing written by the discovery flow.
    """
    octave_default = defaults.get(
        CONF_OCTAVE_DISAMBIGUATION, OCTAVE_GENRE_ONLY
    )
    schema = {
        vol.Required(
            CONF_MEDIA_PLAYER, default=defaults.get(CONF_MEDIA_PLAYER, "")
        ): selector.EntitySelector(
            selector.EntitySelectorConfig(domain="media_player")
        ),
        vol.Required(
            CONF_OCTAVE_DISAMBIGUATION, default=octave_default
        ): selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=OCTAVE_MODES,
                mode=selector.SelectSelectorMode.DROPDOWN,
                translation_key=CONF_OCTAVE_DISAMBIGUATION,
            )
        ),
        vol.Optional(
            CONF_ANALYZER_URL,
            default=defaults.get(CONF_ANALYZER_URL, ""),
        ): str,
        vol.Optional(
            CONF_ANALYZER_API_TOKEN,
            default=defaults.get(CONF_ANALYZER_API_TOKEN, ""),
        ): str,
    }
    return vol.Schema(schema)


def _status_line(analyzer_detected: bool) -> str:
    """Connection status line shown on the config/options form."""
    if analyzer_detected:
        return "AX BPM Analyzer add-on detected."
    return (
        "AX BPM Analyzer add-on not detected — 'Genre + mood' will "
        "degrade to genre-only until it is reachable."
    )


def _clean(user_input: dict) -> dict:
    """Drop empty optional strings so they are not persisted."""
    return {k: v for k, v in user_input.items() if v not in ("", None)}


class AxBpmOptionsHandler(config_entries.OptionsFlow):
    """Handle options for AX BPM."""

    async def async_step_init(self, user_input=None):
        if user_input is not None:
            return self.async_create_entry(title="", data=_clean(user_input))
        current = {**self.config_entry.data, **self.config_entry.options}
        analyzer = await self._probe_analyzer(current)
        return self.async_show_form(
            step_id="init",
            data_schema=_build_schema(current, analyzer),
            description_placeholders={
                "analyzer_status": _status_line(analyzer),
            },
        )

    async def _probe_analyzer(self, current: dict) -> bool:
        """One-shot health probe for the connection status line."""
        client = AnalyzerClient(
            aiohttp_client.async_get_clientsession(self.hass),
            current.get(CONF_ANALYZER_URL),
            discovered_url=current.get(CONF_DISCOVERED_ANALYZER_URL),
        )
        return await client.async_detect() is not None


class AxBpmConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Config flow for AX BPM."""

    VERSION = 2

    def __init__(self) -> None:
        self._discovered_url: str | None = None

    async def async_step_user(self, user_input=None):
        if user_input is not None:
            data = _clean(user_input)
            if self._discovered_url:
                data[CONF_DISCOVERED_ANALYZER_URL] = self._discovered_url
            return self.async_create_entry(title=NAME, data=data)

        analyzer = await self._probe_analyzer({})
        return self.async_show_form(
            step_id="user",
            data_schema=_build_schema({}, analyzer),
            description_placeholders={
                "analyzer_status": _status_line(analyzer),
            },
        )

    async def async_step_hassio(self, discovery_info=None):
        """Handle HA-native discovery from the AX BPM Analyzer add-on.

        The Supervisor routes the add-on's discovery announcement here
        (service "ax_bpm"). `discovery_info` is a HassioServiceInfo whose
        `.config` carries the add-on's real hostname + port.

        We record the discovered URL and show the normal setup form with
        `step_id="user"` — HA routes the user's submission to
        `async_step_user`, which injects the recorded URL into the entry.
        """
        config = getattr(discovery_info, "config", None) or {}
        host = config.get("host")
        port = config.get("port")
        if not host or not port:
            _LOGGER.warning(
                "AX BPM: discovery announcement carried no host/port — "
                "falling back to the manual setup form"
            )
            return await self.async_step_user()

        self._discovered_url = f"http://{host}:{port}"
        _LOGGER.info("AX BPM: analyzer discovered at %s", self._discovered_url)

        return self.async_show_form(
            step_id="user",
            data_schema=_build_schema({}, True),
            description_placeholders={
                "analyzer_status": _status_line(True),
            },
        )

    async def _probe_analyzer(self, current: dict) -> bool:
        """One-shot health probe for the connection status line."""
        client = AnalyzerClient(
            aiohttp_client.async_get_clientsession(self.hass),
            current.get(CONF_ANALYZER_URL),
            discovered_url=(
                current.get(CONF_DISCOVERED_ANALYZER_URL)
                or self._discovered_url
            ),
        )
        return await client.async_detect() is not None

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return AxBpmOptionsHandler()
