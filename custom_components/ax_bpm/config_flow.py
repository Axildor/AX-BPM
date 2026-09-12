"""Config flow for AX BPM (UI only, no YAML).

Phase 1: the legacy genre_correction / mood_correction toggle pair is
replaced by ONE `octave_disambiguation` dropdown (Off / Genre only /
Genre + mood) plus an optional manual `mood_analyzer_url` override.
Legacy entries are migrated on the next options save (or entry reload).
"""

from __future__ import annotations

import logging

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers import aiohttp_client, selector

from .const import (
    CONF_AUBIO_BINARY,
    CONF_MEDIA_PLAYER,
    CONF_MOOD_ANALYZER_URL,
    CONF_MOOD_API_TOKEN,
    CONF_OCTAVE_DISAMBIGUATION,
    DOMAIN,
    NAME,
    OCTAVE_GENRE_MOOD,
    OCTAVE_GENRE_ONLY,
    OCTAVE_MODES,
    OCTAVE_OFF,
)
from .mood_client import MoodClient

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


def _build_schema(defaults: dict, sidecar_detected: bool) -> vol.Schema:
    """Build the shared config/options schema.

    "Genre + mood" is disabled with helper text when no sidecar was
    detected; the user can still select it (the pipeline degrades to
    genre-only at runtime), but the UI makes the state visible.
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
            CONF_MOOD_ANALYZER_URL,
            default=defaults.get(CONF_MOOD_ANALYZER_URL, ""),
        ): str,
        vol.Optional(
            CONF_AUBIO_BINARY, default=defaults.get(CONF_AUBIO_BINARY, "")
        ): str,
        vol.Optional(
            CONF_MOOD_API_TOKEN,
            default=defaults.get(CONF_MOOD_API_TOKEN, ""),
        ): str,
    }
    return vol.Schema(schema)


class AxBpmOptionsHandler(config_entries.OptionsFlow):
    """Handle options for AX BPM."""

    async def async_step_init(self, user_input=None):
        if user_input is not None:
            # Empty optional strings → drop them.
            user_input = {
                k: v for k, v in user_input.items() if v not in ("", None)
            }
            return self.async_create_entry(title="", data=user_input)
        current = {**self.config_entry.data, **self.config_entry.options}
        sidecar = await self._probe_sidecar(current)
        return self.async_show_form(
            step_id="init",
            data_schema=_build_schema(current, sidecar),
            description_placeholders={
                "sidecar_status": (
                    "Sidecar mood analyzer detected."
                    if sidecar
                    else "Sidecar mood analyzer not detected — 'Genre + mood' "
                    "will degrade to genre-only until it is reachable."
                ),
            },
        )

    async def _probe_sidecar(self, current: dict) -> bool:
        """One-shot health probe for the connection status line."""
        client = MoodClient(
            aiohttp_client.async_get_clientsession(self.hass),
            current.get(CONF_MOOD_ANALYZER_URL),
        )
        return await client.async_detect() is not None


class AxBpmConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Config flow for AX BPM."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        errors = {}
        if user_input is not None:
            # Empty optional strings → drop them.
            user_input = {
                k: v for k, v in user_input.items() if v not in ("", None)
            }
            return self.async_create_entry(title=NAME, data=user_input)

        sidecar = await self._probe_sidecar({})
        return self.async_show_form(
            step_id="user",
            data_schema=_build_schema({}, sidecar),
            errors=errors,
            description_placeholders={
                "sidecar_status": (
                    "Sidecar mood analyzer detected."
                    if sidecar
                    else "Sidecar mood analyzer not detected — 'Genre + mood' "
                    "will degrade to genre-only until it is reachable."
                ),
            },
        )

    async def _probe_sidecar(self, current: dict) -> bool:
        """One-shot health probe for the connection status line."""
        client = MoodClient(
            aiohttp_client.async_get_clientsession(self.hass),
            current.get(CONF_MOOD_ANALYZER_URL),
        )
        return await client.async_detect() is not None

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return AxBpmOptionsHandler()