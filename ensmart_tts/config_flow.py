"""Config flow for enSmart TTS."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_NAME

from .const import (
    CONF_DEFAULT_LANGUAGE,
    CONF_MODEL,
    CONF_SPEED,
    CONF_TIMEOUT,
    CONF_VOICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_NAME,
    DEFAULT_SPEED,
    DEFAULT_TIMEOUT,
    DEFAULT_VOICE,
    DOMAIN,
    SUPPORTED_VOICES_ALL,
    VOICE_SERVER_DEFAULT,
)


def _schema_user(defaults: dict[str, Any]) -> vol.Schema:
    return vol.Schema(
        {
            vol.Optional(CONF_NAME, default=defaults.get(CONF_NAME, DEFAULT_NAME)): str,
        }
    )


def _schema_options(current: dict[str, Any]) -> vol.Schema:
    return vol.Schema(
        {
            vol.Optional(CONF_MODEL, default=current.get(CONF_MODEL, DEFAULT_MODEL)): str,
            vol.Optional(CONF_VOICE, default=current.get(CONF_VOICE, DEFAULT_VOICE)): vol.In(
                SUPPORTED_VOICES_ALL
            ),
            vol.Optional(CONF_SPEED, default=float(current.get(CONF_SPEED, DEFAULT_SPEED))): vol.Coerce(
                float
            ),
            vol.Optional(CONF_TIMEOUT, default=int(current.get(CONF_TIMEOUT, DEFAULT_TIMEOUT))): vol.Coerce(
                int
            ),
            vol.Optional(
                CONF_DEFAULT_LANGUAGE, default=current.get(CONF_DEFAULT_LANGUAGE, DEFAULT_LANGUAGE)
            ): str,
        }
    )


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for enSmart TTS."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None):
        # Single instance is enough
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

        if user_input is not None:
            title = (user_input.get(CONF_NAME) or DEFAULT_NAME).strip() or DEFAULT_NAME
            return self.async_create_entry(title=title, data={CONF_NAME: title})

        return self.async_show_form(step_id="user", data_schema=_schema_user({}))

    async def async_step_import(self, user_input: dict[str, Any] | None = None):
        # Support YAML import if ever used, but keep single instance
        return await self.async_step_user(user_input)


class OptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self.config_entry = config_entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None):
        if user_input is not None:
            # Normalize numeric fields
            user_input = dict(user_input)
            user_input[CONF_SPEED] = float(user_input[CONF_SPEED])
            user_input[CONF_TIMEOUT] = int(user_input[CONF_TIMEOUT])
            return self.async_create_entry(title="", data=user_input)

        current = dict(self.config_entry.options)
        # If first setup stored things in data, inherit them
        for k in (CONF_MODEL, CONF_VOICE, CONF_SPEED, CONF_TIMEOUT, CONF_DEFAULT_LANGUAGE):
            if k not in current and k in self.config_entry.data:
                current[k] = self.config_entry.data[k]

        return self.async_show_form(step_id="init", data_schema=_schema_options(current))


async def async_get_options_flow(config_entry: config_entries.ConfigEntry):
    return OptionsFlowHandler(config_entry)
