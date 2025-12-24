"""Config flow for OpenAI Compatible TTS."""

from __future__ import annotations

import logging
from typing import Any

import aiohttp
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    CONF_API_TOKEN,
    CONF_API_URL,
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
    SUPPORTED_VOICES,
)

_LOGGER = logging.getLogger(__name__)


def _schema_user(defaults: dict[str, Any] | None = None) -> vol.Schema:
    defaults = defaults or {}
    return vol.Schema(
        {
            vol.Optional(CONF_NAME, default=defaults.get(CONF_NAME, DEFAULT_NAME)): str,
            vol.Required(CONF_API_URL, default=defaults.get(CONF_API_URL, "")): str,
            vol.Required(CONF_API_TOKEN, default=defaults.get(CONF_API_TOKEN, "")): str,
            vol.Optional(
                CONF_DEFAULT_LANGUAGE, default=defaults.get(CONF_DEFAULT_LANGUAGE, DEFAULT_LANGUAGE)
            ): str,
            vol.Optional(CONF_MODEL, default=defaults.get(CONF_MODEL, DEFAULT_MODEL)): str,
            vol.Optional(CONF_VOICE, default=defaults.get(CONF_VOICE, DEFAULT_VOICE)): vol.In(
                SUPPORTED_VOICES
            ),
            vol.Optional(CONF_SPEED, default=defaults.get(CONF_SPEED, DEFAULT_SPEED)): vol.Coerce(
                float
            ),
            vol.Optional(
                CONF_TIMEOUT, default=defaults.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)
            ): vol.Coerce(int),
        }
    )


def _schema_options(current: dict[str, Any]) -> vol.Schema:
    return vol.Schema(
        {
            vol.Optional(CONF_MODEL, default=current.get(CONF_MODEL, DEFAULT_MODEL)): str,
            vol.Optional(CONF_VOICE, default=current.get(CONF_VOICE, DEFAULT_VOICE)): vol.In(
                SUPPORTED_VOICES
            ),
            vol.Optional(CONF_SPEED, default=current.get(CONF_SPEED, DEFAULT_SPEED)): vol.Coerce(
                float
            ),
            vol.Optional(CONF_TIMEOUT, default=current.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)): vol.Coerce(
                int
            ),
        }
    )


async def _async_test_connection(
    session: aiohttp.ClientSession,
    url: str,
    token: str,
    model: str,
    voice: str,
    speed: float,
    timeout_s: int,
) -> None:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
        "Accept-Encoding": "identity",
        "User-Agent": "enSmart-TTS",
    }
    payload = {
        "model": model,
        "input": "Home Assistant kapcsolat teszt.",
        "voice": voice,
        "response_format": "mp3",
        "speed": speed,
    }

    async with session.post(
        url,
        json=payload,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=timeout_s),
    ) as resp:
        if resp.status in (401, 403):
            raise PermissionError("unauthorized")
        resp.raise_for_status()

        # We don't need to consume the whole audio; just ensure we got something.
        data = await resp.read()
        if not data:
            raise ConnectionError("empty response")


class OpenAICompatTTSConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for OpenAI Compatible TTS."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None):
        errors: dict[str, str] = {}

        if user_input is not None:
            # Basic normalization
            user_input = dict(user_input)
            user_input[CONF_API_URL] = user_input[CONF_API_URL].strip()
            user_input[CONF_API_TOKEN] = user_input[CONF_API_TOKEN].strip()

            session = async_get_clientsession(self.hass)
            try:
                await _async_test_connection(
                    session=session,
                    url=user_input[CONF_API_URL],
                    token=user_input[CONF_API_TOKEN],
                    model=str(user_input.get(CONF_MODEL, DEFAULT_MODEL)),
                    voice=str(user_input.get(CONF_VOICE, DEFAULT_VOICE)),
                    speed=float(user_input.get(CONF_SPEED, DEFAULT_SPEED)),
                    timeout_s=int(user_input.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)),
                )
            except PermissionError:
                errors["base"] = "invalid_auth"
            except (TimeoutError, aiohttp.ClientError, ConnectionError):
                errors["base"] = "cannot_connect"
            except Exception:  # noqa: BLE001
                _LOGGER.exception("Unexpected error testing TTS endpoint")
                errors["base"] = "unknown"

            if not errors:
                name = user_input.pop(CONF_NAME, DEFAULT_NAME)
                await self.async_set_unique_id(f"{DOMAIN}:{user_input[CONF_API_URL]}")
                self._abort_if_unique_id_configured()

                return self.async_create_entry(title=name, data=user_input)

        return self.async_show_form(
            step_id="user",
            data_schema=_schema_user(user_input),
            errors=errors,
        )

    @staticmethod
    def async_get_options_flow(config_entry: config_entries.ConfigEntry):
        return OpenAICompatTTSOptionsFlowHandler(config_entry)


class OpenAICompatTTSOptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self.config_entry = config_entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None):
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        current = dict(self.config_entry.options)
        # Also allow editing values originally stored in data (common on first setup)
        for k in (CONF_MODEL, CONF_VOICE, CONF_SPEED, CONF_TIMEOUT):
            if k not in current and k in self.config_entry.data:
                current[k] = self.config_entry.data[k]

        return self.async_show_form(
            step_id="init",
            data_schema=_schema_options(current),
        )
