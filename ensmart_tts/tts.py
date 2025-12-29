"""TTS platform for the enSmart cloud endpoint.

- The API URL is fixed: https://api.energysmarthome.cloud/ai/openai/tts
- The Bearer token is read live from: input_text.ensmart_http_api_token

Why:
- You can rotate tokens without reconfiguring the integration.
- Assist UI won't get stuck on an empty 'voice' selector when you use server-side default voice.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import aiohttp
import asyncio
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from homeassistant.components.tts import TextToSpeechEntity

try:
    # Newer HA has a Voice dataclass (with variants)
    from homeassistant.components.tts.models import Voice as HAVoice  # type: ignore
except Exception:  # pragma: no cover
    HAVoice = None  # type: ignore[assignment]

from .const import (
    API_URL,
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
    SUPPORTED_VOICES_ALL,
    TOKEN_ENTITY_ID,
    VOICE_SERVER_DEFAULT,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry, async_add_entities) -> None:
    async_add_entities([EnsmartTextToSpeechEntity(hass, entry)])


def _read_bearer_token(hass: HomeAssistant) -> str:
    state = hass.states.get(TOKEN_ENTITY_ID)
    if state is None:
        raise HomeAssistantError(
            f"Missing {TOKEN_ENTITY_ID}. Create it in Helpers (Input text) and paste the token."
        )

    token = (state.state or "").strip()
    if not token or token.lower() in ("unknown", "unavailable"):
        raise HomeAssistantError(
            f"{TOKEN_ENTITY_ID} is empty. Paste your bearer token into this input_text helper."
        )
    return token


class EnsmartTextToSpeechEntity(TextToSpeechEntity):
    """Represent the enSmart TTS entity."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, hass: HomeAssistant, entry) -> None:
        self.hass = hass
        self._entry_id = entry.entry_id
        self._name = entry.data.get(CONF_NAME, DEFAULT_NAME)

        # Defaults (can be overridden by per-call options)
        opts = hass.data.get(DOMAIN, {}).get(entry.entry_id, {})
        self._model = opts.get(CONF_MODEL, DEFAULT_MODEL)
        self._voice = opts.get(CONF_VOICE, DEFAULT_VOICE)
        self._speed = float(opts.get(CONF_SPEED, DEFAULT_SPEED))
        self._timeout = int(opts.get(CONF_TIMEOUT, DEFAULT_TIMEOUT))
        self._default_language = opts.get(CONF_DEFAULT_LANGUAGE, DEFAULT_LANGUAGE)

        self._attr_unique_id = f"{DOMAIN}_{entry.entry_id}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def supported_languages(self) -> list[str]:
        # Your endpoint can likely speak many languages, but Assist uses this list
        # to decide what it can request. Keep Hungarian as the primary default.
        return [self._default_language]

    @property
    def default_language(self) -> str:
        return self._default_language

    @property
    def supported_options(self) -> list[str] | None:
        # If we let the server decide the voice, don't advertise voice as an option,
        # so Assist won't force selecting one.
        if self._voice == VOICE_SERVER_DEFAULT:
            return None
        return [CONF_VOICE]

    @property
    def default_options(self) -> Mapping[str, Any] | None:
        # Keep these as integration-level defaults
        return {
            CONF_MODEL: self._model,
            CONF_SPEED: self._speed,
            CONF_TIMEOUT: self._timeout,
            # Only include voice if not server default
            **({CONF_VOICE: self._voice} if self._voice != VOICE_SERVER_DEFAULT else {}),
        }

    @callback
    def async_get_supported_voices(self, language: str) -> list[Any] | None:
        # Return Voice objects on new HA, or strings on old HA
        if HAVoice is not None:
            return [
                HAVoice(voice_id=v, name=("Server default" if v == VOICE_SERVER_DEFAULT else v), variants=[])
                for v in SUPPORTED_VOICES_ALL
            ]
        return SUPPORTED_VOICES_ALL

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ):
        token = _read_bearer_token(self.hass)

        model = options.get(CONF_MODEL, self._model)
        voice = options.get(CONF_VOICE, self._voice)
        speed = float(options.get(CONF_SPEED, self._speed))
        timeout = int(options.get(CONF_TIMEOUT, self._timeout))

        payload: dict[str, Any] = {
            "model": model,
            "input": message,
            "speed": speed,
        }
        if voice and voice != VOICE_SERVER_DEFAULT:
            payload["voice"] = voice

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        session = async_get_clientsession(self.hass)
        try:
            async with session.post(
                API_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status >= 400:
                    # try to extract useful error
                    content_type = (resp.headers.get("Content-Type") or "").lower()
                    body = await resp.text()
                    raise HomeAssistantError(
                        f"TTS request failed ({resp.status}). {body[:500]}"
                        if "json" in content_type or body
                        else f"TTS request failed ({resp.status})."
                    )

                audio = await resp.read()

        except asyncio.TimeoutError as err:  # type: ignore[name-defined]
            raise HomeAssistantError(f"TTS request timed out after {timeout}s") from err
        except aiohttp.ClientError as err:
            raise HomeAssistantError(f"TTS request error: {err}") from err

        # Assume mp3 unless your API says otherwise
        return "mp3", audio
